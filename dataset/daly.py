import torch
from torch.utils import data
import torchvision.transforms as transforms
import PIL
from PIL import Image
import numpy as np
import pickle
import os
import copy

def load_tracks(load_path='../DALY/'):
    print('Loading labeled tracks...')
            
    with open(os.path.join(load_path, 'annotated_data.pkl'), 'rb') as f:
        annot_data = pickle.load(f)

    print('Finished.')
    return annot_data
      
def is_next_video(video_labels, classes_to_exclude, split):
    next_video = False
    if split == 'training' or split == 'validation':
        for label_name in video_labels:
            if label_name in classes_to_exclude:
                next_video = True
    elif split == 'test':
        for label_name in video_labels:
            if label_name not in classes_to_exclude:
                next_video = True
    return next_video

def get_frames(annot_data, cfg, split='training', on_keyframes=False):
    
    if cfg.zero_shot:
        with open(os.path.join(cfg.annot_path, 'daly1.1.0.pkl'), 'rb') as f:
            annot = pickle.load(f, encoding='latin1')
    
    ######################################
    available_videos = os.listdir(cfg.data_path)
    ######################################

    if split == 'training':
        frames = []
        for i, video in enumerate(annot_data[split]):
            ########################################
            if video not in available_videos:
                continue
            ########################################
            if cfg.zero_shot:
                video_labels = list(annot['annot'][video]['annot'].keys())
                next_video = is_next_video(video_labels, cfg.classes_to_exclude, split)
                if next_video:
                    continue
            vid_annot = annot_data[split][video]
            for instance in vid_annot['action_instances']:
                frames.append((video, instance, split))
        return frames  
    else:
        if on_keyframes:
            frames = []
            for video in annot_data[split]:
                if cfg.zero_shot:
                    video_labels = list(annot['annot'][video]['annot'].keys())
                    next_video = is_next_video(video_labels, cfg.classes_to_exclude, split)
                    if next_video:
                        continue
                vid_annot = annot_data[split][video]
                for instance in vid_annot['action_instances']:
                    for keyframe in vid_annot['action_instances'][instance]['keyframes']:
                        frames.append((video, instance, keyframe, split))
            return frames
        else: 
            if split == 'validation' or split == 'test':
                np.random.seed(1001) # always get the same samples across different models
                num_frames = 10
                frames = []
                for video in annot_data[split]:
                    ########################################
                    if video not in available_videos:
                        continue
                    ########################################
                    if cfg.zero_shot:
                        video_labels = list(annot['annot'][video]['annot'].keys())
                        next_video = is_next_video(video_labels, cfg.classes_to_exclude, split)
                        if next_video:
                            continue
                    vid_annot = annot_data[split][video]
                    for instance in vid_annot['action_instances']:
                        instance_frames = list(vid_annot['action_instances'][instance]['frames'].keys())
                        if len(instance_frames) <= num_frames:
                            for frame_num in instance_frames:
                                frames.append((video, instance, frame_num, split))
                        else:
                            sample_frames = np.random.choice(instance_frames, size=num_frames, replace=False)
                            for frame_num in sample_frames:
                                frames.append((video, instance, int(frame_num), split))
                return frames

class DALYDataset(data.Dataset):
    
    def __init__(self, 
                 annot_data, 
                 frames, 
                 cfg,
                 split='training'):
        
        self.cfg = cfg
        self.annot_data = annot_data
        self.frames = frames
        self.data_path = cfg.data_path
        self.img_size = cfg.img_size
        self.out_feature_size = cfg.out_feature_size
        self.num_person_boxes = cfg.num_person_boxes
        self.num_in_frames = cfg.num_in_frames 
        self.split = split
    
    def __len__(self):
        
        return len(self.frames)
    
    def __getitem__(self, index):
                
        frame = self.frames[index]
        split = frame[-1]
        
        if split == 'training':            
            frames_to_return = []
            video_name = frame[0]
            instance = frame[1]
            frames_list = list(self.annot_data[split][video_name]['action_instances'][instance]['frames'].keys())
            sample_frames = np.random.choice(frames_list, size=1, replace=False)
            frames_to_return = [video_name, instance, int(sample_frames[0])]
            sample = self.load_sample(frames_to_return)
        elif split == 'validation' or split == 'test':
            sample = self.load_sample(frame)
        return sample
    
    def load_sample(self, frame):

        OH, OW = self.out_feature_size # output feature map size of I3D
        
        imgs = []
        video_name = frame[0]
        instance = frame[1]
        center_frame = frame[2]        
        
        instance_annot = self.annot_data[self.split][video_name]['action_instances'][instance]
        W, H = self.annot_data[self.split][video_name]['(width, height)']
        
        self.tbound = self.annot_data[self.split][video_name]['action_instances'][instance]['tbound'] # temporal bound of action instance
        self.tbound_seq = [frame_num for frame_num in range(self.tbound[0], self.tbound[1] + 1)]
        
        num_frames_after = int(self.num_in_frames / 2)
        num_frames_before = num_frames_after - 1
        self.clip_tbound = (center_frame - num_frames_before, center_frame + num_frames_after) # temporal bound of clip
        
        self.clip = [frame_num for frame_num in range(self.clip_tbound[0], self.clip_tbound[1] + 1)]
        
        person_boxes = []
        action_labels = []
        num_boxes_per_frame = []
        for idx, frame_num in enumerate(self.clip):
            
            person_boxes_frame = []
            action_labels_frame = []
            
            if frame_num in self.tbound_seq:
                img = Image.open(os.path.join(self.data_path, video_name, 'frame' + str(frame_num).zfill(6) + '.jpg'))
            else:
                # use first/last frame when clip goes out of action instance bounds
                if frame_num < center_frame:
                    img = Image.open(os.path.join(self.data_path, video_name, 'frame' + str(self.tbound_seq[0]).zfill(6) + '.jpg'))
                else:     
                    img = Image.open(os.path.join(self.data_path, video_name, 'frame' + str(self.tbound_seq[-1]).zfill(6) + '.jpg'))
            
            #img = img.resize((self.img_size[1], self.img_size[0]), PIL.Image.ANTIALIAS)
            img = np.array(img)
            imgs.append(img)
            
            if frame_num in self.tbound_seq:
                action_frame_num = frame_num 
            else: 
                assert frame_num != center_frame # center frame is always in tbound_seq
                if frame_num < center_frame:
                    action_frame_num = self.tbound_seq[0]
                else: 
                    action_frame_num = self.tbound_seq[-1]

            person_boxes_frame = np.copy(instance_annot['frames'][action_frame_num])
            person_boxes_frame[:, [0, 2]] = (np.copy(person_boxes_frame[:, [0, 2]]) / W) * OW # rescale box width to feature map size
            person_boxes_frame[:, [1, 3]] = (np.copy(person_boxes_frame[:, [1, 3]]) / H) * OH # rescale box height to feature map size
            #assert np.sum(person_boxes_frame > 14) == 0, print(video_name, instance, center_frame)

            num_boxes_per_frame.append(len(person_boxes_frame))                 
            assert len(person_boxes_frame) <= 20, print(video_name, instance, center_frame)
            while len(person_boxes_frame) != self.num_person_boxes:
                person_boxes_frame = np.vstack((person_boxes_frame, np.array([0, 0, 0, 0])))
            
            if self.cfg.zero_shot:
                if (self.split == 'training') or (self.split == 'validation'):
                    action_labels_frame = []
                    tube_labelnames = np.copy(np.array(instance_annot['tube_labelnames']))
                    for label_name in tube_labelnames:
                        action_labels_frame.append(self.cfg.class_map[label_name])
                elif self.split == 'test':
                    tube_labelnames = np.copy(np.array(instance_annot['tube_labelnames']))
                    action_labels_frame = [-1] * self.num_person_boxes
            else:
                action_labels_frame = np.copy(np.array(instance_annot['tube_labels']))
            while len(action_labels_frame) != self.num_person_boxes:
                action_labels_frame = np.hstack((action_labels_frame, -1))
            
            person_boxes.append(person_boxes_frame)
            action_labels.append(action_labels_frame)
        
        # convert images to tensor
        imgs = np.stack(imgs, axis=0)
        imgs = (imgs / 255.) * 2 - 1 # normalize images
        imgs = torch.from_numpy(imgs.transpose([3, 0, 1, 2])).float()
        
        num_boxes_per_frame = torch.IntTensor(num_boxes_per_frame)
        
        # convert person boxes to tensor
        person_boxes = np.vstack(person_boxes)
        person_boxes = torch.from_numpy(person_boxes)
        person_boxes = torch.reshape(person_boxes, (-1, self.num_person_boxes, 4)).float()
        
        # convert action labels to tensor
        action_labels = np.vstack(action_labels)
        action_labels = torch.from_numpy(action_labels)
        action_labels = torch.reshape(action_labels, (-1, self.num_person_boxes)).long()
        
        batch = (imgs, 
                 person_boxes, 
                 action_labels, 
                 num_boxes_per_frame, 
                 video_name, 
                 instance, 
                 center_frame)
        return batch
