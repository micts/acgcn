import torch
from torch.utils import data
import torchvision.transforms as transforms
import PIL
from PIL import Image
import numpy as np
import pickle
import os
import copy
import utils

def gen_labels_keyframes(label_tracks=False, load_path='/project/DALY', save_path=None):
        
    if label_tracks:
        print('Generating labels for tracks...')
        
        with open('/project/DALY/all_videos.pkl', 'rb') as f:
            all_videos = pickle.load(f)
        
        with open('/project/DALY/daly1.1.0.pkl', 'rb') as f:
            annot = pickle.load(f, encoding='latin1')
        
        with open('/project/DALY/humantubes_DALY.pkl', 'rb') as f:
            tubes = pickle.load(f, encoding='latin1')
        
        with open('/project/DALY/validation_videos.pkl', 'rb') as f:
            validation_videos = pickle.load(f)
        
        with open('/project/DALY/frames_size.pkl', 'rb') as f:
            frames_size = pickle.load(f)
        
        action_dict = {}
        for idx, video in enumerate(all_videos):
            vid_annot = annot['annot'][video]['annot']
            # correct inconsistencies between humantubes_DALY.pkl and daly1.1.0.pkl
            if video == 'TnXBUmIYmyI.mp4':
                actions = ['FoldingTextile', 'CleaningFloor']
            elif video == 'sjgkGNQyOWg.mp4':
                actions = ['TakingPhotosOrVideos', 'CleaningWindows', 'CleaningFloor']
            else:
                actions = [action for action in vid_annot]
            action_dict[video] = actions       

        iou_d = {}
        ious_ind = []
        annot_data = {}
        annot_data['training'] = {}
        annot_data['validation'] = {}
        annot_data['test'] = {}
        for idx, video in enumerate(all_videos):

            if video in annot['splits'][0]: # test videos
                split = 'test'
            elif video in validation_videos:
                split = 'validation'
            else:
                split = 'training'

            video_id = video.split('.')[0]
            vid_annot = annot['annot'][video]['annot']

            W, H = frames_size[video]

            annot_data[split][video] = {}
            annot_data[split][video]['(width, height)'] = (W, H)
            annot_data[split][video]['action_instances'] = {}

            actions = action_dict[video]
            
            instance_idx = 0
            for action_idx, action in enumerate(actions):
                annot_instance_idx = 0
                for instance in vid_annot[action]:
                    if (instance['flags']['isReflection'] == True) or (instance['flags']['isAmbiguous'] == True):
                        annot_instance_idx += 1
                        continue
                        
                    annot_data[split][video]['action_instances'][instance_idx] = {}
                    annot_data[split][video]['action_instances'][instance_idx]['frames'] = {}
                    annot_data[split][video]['action_instances'][instance_idx]['tubes'] = {}
                    annot_data[split][video]['action_instances'][instance_idx]['tube_labels'] = []
                    annot_data[split][video]['action_instances'][instance_idx]['tube_labelnames'] = []
                    annot_data[split][video]['action_instances'][instance_idx]['tube_ious'] = []
                    annot_data[split][video]['action_instances'][instance_idx]['keyframes'] = {}
                    tubes_instance = tubes[0][video][instance_idx]
                    
                    start_frame_tubes = int(tubes_instance[0][0, 0]) # first frame of tube/instance
                    end_frame_tubes = int(tubes_instance[0][-1, 0]) # last frame of tube/instance
                    begin_time = vid_annot[action][annot_instance_idx]['beginTime']
                    end_time = vid_annot[action][annot_instance_idx]['endTime']
                    annot_data[split][video]['action_instances'][instance_idx]['time_bound'] = (begin_time, end_time)
                    annot_data[split][video]['action_instances'][instance_idx]['tbound'] = (start_frame_tubes, end_frame_tubes)
                    annot_data[split][video]['action_instances'][instance_idx]['label_name'] = action
                    annot_data[split][video]['action_instances'][instance_idx]['label'] = utils.class2idx(action)
                    
                    annot_keylist = vid_annot[action][annot_instance_idx]['keyframes']
                    keyframes = []
                    for annot_keyframe in annot_keylist:
                        keyframes.append(annot_keyframe['frameNumber'])
                        assert (annot_keyframe['frameNumber'] >= start_frame_tubes) and (annot_keyframe['frameNumber'] <= end_frame_tubes)
                        annot_data[split][video]['action_instances'][instance_idx]['keyframes'][annot_keyframe['frameNumber']] = annot_keyframe['boundingBox'][0]
                    
                    for tube_idx, tube in enumerate(tubes_instance):
                        annot_data[split][video]['action_instances'][instance_idx]['tubes'][tube_idx] = tube 
                        for frame in tube:
                            frame_num = int(frame[0])
                            if frame_num not in annot_data[split][video]['action_instances'][instance_idx]['frames']:
                                annot_data[split][video]['action_instances'][instance_idx]['frames'][frame_num] = []
                            annot_data[split][video]['action_instances'][instance_idx]['frames'][frame_num].append(frame[1:5])
                    
                        tube_iou_class = [0, 'Background']
                        for action_idx2, action2 in enumerate(actions):
                            for instance2_idx, instance2 in enumerate(vid_annot[action2]):
                                if (instance2['flags']['isReflection'] == True) or (instance2['flags']['isAmbiguous'] == True):
                                    continue
                                #vid_annot[action][instance2]
                                annot_keylist2 = vid_annot[action2][instance2_idx]['keyframes']
                                keyframes2 = []
                                keyframe_boxes2 = []
                                for annot_keyframe2 in annot_keylist2:
                                    if (annot_keyframe2['frameNumber'] >= start_frame_tubes) and (annot_keyframe2['frameNumber'] <= end_frame_tubes):
                                        keyframes2.append(annot_keyframe2['frameNumber'])
                                        keyframe_box2 = np.copy(annot_keyframe2['boundingBox'][0])
                                        keyframe_box2 = np.array([keyframe_box2[0] * W, keyframe_box2[1] * H, keyframe_box2[2] * W, keyframe_box2[3] * H])
                                        keyframe_boxes2.append(keyframe_box2)
                                if len(keyframes2) > 0:
                                    keyframe_boxes2 = np.vstack(keyframe_boxes2)
                                    keyframes2 = np.array(keyframes2)
                                    spt_iou = np.mean(utils.get_tube_iou(tube[np.in1d(tube[:, 0], keyframes2), 1:5], keyframe_boxes2))
                                    if spt_iou > 0.5:
                                        if spt_iou > tube_iou_class[0]:
                                            tube_iou_class[0] = spt_iou
                                            tube_iou_class[1] = action2
                                    else:
                                        if spt_iou > tube_iou_class[0]:
                                            tube_iou_class[0] = spt_iou
                        annot_data[split][video]['action_instances'][instance_idx]['tube_labels'].append(utils.class2idx(tube_iou_class[1]))
                        annot_data[split][video]['action_instances'][instance_idx]['tube_labelnames'].append(tube_iou_class[1])
                        annot_data[split][video]['action_instances'][instance_idx]['tube_ious'].append(tube_iou_class[0])
                                                                        
                    instance_idx += 1
                    annot_instance_idx += 1
        
        annot_data_ = copy.deepcopy(annot_data)
        splits = list(annot_data.keys())
        for split in splits:
            for video in annot_data[split]:
                instances = annot_data[split][video]['action_instances']
                for instance in instances:
                    frames = list(annot_data[split][video]['action_instances'][instance]['frames'].keys())
                    for frame in frames:
                        boxes = np.copy(np.vstack(annot_data[split][video]['action_instances'][instance]['frames'][frame]))
                        annot_data_[split][video]['action_instances'][instance]['frames'][frame] = boxes
                    labels = np.copy(np.hstack(annot_data[split][video]['action_instances'][instance]['tube_labels']))
                    annot_data_[split][video]['action_instances'][instance]['tube_labels'] = labels
    
    else:
        print('Loading labeled tracks...')
        
        with open(os.path.join(load_path, 'annotated_data.pkl'), 'rb') as f:
            annot_data_ = pickle.load(f)
        
        print('Loading finished.')

    if save_path != None:
        print('Saving dictionary under', save_path)
        with open(os.path.join(save_path, 'annotated_data.pkl'), 'wb') as f:
            pickle.dump(annot_data, f)
        print('Saving finished.')
        
    return(annot_data_)

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
    
    if split == 'training':
        frames = []
        for i, video in enumerate(annot_data[split]):
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
