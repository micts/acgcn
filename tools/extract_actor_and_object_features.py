from __future__ import print_function, division
import os
import sys
import argparse
import pickle
import numpy as np
import time

import torch
import torch.nn as nn
from torch.utils import data

sys.path.append('.')

from config import config
from dataset import dataset, daly
from models import gcn_model, baseline_model
from utils import utils, eval_utils

def extract_features(dataloader, dataset, model, device, annot_data, annot):
    
    model.eval()
    
    obj_annotations = utils.get_obj_annotations(annot_data, annot)
    class_map = utils.class2idx_map(classes_to_exclude=None)
    features_dict = {}
    
    for idx, batch_data in enumerate(dataloader):
        
        imgs = batch_data[0]
        person_boxes = batch_data[1]
        action_labels = batch_data[2]
        num_boxes_per_frame = batch_data[3]
        video_names = batch_data[4]
        instances = batch_data[5]
        center_frames = batch_data[6]
        
        video_name = video_names[0]
        instance = instances[0].item()
        keyframe = center_frames[0].item()
            
        if video_name not in obj_annotations[dataset.split]:
            continue
        if instance not in obj_annotations[dataset.split][video_name]['action_instances']:
            continue
        if keyframe not in obj_annotations[dataset.split][video_name]['action_instances'][instance]:
            continue
       
        num_actors_list = [num_boxes_per_frame[b][15].item() for b in range(imgs.shape[0])]
        
        batch = [data.to(device=device) for data in [imgs, person_boxes]]
        batch.append(num_boxes_per_frame)
        batch.append(action_labels)

        with torch.set_grad_enabled(False):
            actor_features_emb_list, context_features_list = model(batch, 'return_features')
        
        for graph_num in range(len(actor_features_emb_list)):
            if graph_num not in features_dict:
                features_dict[graph_num] = []
            actor_features_emb = actor_features_emb_list[graph_num].detach().cpu().numpy()    
            tube_labels = np.copy(annot_data[dataset.split][video_name]['action_instances'][instance]['tube_labels'])
            for tube_id, tube_label in enumerate(tube_labels):
                if tube_label > 0: # not background
                    features_dict[graph_num].append([actor_features_emb[tube_id, :], utils.idx2class(class_map, tube_label)])
        
        vid_annot = obj_annotations[dataset.split][video_name]
        for box_idx in range(len(vid_annot['action_instances'][instance][keyframe])):
            obj_box = vid_annot['action_instances'][instance][keyframe][box_idx][0:4]
            obj_box = obj_box * 14
            x1 = int(round(obj_box[0]))
            y1 = int(round(obj_box[1]))
            x2 = int(round(obj_box[2]))
            y2 = int(round(obj_box[3]))
            if x1 == x2:
                x1 = int(np.floor(obj_box[0]))
                x2 = int(np.ceil(obj_box[2]))
            if y1 == y2:
                y1 = int(np.floor(obj_box[1]))
                y2 = int(np.ceil(obj_box[3]))
            for graph_num in range(len(context_features_list)):
                obj_features = context_features_list[graph_num][0, :, 3, y1:y2 + 1, x1:x2 + 1].detach().cpu().numpy()
                obj_features = np.mean(obj_features, axis=(1, 2))
                obj_id = int(vid_annot['action_instances'][instance][keyframe][box_idx][4])
                obj_name = annot['objectList'][obj_id]
                features_dict[graph_num].append([obj_features, obj_name])
    
    return features_dict
    
if __name__ == '__main__':
   
    parser = argparse.ArgumentParser()

    parser.add_argument('--annot_path', default='../data/DALY/annotations/', help='Path to annotations folder')
    parser.add_argument('--config_path', required=True, help='Path to model\'s config file')
    parser.add_argument('--model_path', required=True, help='Path to model\'s saved weights (model checkpoint)')
    parser.add_argument('--features_path', default='../features/', help='Path to save actor and object features')
    parser.add_argument('--gpu_device', type=str, default=0, help='GPU device (number) to use; defaults to 0')
    parser.add_argument('--cpu', action='store_true', help='Whether to use CPU instead of GPU; this option overwrites the --gpu_device argument')
    parser.add_argument('--split', default='test', help='Dataset split; possible values are \'training\', \'validation\', \'test\'')

    args = parser.parse_args() 

    with open(os.path.join(args.config_path, 'config.pkl'), 'rb') as f:
        cfg_dict = pickle.load(f)
    if 'merge_function' not in cfg_dict:
        cfg_dict['merge_function'] = 'concat'

    cfg = config.GetConfig(**cfg_dict)
    utils.print_config(cfg)

    if args.cpu == False:
        use_gpu = True
    else:
        use_gpu = False

    if use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_device

    checkpoint = torch.load(args.model_path)
    model_state_dict = checkpoint['model_state_dict']

    annot_data = daly.load_tracks(load_path=args.annot_path)
    with open(os.path.join(args.annot_path, 'daly1.1.0.pkl'), 'rb') as f:
        annot = pickle.load(f, encoding='latin1')


    frames = daly.get_frames(annot_data, cfg, split=args.split, on_keyframes=False)
    dataset = daly.DALYDataset(annot_data,
                               frames,
                               cfg,
                               split=args.split)
    
    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    print('\nExtracting actor features and object features using model {}...'.format(args.model_path))

    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if cfg.model_name == 'baseline':
        model = base_model.BaseNet(cfg)
        model.load_state_dict(model_state_dict)
    elif cfg.model_name == 'gcn':
        model = gcn_model.GCNNet(cfg)
        model.load_state_dict(model_state_dict)

    start_time = time.time()
    model = model.to(device=device)
    features_dict = extract_features(dataloader, dataset, model, device, annot_data, annot)

    model_features_path = os.path.join(args.features_path, cfg.model_name, cfg.filename, args.split)
    utils.make_dirs(model_features_path)
    epoch_nr = args.model_path.rsplit('/')[-1].rsplit('_', 1)[0]
    with open(os.path.join(model_features_path, 'features_{}.pkl'.format(epoch_nr)), 'wb') as f:
        pickle.dump(features_dict, f)

    end_time = time.time() - start_time
    print('Completed in {:.0f}m {:.0f}s'.format(end_time // 60, end_time % 60))


