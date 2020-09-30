from __future__ import print_function, division
import os
import argparse
import pickle
import numpy as np
import time

import torch
import torch.nn as nn
from torch.utils import data

import daly
import dataset
import config
from models import baseline_model
from models import gcn_model
import utils
import eval_utils

def extract_features(dataloader, dataset, model, device, model_instance, annot_data, cfg):
    
    with open(os.path.join(cfg.annot_path, 'daly1.1.0.pkl'), 'rb') as f:
        annot = pickle.load(f, encoding='latin1')
    
    start_time = time.time()
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
        
        num_actors_list = []
        for b in range(imgs.shape[0]):
            num_actors_list.append(num_boxes_per_frame[b][15].item())
        
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
        
    if not os.path.exists(cfg.features_path):
        os.mkdir(cfg.features_path)
    if not os.path.exists(os.path.join(cfg.features_path, cfg.filename)):
        os.mkdir(os.path.join(cfg.features_path, cfg.filename))
    if not os.path.exists(os.path.join(cfg.features_path, cfg.filename, dataset.split)):
        os.mkdir(os.path.join(cfg.features_path, cfg.filename, dataset.split))
    with open(os.path.join(cfg.features_path, cfg.filename, dataset.split, 'features_' + model_instance.rsplit('_', 1)[0] + '.pkl'), 'wb') as f:
        pickle.dump(features_dict, f)    
            
    end_time = time.time() - start_time
    print('Completed in {:.0f}m {:.0f}s'.format(end_time // 60, end_time % 60))
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', required=True, help='model\'s filename under results/, e.g. 2020-03-15_20-12-17')
    parser.add_argument('-mi', '--model_instance', required=True, help='model\'s instance under results/filename/, e.g. epoch_300_1.158')
    parser.add_argument('-s', '--split', required=True, help='dataset split; possible values are \'training\', \'validation\', \'test\'')

    args = parser.parse_args()

    model_name = 'gcn'
    filename = args.filename
    model_instance = args.model_instance
    split = args.split

    cfg = config.Config(model_name, 0, 0, 'sum', False) # default config values
    with open(os.path.join(cfg.results_path, model_name, filename, 'config.pkl'), 'rb') as f:
        cfg_dict = pickle.load(f)

    cfg = utils.overwrite_config(cfg, cfg_dict) # overwrite default config file
    utils.print_config(cfg)

    if cfg.use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    model_path = os.path.join(cfg.results_path, model_name, filename, model_instance)
    checkpoint = torch.load(model_path)
    model_state_dict = checkpoint['model_state_dict']
    
    annot_data = daly.gen_labels_keyframes(label_tracks=False, load_path=cfg.annot_path)

    test_frames = daly.get_frames(annot_data, cfg, split=split, on_keyframes=True)
    test_set = daly.DALYDataset(annot_data, 
                                test_frames,
                                cfg,
                                split=split)

    print()
    print('Extracting actor features and object features using model', os.path.join(cfg.model_name, filename, model_instance))

    test_loader = data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4)

    if cfg.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if cfg.model_name == 'baseline':
        model = base_model.BaseNet(cfg)
        model.load_state_dict(model_state_dict)
    elif cfg.model_name == 'gcn':
        model = gcn_model.GCNNet(cfg)
        model.load_state_dict(model_state_dict)
    else:
        assert(False), 'Variable model_name should be either \'baseline\' or 2 \'gcn\'.'

    model = model.to(device=device)
    extract_features(test_loader, test_set, model, device, model_instance, annot_data, cfg)




