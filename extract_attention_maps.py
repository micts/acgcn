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

def extract_attention_maps(dataloader, dataset, model, device, model_instance, cfg):
    
    start_time = time.time()
    model.eval()

    am_dict = {}
    for idx, batch_data in enumerate(dataloader):

        imgs = batch_data[0]
        person_boxes = batch_data[1]
        action_labels = batch_data[2]
        num_boxes_per_frame = batch_data[3]
        video_names = batch_data[4]
        instances = batch_data[5]
        center_frames = batch_data[6]
        
        num_actors_list = []
        for b in range(imgs.shape[0]):
            num_actors_list.append(num_boxes_per_frame[b][15].item())
        
        batch = [data.to(device=device) for data in [imgs, person_boxes]]
        batch.append(num_boxes_per_frame)
        batch.append(action_labels)

        with torch.set_grad_enabled(False):

            _, _, am_list = model(batch)
            
        video_info = (video_names, instances.tolist(), center_frames.tolist())
        am_dict = utils.save_am(am_list, am_dict, video_info)
                
    if not os.path.exists(cfg.am_path):
        os.mkdir(cfg.am_path)
    if not os.path.exists(os.path.join(cfg.am_path, cfg.filename)):
        os.mkdir(os.path.join(cfg.am_path, cfg.filename))
    if not os.path.exists(os.path.join(cfg.am_path, cfg.filename, dataset.split)):
        os.mkdir(os.path.join(cfg.am_path, cfg.filename, dataset.split))
    if on_keyframes:
        with open(os.path.join(cfg.am_path, cfg.filename, dataset.split, 'am_' + model_instance.rsplit('_', 1)[0] + '_keyframes' + '.pkl'), 'wb') as f:
            pickle.dump(am_dict, f)    
    else:
        with open(os.path.join(cfg.am_path, cfg.filename, dataset.split, 'am_' + model_instance.rsplit('_', 1)[0] + '.pkl'), 'wb') as f:
            pickle.dump(am_dict, f)    
    
    end_time = time.time() - start_time
    print('Completed in {:.0f}m {:.0f}s'.format(end_time // 60, end_time % 60))
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', required=True, help='model\'s filename under results/, e.g. 2020-03-15_20-12-17')
    parser.add_argument('-mi', '--model_instance', required=True, help='model\'s instance under results/filename/, e.g. epoch_300_1.158')
    parser.add_argument('-s', '--split', required=True, help='dataset split; possible values are \'training\', \'validation\', \'test\'')
    parser.add_argument('-kf', '--on_keyframes', action='store_true', default=False, help='extract attention maps only on keyframes')
    
    args = parser.parse_args()

    model_name = 'gcn'
    filename = args.filename
    model_instance = args.model_instance
    split = args.split
    on_keyframes = args.on_keyframes

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
    
    test_frames = daly.get_frames(annot_data, cfg, split=split, on_keyframes=on_keyframes)
    test_set = daly.DALYDataset(annot_data, 
                                test_frames,
                                cfg,
                                split=split)

    print()
    if cfg.zero_shot:
        print('Extracting zero-shot attention maps using model', os.path.join(cfg.model_name, filename, model_instance))
    else:
        print('Extracting attention maps using model', os.path.join(cfg.model_name, filename, model_instance))

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
    extract_attention_maps(test_loader, test_set, model, device, model_instance, cfg)




