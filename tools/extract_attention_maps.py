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

from config import config
from dataset import dataset, daly
from models import gcn_model, baseline_model
from utils import utils, eval_utils

def extract_attention_maps(dataloader, model, device):
    
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
        
        num_actors_list = [num_boxes_per_frame[b][15].item() for b in range(imgs.shape[0])]        
        
        batch = [data.to(device=device) for data in [imgs, person_boxes]]
        batch.append(num_boxes_per_frame)
        batch.append(action_labels)

        with torch.set_grad_enabled(False):

            _, _, am_list = model(batch)
            
        video_info = (video_names, instances.tolist(), center_frames.tolist())
        am_dict = utils.save_am(am_list, am_dict, video_info)
    
    return am_dict            
    
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--annot_path', default='../data/DALY/annotations/', help='Path to annotations folder')
    parser.add_argument('--config_path', required=True, help='Path to model\'s config file')
    parser.add_argument('--model_path', required=True, help='Path to model\'s saved weights (model checkpoint)')
    parser.add_argument('--am_path', default='../am/', help='Path to save attention maps (adjacency matrices)')
    parser.add_argument('--gpu_device', type=str, default=0, help='GPU device (number) to use; defaults to 0')
    parser.add_argument('--cpu', action='store_true', help='Whether to use CPU instead of GPU; this option overwrites the --gpu_device argument')
    parser.add_argument('--on_keyframes', action='store_true', default=False, help='Extract attention maps only on keyframes; used for attention map evaluation')
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
    
    frames = daly.get_frames(annot_data, cfg, split=args.split, on_keyframes=args.on_keyframes)
    dataset = daly.DALYDataset(annot_data, 
                                frames,
                                cfg,
                                split=args.split)
    
    if cfg.zero_shot:
        print("\nExtracting zero-shot attention maps using model {}...".format(args.model_path))
    else:
        print("\nExtracting attention maps using model {}...".format(args.model_path))

    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

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
    am_dict = extract_attention_maps(dataloader, model, device)

    model_am_path = os.path.join(args.am_path, cfg.filename, args.split)
    utils.make_dirs(model_am_path)
    epoch_nr = args.model_path.rsplit('/')[-1].rsplit('_', 1)[0]
    if args.on_keyframes:
        with open(os.path.join(model_am_path, 'am_{}_keyframes.pkl'.format(epoch_nr)), 'wb') as f:
            pickle.dump(am_dict, f)
    else:
        with open(os.path.join(model_am_path, 'am_{}.pkl'.format(epoch_nr)), 'wb') as f:
            pickle.dump(am_dict, f)

    end_time = time.time() - start_time
    print('Completed in {:.0f}m {:.0f}s'.format(end_time // 60, end_time % 60))


