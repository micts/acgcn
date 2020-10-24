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

def inference(dataloader, dataset, model, device, model_instance, cfg, save_scores):
    
    inference_start_time = time.time()
    cross_entropy_loss = torch.nn.CrossEntropyLoss()
    cross_entropy_loss = cross_entropy_loss.to(device)
    softmax = torch.nn.Softmax(dim=1)

    model.eval()
    epoch_start_time = time.time()

    scores_dict = {}
    running_loss = 0.0
    running_corrects = 0
    split_size = 0
    
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

            scores, labels, am_list = model(batch)
            labels = labels.to(device)
            loss = cross_entropy_loss(scores, labels)

        running_loss += loss.item() * labels.shape[0]
        predictions = torch.max(scores, dim=1)[1]
        running_corrects += torch.sum(predictions == labels)
        split_size += labels.shape[0]
        
        scores_numpy = softmax(scores.detach().cpu()).numpy()
        labels_numpy = labels.detach().cpu().numpy()
        scores_dict = utils.save_scores(scores_numpy,
                                        labels_numpy,
                                        num_actors_list,
                                        scores_dict,
                                        video_names,
                                        instances.tolist(),
                                        center_frames.tolist())
        
    loss = running_loss / split_size
    acc = running_corrects.double() / split_size
    mAP_05, AP_05 = eval_utils.videomAP(scores_dict, dataset.annot_data, dataset.split, cfg, iou_threshold=0.5)
    mAP_03, AP_03 = eval_utils.videomAP(scores_dict, dataset.annot_data, dataset.split, cfg, iou_threshold=0.3)
    
    inference_time = time.time() - inference_start_time
    print('Inference complete in {:.0f}m {:.0f}s'.format(inference_time // 60, inference_time % 60))
    print('Loss: {:.4f} | Accuracy: {:.4f} | Video mAP@ 0.5: {:.4f} | Video mAP@ 0.3: {:.4f}'.format(loss, acc, mAP_05, mAP_03))
    print()
    print('Average Precision @0.5')
    for label in AP_05:
        print(label + ': ', AP_05[label])
    
    if save_scores:
        if not os.path.exists(cfg.scores_path):
            os.mkdir(cfg.scores_path)
        if not os.path.exists(os.path.join(cfg.scores_path, cfg.model_name)):
            os.mkdir(os.path.join(cfg.scores_path, cfg.model_name))
        if not os.path.exists(os.path.join(cfg.scores_path, cfg.model_name, cfg.filename)):
            os.mkdir(os.path.join(cfg.scores_path, cfg.model_name, cfg.filename))
        if not os.path.exists(os.path.join(cfg.scores_path, cfg.model_name, cfg.filename, dataset.split)):
            os.mkdir(os.path.join(cfg.scores_path, cfg.model_name, cfg.filename, dataset.split))
        with open(os.path.join(cfg.scores_path, cfg.model_name, cfg.filename, dataset.split, 'scores_' + model_instance.rsplit('_', 1)[0] + '.pkl'), 'wb') as f:
            pickle.dump(scores_dict, f)    
        
    return(loss, acc, mAP_05, mAP_03)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-mn', '--model_name', required=True, help='model name: \'baseline\' or \'gcn\'')
    parser.add_argument('-f', '--filename', required=True, help='model\'s filename under results/, e.g. 2020-03-15_20-12-17')
    parser.add_argument('-mi', '--model_instance', required=True, help='model\'s instance under results/filename/, e.g. epoch_300_1.158')
    parser.add_argument('-s', '--split', required=True, help='dataset split; possible values are \'training\', \'validation\', \'test\'')

    args = parser.parse_args()

    model_name = args.model_name
    filename = args.filename
    model_instance = args.model_instance
    split = args.split
    
    cfg = config.Config(model_name, 0, 0, 'sum', False) # default config values
    
    with open(os.path.join(cfg.results_path, model_name, filename, 'config.pkl'), 'rb') as f:
        cfg_dict = pickle.load(f)
    
    results_path = cfg.results_path
    scores_path = cfg.scores_path
    cfg = utils.overwrite_config(cfg, cfg_dict) # overwrite default config file
    if model_name == 'baseline':
        cfg.results_path = results_path
        cfg.scores_path = scores_path
    utils.print_config(cfg)

    if cfg.use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    model_path = os.path.join(cfg.results_path, model_name, filename, model_instance)
    checkpoint = torch.load(model_path)
    model_state_dict = checkpoint['model_state_dict']
    
    annot_data = daly.gen_labels_keyframes(label_tracks=False, load_path=cfg.annot_path)

    test_frames = daly.get_frames(annot_data, cfg, split=split, on_keyframes=False)
    test_set = daly.DALYDataset(annot_data, 
                                test_frames,
                                cfg,
                                split=split)
    
    print()
    print('Performing inference using model', os.path.join(cfg.model_name, filename, model_instance))

    test_loader = data.DataLoader(test_set, batch_size=cfg.test_batch_size, shuffle=False, num_workers=4)

    if cfg.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if cfg.model_name == 'baseline':
        model = baseline_model.BaseNet(cfg)
        model.load_state_dict(model_state_dict)
    elif cfg.model_name == 'gcn':
        model = gcn_model.GCNNet(cfg)
        model.load_state_dict(model_state_dict)
    else:
        assert(False), 'Variable model_name should be either \'baseline\' or 2 \'gcn\'.'

    model = model.to(device=device)
    loss, acc, mAP_05, mAP_03 = inference(test_loader, test_set, model, device, model_instance, cfg, save_scores=True)




