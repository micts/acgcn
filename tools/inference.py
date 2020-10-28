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

def prepare_inference(cfg, args):
   
    if args.cpu == False:
        use_gpu = True
    else:
        use_gpu = False

    if use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_device

    checkpoint = torch.load(args.model_path)
    model_state_dict = checkpoint['model_state_dict']
    
    annot_data = daly.load_tracks(load_path=args.annot_path)

    frames = daly.get_frames(annot_data, cfg, split=args.split, on_keyframes=False)
    dataset = daly.DALYDataset(annot_data,
                               frames,
                               cfg,
                               split=args.split)

    print("\nPerforming inference using model {}...".format(args.model_path))

    dataloader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if cfg.model_name == 'baseline':
        model = baseline_model.BaseNet(cfg)
        model.load_state_dict(model_state_dict)
    elif cfg.model_name == 'gcn':
        model = gcn_model.GCNNet(cfg)
        model.load_state_dict(model_state_dict)

    model = model.to(device=device)
    
    return dataloader, dataset, model, device

def inference(dataloader, dataset, model, device, cfg):

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
    print('Loss: {:.4f} | Accuracy: {:.4f} | Video mAP@ 0.5: {:.4f} | Video mAP@ 0.3: {:.4f}\n'.format(loss, acc, mAP_05, mAP_03))
    #print()
    #print('Average Precision @0.5')
    #for label in AP_05:
    #    print(label + ': ', AP_05[label])

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--annot_path', default='../data/DALY/annotations/', help='Path to annotations folder')
    parser.add_argument('--config_path', required=True, help='Path to model\'s config file')
    parser.add_argument('--model_path', required=True, help='Path to model\'s saved weights (model checkpoint)')
    parser.add_argument('--batch_size', type=int, default=3, help='Batch size')
    parser.add_argument('--gpu_device', type=str, default=0, help='GPU device (number) to use; defaults to 0')
    parser.add_argument('--cpu', action='store_true', help='Whether to use CPU instead of GPU; this option overwrites the --gpu_device argument')
    parser.add_argument('--save_scores', action='store_true', help='Whether to save model scores during inference')
    parser.add_argument('--split', default='test', help='Dataset split; possible values are \'training\', \'validation\', \'test\'')

    args = parser.parse_args()

    with open(os.path.join(args.config_path, 'config.pkl'), 'rb') as f:
        cfg_dict = pickle.load(f)
    if 'merge_function' not in cfg_dict:
        cfg_dict['merge_function'] = 'concat'
        
    cfg = config.GetConfig(**cfg_dict)
    utils.print_config(cfg)

    dataloader, dataset, model, device = prepare_inference(cfg, args)
    inference(dataloader, dataset, model, device, cfg)

