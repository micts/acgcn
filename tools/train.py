from __future__ import print_function, division
import os
import sys
import argparse
import warnings
import pickle
import numpy as np
import math
import time
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils import data
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

sys.path.append('.')

from config import config
from dataset import dataset
from models import gcn_model, baseline_model
from utils import utils, eval_utils, vis_utils

class WarmupCosineSchedule(LambdaLR):
    """
    Adapted from https://huggingface.co/transformers/_modules/transformers/optimization.html#get_cosine_schedule_with_warmup.

    Linear warmup and then cosine decay.
        Linearly increases learning rate from `init_lr` to `max_lr` over `warmup_steps` training steps.
        Decreases learning rate from `max_lr` to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, init_lr, max_lr, min_lr, warmup_steps, total_steps, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.cycles = cycles
        self.init_lr = init_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return self.init_lr + (float(step) * (self.max_lr - self.init_lr)) / float(max(1.0, self.warmup_steps))
        progress = float(step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
        return max(0.0, 0.5 * self.max_lr * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))

def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def load_checkpoint(model, optimizer, checkpoint_path):
    start_epoch = 0

    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("Checkpoint '{}' loaded | Continuing from epoch {}".format(checkpoint_path.split('/')[-1], checkpoint['epoch']))
    print()
    return model, optimizer, start_epoch

def prepare_training(cfg):

    if cfg.use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.device_list

    training_set, validation_set = dataset.return_dataset(cfg)
    datasets = {'train': training_set, 'val': validation_set}

    training_loader = data.DataLoader(training_set, batch_size=cfg.training_batch_size, shuffle=True, num_workers=4, worker_init_fn=worker_init_fn)
    validation_loader = data.DataLoader(validation_set, batch_size=cfg.validation_batch_size, shuffle=False, num_workers=4)

    dataloaders = {'train': training_loader, 'val': validation_loader}

    if cfg.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if cfg.model_name == 'baseline': # initalize baseline model
        model = baseline_model.BaseNet(cfg)
    elif cfg.model_name == 'gcn': # initalize gcn model
        model = gcn_model.GCNNet(cfg)
    else:
        assert(False), 'Variable model_name should be either \'baseline\' or 2 \'gcn\'.'

    model = model.to(device=device)

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=1, momentum=cfg.momentum, weight_decay=cfg.weight_decay)

    if cfg.resume_training:
        cfg.filename = 'checkpoint_' + cfg.checkpoint_path.split('/')[-2]
        model, optimizer, start_epoch = load_checkpoint(model, optimizer, cfg.checkpoint_path)
        cfg.start_epoch = start_epoch
    else:
        cfg.filename = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

    utils.make_dirs(os.path.join(cfg.results_path, cfg.model_name, cfg.filename))
    if cfg.save_scores:
        utils.make_dirs(os.path.join(cfg.scores_path, cfg.model_name, cfg.filename, datasets['train'].split))
        utils.make_dirs(os.path.join(cfg.scores_path, cfg.model_name, cfg.filename, datasets['val'].split))
    if cfg.plot_grad_flow:
        utils.make_dirs(os.path.join(cfg.results_path, cfg.model_name, cfg.filename, 'grad_flow'))

    utils.save_config(cfg)
    utils.print_config(cfg)

    return dataloaders, datasets, model, device, optimizer

def train(dataloaders, datasets, model, device, optimizer, cfg):

    train_start_time = time.time()

    cross_entropy_loss = nn.CrossEntropyLoss()
    cross_entropy_loss = cross_entropy_loss.to(device)
    softmax = nn.Softmax(dim=1)
    scheduler = WarmupCosineSchedule(optimizer,
                                     cfg.init_lr,
                                     cfg.max_lr,
                                     cfg.min_lr,
                                     len(dataloaders['train']) * cfg.warmup_epochs,
                                     len(dataloaders['train']) * cfg.total_epochs)

    for epoch in range(cfg.start_epoch, cfg.total_epochs):

        np.random.seed()

        print('Epoch {}/{}'.format(epoch, cfg.total_epochs - 1))
        print('Learning Rate {}'.format([group['lr'] for group in optimizer.param_groups]))
        print('-' * 10)

        results = {}
        results['epoch'] = epoch
        for phase in ['train', 'val']:

            if (phase == 'val') and (epoch % cfg.num_epochs_to_val != 0):
                continue

            epoch_start_time = time.time()

            if phase == 'train':
                model.train()
            else:
                model.eval()

            scores_dict = {}
            running_loss = 0.0
            running_corrects = 0
            split_size = 0
            for idx, batch_data in enumerate(dataloaders[phase]):

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

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):

                    if phase == 'train' and cfg.set_bn_eval:
                        model.apply(set_bn_eval)

                    scores, labels, am_list = model(batch) # forward pass
                    labels = labels.to(device)
                    loss = cross_entropy_loss(scores, labels)

                    if phase == 'train':
                        loss.backward()
                        if cfg.plot_grad_flow and (idx % cfg.num_epochs_to_val == 0):
                            vis_utils.plot_grad_flow(model.named_parameters(), epoch, idx)
                        optimizer.step()

                if phase == 'train':
                    scheduler.step()

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

            if phase == 'train' and cfg.plot_grad_flow:
                plt.savefig(os.path.join(cfg.results_path, cfg.model_name, cfg.filename, 'grad_flow', '_epoch_' + str(epoch).zfill(3) + '.png'), bbox_inches="tight")
                plt.close()

            epoch_loss = running_loss / split_size
            epoch_acc = running_corrects.double() / split_size

            mAP_05, AP_05 = eval_utils.videomAP(scores_dict,
                                                datasets[phase].annot_data,
                                                datasets[phase].split,
                                                cfg,
                                                iou_threshold=0.5)
            mAP_03, _ = eval_utils.videomAP(scores_dict,
                                            datasets[phase].annot_data,
                                            datasets[phase].split,
                                            cfg,
                                            iou_threshold=0.3)

            results[phase + '_loss'] = epoch_loss
            results[phase + '_acc'] = epoch_acc
            results[phase + '_mAP_05'] = mAP_05
            results[phase + '_mAP_03'] = mAP_03

            epoch_time = time.time() - epoch_start_time
            print('{} epoch complete in {:.0f}m {:.0f}s'.format(phase, epoch_time // 60, epoch_time % 60))
            print('{} Loss: {:.4f} | Accuracy: {:.4f} | Video mAP @0.5: {:.4f} | Video mAP @0.3: {:.4f}'.format(phase, epoch_loss, epoch_acc, mAP_05, mAP_03))
            # if epoch % cfg.num_epochs_to_val == 0:
            #     print('AP @0.5')
            #     for label in AP_05:
            #         print('{}: {:.4f}'.format(label, AP_05[label]))

            if cfg.save_scores and (phase == 'val') and (epoch % cfg.num_epochs_to_val == 0):
                with open(os.path.join(cfg.scores_path, cfg.model_name, cfg.filename, datasets[phase].split, 'scores_epoch_' + str(epoch).zfill(3)) + '.pkl', 'wb') as f:
                    pickle.dump(scores_dict, f)

        #save intermediate model and results
        if epoch % cfg.num_epochs_to_val == 0:
            results['model_state_dict'] = model.state_dict()
            results['optimizer_state_dict'] = optimizer.state_dict()
            torch.save(results, os.path.join(cfg.results_path, cfg.model_name, cfg.filename, 'epoch_{}_{:.3f}.pth'.format(str(epoch).zfill(3), epoch_loss)))
        # always save?
        #torch.save(results, os.path.join(cfg.results_path, cfg.model_name, cfg.filename, 'epoch_{}_{:.3f}.pth'.format(str(epoch).zfill(3), epoch_loss)))

        print()

    train_time = time.time() - train_start_time
    print('Training complete in {:.0f}h {:.0f}m'.format(
        (train_time // 60) // 60, (train_time // 60) % 60))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', default='../data/DALY/frames', help='Path to dataset folder')
    parser.add_argument('--annot_path', default='../data/DALY/annotations/', help='Path to annotations folder')
    parser.add_argument('--model_name', required=True, help='Model name: \'baseline\' or \'gcn\'')
    parser.add_argument('--num_layers', type=int, help='Number of gcn layers')
    parser.add_argument('--num_graphs', type=int, help='Number of graphs per layer')
    parser.add_argument('--merge_function', help='Function to merge output of multiple graphs in final layer: \'sum\' or \'concat\'')
    parser.add_argument('--zero_shot', action='store_true', default=False, help='Exclude certain classes during training; refer to config.py to modify the classes to be exluded')
    parser.add_argument('--total_epochs', type=int, required=True, help='Total number of epochs')
    parser.add_argument('--warmup_epochs', type=int, default=0, help='Number of epochs to apply linear learning rate warm-up; valid only when --init_lr is provided')
    parser.add_argument('--init_lr', type=float, help='Initial learning rate; valid only when --warmup_epochs > 0')
    parser.add_argument('--max_lr', type=float, required=True, help='Maximum learning rate')
    parser.add_argument('--min_lr', type=float, default=0, help='Minimum learning rate')
    parser.add_argument('--batch_size', type=int, default=3, help='Batch size')
    parser.add_argument('--gpu_device', default='0', type=str, help='GPU device (number) to use; defaults to 0')
    parser.add_argument('--cpu', action='store_true', help='Whether to use CPU instead of GPU; this option overwrites the --gpu_device argument')
    parser.add_argument('--results_path', default='../results/', help='Path to save training and validation results (e.g. metrics, model weights)')
    parser.add_argument('--save_scores', action='store_true', help='Whether to save model scores during training and validation')
    parser.add_argument('--scores_path', default='../scores/', help='Path to save model scores during training and validation')
    parser.add_argument('--resume_training', action='store_true', help='Whether to resume training from a checkpoint')
    parser.add_argument('--checkpoint_path', help='Path to model\'s saved weights (model\'s checkpoint); used to resume training from a checkpoint')
    parser.add_argument('--num_epochs_to_val', default=10, help='Perform validation every --num_epochs_to_val epochs')

    args = parser.parse_args()
    cfg = config.Config(args)

    dataloaders, datasets, model, device, optimizer = prepare_training(cfg)
    train(dataloaders, datasets, model, device, optimizer, cfg)
