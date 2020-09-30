from __future__ import print_function, division
import os
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

import dataset
from models import gcn_model
from models import baseline_model
import utils
import eval_utils
import vis_utils

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

def train(cfg):

    if cfg.save_log:
        if not os.path.exists(cfg.results_path):
            os.mkdir(cfg.results_path)
        if not os.path.exists(os.path.join(cfg.results_path, cfg.model_name)):
            os.mkdir(os.path.join(cfg.results_path, cfg.model_name))
        if cfg.load_checkpoint:
            cfg.filename = 'checkpoint_' + cfg.checkpoint_path.split('/')[-2]
        else:
            cfg.filename = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        if not os.path.exists(os.path.join(cfg.results_path, cfg.model_name, cfg.filename)):
            os.mkdir(os.path.join(cfg.results_path, cfg.model_name, cfg.filename))
            utils.save_config(cfg)
    
    if cfg.save_scores:
        if not os.path.exists(cfg.scores_path):
            os.mkdir(cfg.scores_path)
        if not os.path.exists(os.path.join(cfg.scores_path, cfg.model_name)):
            os.mkdir(os.path.join(cfg.scores_path, cfg.model_name))
        if not os.path.exists(os.path.join(cfg.scores_path, cfg.model_name, cfg.filename)):
            os.mkdir(os.path.join(cfg.scores_path, cfg.model_name, cfg.filename))    
    
    if cfg.save_am:
        if not os.path.exists(cfg.am_path):
            os.mkdir(cfg.am_path)
        if not os.path.exists(os.path.join(cfg.am_path, cfg.filename)):
            os.mkdir(os.path.join(cfg.am_path, cfg.filename))
    
    if not os.path.exists(os.path.join(cfg.results_path, cfg.model_name, cfg.filename, 'grad_flow')):
        os.mkdir(os.path.join(cfg.results_path, cfg.model_name, cfg.filename, 'grad_flow'))
    
    if cfg.use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.device_list
    
    utils.print_config(cfg)
    
    training_set, validation_set = dataset.return_dataset(cfg)
    datasets = {'train': training_set, 'val': validation_set}     
        
    training_loader = data.DataLoader(training_set, batch_size=cfg.training_batch_size, shuffle=True, num_workers=4, worker_init_fn=worker_init_fn)
    validation_loader = data.DataLoader(validation_set, batch_size=cfg.test_batch_size, shuffle=False, num_workers=4)
    
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
    
    if cfg.load_checkpoint:
        model, optimizer, start_epoch = load_checkpoint(model, optimizer, cfg.checkpoint_path)
        cfg.start_epoch = start_epoch
        
    train_model(dataloaders, datasets, model, device, optimizer, cfg)

def train_model(dataloaders, datasets, model, device, optimizer, cfg): 
            
    train_start_time = time.time()
    
    if cfg.save_am:
        am_dict = {}
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
        
        for phase in ['train', 'val']:
            
            if (phase == 'val') and (epoch % cfg.num_epochs_to_val != 0):
                continue
            
            if phase == 'train':
                model.train()
                epoch_start_time = time.time()
            else:
                model.eval()  
                epoch_start_time = time.time()
            
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
                
                num_actors_list = []
                for b in range(imgs.shape[0]):
                    num_actors_list.append(num_boxes_per_frame[b][15].item())
                                    
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
                        if cfg.plot_grad_flow:
                            if idx % cfg.num_epochs_to_val == 0:
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
                if cfg.save_am:
                    if 'MLKCbW7c9Wg.mp4' in video_names:
                        if torch.tensor([2]) in instances:
                            if torch.tensor([2161]) in center_frames:
                                video_info = (video_names, instances.tolist(), center_frames.tolist())
                                am_dict = utils.save_am(am_list, am_dict, video_info)
                
            if phase == 'train':
                if cfg.plot_grad_flow:
                    plt.savefig(os.path.join(cfg.results_path, cfg.model_name, cfg.filename, 'grad_flow', '_epoch_' + str(epoch).zfill(3) + '.png'), bbox_inches = "tight")
                    plt.close()
                
            epoch_loss = running_loss / split_size
            epoch_acc = running_corrects.double() / split_size
            
            mAP_05, AP_05 = eval_utils.videomAP(scores_dict, datasets[phase].annot_data, datasets[phase].split, cfg, iou_threshold=0.5)
            mAP_03, _ = eval_utils.videomAP(scores_dict, datasets[phase].annot_data, datasets[phase].split, cfg, iou_threshold=0.3)
            
            if phase == 'train':
                train_loss = epoch_loss
                train_acc = epoch_acc
                train_mAP_05 = mAP_05
                train_mAP_03 = mAP_03
            elif phase == 'val':
                val_loss = epoch_loss
                val_acc = epoch_acc
                val_mAP_05 = mAP_05
                val_mAP_03 = mAP_03
            
            epoch_time = time.time() - epoch_start_time
            print('{} epoch complete in {:.0f}m {:.0f}s'.format(phase, epoch_time // 60, epoch_time % 60))
            print('{} Loss: {:.4f} | Accuracy: {:.4f} | Video mAP @0.5: {:.4f} | Video mAP @0.3: {:.4f}'.format(phase, epoch_loss, epoch_acc, mAP_05, mAP_03))
            if epoch % cfg.num_epochs_to_val == 0:
                print('AP @0.5')
                for label in AP_05:
                    print('{}: {:.4f}'.format(label, AP_05[label]))
            
            if cfg.save_scores:
                if phase == 'val':
                    if epoch % cfg.num_epochs_to_val == 0:
                        if mAP_05 > 45:
                            if not os.path.exists(os.path.join(cfg.scores_path, cfg.model_name, cfg.filename, datasets[phase].split)):
                                os.mkdir(os.path.join(cfg.scores_path, cfg.model_name, cfg.filename, datasets[phase].split))    
                            with open(os.path.join(cfg.scores_path, cfg.model_name, cfg.filename, datasets[phase].split, 'scores_epoch_' + str(epoch).zfill(3)) + '.pkl', 'wb') as f:
                                pickle.dump(scores_dict, f)
            
            if cfg.save_am:
                if epoch % cfg.num_epochs_to_val == 0:
                    if not os.path.exists(os.path.join(cfg.am_path, cfg.filename, datasets[phase].split)):
                        os.mkdir(os.path.join(cfg.am_path, cfg.filename, datasets[phase].split))
                    with open(os.path.join(cfg.am_path, cfg.filename, datasets[phase].split, 'am_epoch_' + str(epoch).zfill(3)) + '.pkl', 'wb') as f:
                        pickle.dump(am_dict, f)
                    
        #save intermediate model and results
        if cfg.save_log:
            if epoch % cfg.num_epochs_to_val == 0:
                if val_mAP_05 > 45:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': train_loss,
                        'train_acc': train_acc,
                        'train_mAP_05': train_mAP_05,
                        'train_mAP_03': train_mAP_03,
                        'val_loss': val_loss,
                        'val_acc': val_acc,
                        'val_mAP_05': val_mAP_05,
                        'val_mAP_03': val_mAP_03,
                        }, os.path.join(cfg.results_path, cfg.model_name, cfg.filename, 'epoch_' + str(epoch).zfill(3) + '_' + str(round(val_loss, 3)) + '.pth'))
                else:
                    torch.save({
                        'epoch': epoch,
                        'train_loss': train_loss,
                        'train_acc': train_acc,
                        'train_mAP_05': train_mAP_05,
                        'train_mAP_03': train_mAP_03,
                        'val_loss': val_loss,
                        'val_acc': val_acc,
                        'val_mAP_05': val_mAP_05,
                        'val_mAP_03': val_mAP_03,
                        }, os.path.join(cfg.results_path, cfg.model_name, cfg.filename, 'epoch_' + str(epoch).zfill(3) + '_' + str(round(val_loss, 3)) + '.pth'))
        
        print()
    
    train_time = time.time() - train_start_time
    print('Training complete in {:.0f}h {:.0f}m'.format(
        (train_time // 60) // 60, (train_time // 60) % 60))
