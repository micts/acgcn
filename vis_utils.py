import os
import pickle
import numpy as np
import scipy
import torch
import cv2
import matplotlib
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE

import utils
import eval_utils

def plot_grad_flow(named_parameters, epoch, idx):
    """
    Adapted from https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/7.
    """
    
    if idx == 0:
        plt.figure(figsize=(18, 6))
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            if p.grad is None:
                continue
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    
def plot_objects_recall(objects_recall, fig_width=13, fig_height=9):
    class_map = utils.class2idx_map()
    name = "tab10"
    cmap = get_cmap(name)
    colors = cmap.colors
    fig = plt.figure(figsize=(fig_width, fig_height))
    plt.grid(alpha=0.4)
    plt.xlabel('Attention Threshold', fontsize=fig_width)
    plt.ylabel('Recall', fontsize=fig_width)
    plt.xticks(np.linspace(0, 1, len(class_map)))
    plt.yticks(np.linspace(0, 1, len(class_map)))
    fig.axes[0].tick_params(labelsize=fig_width - 2)
    hex_colors = [matplotlib.colors.to_hex(color) for color in colors]
    for idx, class_label in enumerate(objects_recall):
        plt.plot(objects_recall[class_label][:, 1], objects_recall[class_label][:, 0], c=hex_colors[idx], label=utils.idx2class(class_map, class_label), linewidth=2)
    fig.axes[0].legend(fontsize=fig_width - 4)    
    
def scatterplot_recall(tubes_recall, objects_recall, fig_width=13, fig_height=9, exclude_classes=None):
    class_map = utils.class2idx_map()
    AUC = []
    for class_label in range(1, len(class_map)): # for each class
        # area under the curve
        AUC.append(eval_utils.average_precision(objects_recall[class_label]))
    name = "tab10"
    cmap = get_cmap(name)
    colors = cmap.colors
    fig = plt.figure(figsize=(fig_width, fig_height))
    plt.grid(alpha=0.4)
    plt.xlabel('AUC (Objects Recall)', fontsize=fig_width)
    plt.ylabel('Tubes Recall', fontsize=fig_width)
    fig.axes[0].tick_params(labelsize=fig_width - 2)
    hex_colors = [matplotlib.colors.to_hex(color) for color in colors]
    for idx, class_label in enumerate(range(1, len(class_map))):
        if exclude_classes is not None:
            if utils.idx2class(class_map, class_label) in exclude_classes:
                continue
        plt.scatter(AUC[idx], tubes_recall[idx], c=hex_colors[idx], label=utils.idx2class(class_map, class_label), s=50)
    fig.axes[0].legend(fontsize=fig_width - 3)    
    
def plot_tSNE(config_path, epoch, split, plot_type='actions', fig_width=14, fig_height=9):
    
    with open(os.path.join(config_path, 'config.pkl'), 'rb') as f:
        cfg_dict = pickle.load(f)
    with open(os.path.join(cfg_dict['features_path'], cfg_dict['filename'], split, 'features_epoch_' + str(epoch) + '.pkl'), 'rb') as f:
        actor_and_obj_features = pickle.load(f)
    with open(os.path.join(cfg_dict['annot_path'], 'daly1.1.0.pkl'), 'rb') as f:
        annot = pickle.load(f, encoding='latin1')
    
    filename = cfg_dict['filename']
    if plot_type == 'actions':
        class_map = utils.class2idx_map(cfg_dict['classes_to_exclude'])
        action_list = list(class_map.keys())
        action_list.remove('Background')
        palette = "tab10"
        cmap = get_cmap(palette)
        colors = cmap.colors
        # total of 10 colors
        hex_colors = [matplotlib.colors.to_hex(color) for color in colors]
    elif plot_type == 'objects':
        obj_list = annot['objectList']
        palette = "Paired"
        cmap = get_cmap(palette)
        colors = cmap.colors
        # total of 14 colors
        hex_colors = [matplotlib.colors.to_hex(color) for color in colors]
        hex_colors.append(matplotlib.colors.to_hex('black'))
        hex_colors.append(matplotlib.colors.to_hex('grey'))

    features_dict = {}
    names_dict = {}
    for graph_num in range(len(actor_and_obj_features)):
        features_dict[graph_num] = []
        names_dict[graph_num] = []
    for graph_num in range(len(actor_and_obj_features)):
        for features in actor_and_obj_features[graph_num]:
            name = features[1]
            if (plot_type == 'actions') and (name not in action_list):
                continue
            if (plot_type == 'objects') and (name not in obj_list):
                continue
            features_dict[graph_num].append(features[0])
            names_dict[graph_num].append(features[1])

    for graph_num in features_dict:
        fig = plt.figure(figsize=(fig_width, fig_height))
        features = np.vstack(features_dict[graph_num])
        names = np.array(names_dict[graph_num])
        if plot_type == 'objects':
            # collect names of 14 most frequent objects
            obj_freqs = []
            for obj in obj_list:
                idxs = np.where(names == obj)[0]
                obj_freqs.append((obj, len(idxs)))
            most_freq_obj = sorted(obj_freqs, key=lambda tup: tup[1], reverse=True)
            most_freq_obj = most_freq_obj[0:14]
            most_freq_obj = [obj for obj, _ in most_freq_obj]
            # collect features of 14 most frequent objects
            idxs = np.in1d(names, most_freq_obj)
            features = features[idxs, :]
            names = names[idxs]
        np.random.seed(5186)
        print('Fitting t-SNE for graph', str(graph_num) + '...')
        features_emb = TSNE(n_components=2, perplexity=30.0, n_iter=1000, learning_rate=200).fit_transform(features)
        if plot_type == 'actions':
            for idx, action in enumerate(action_list):
                idxs = np.where(names == action)[0]
                plt.scatter(features_emb[idxs, 0], features_emb[idxs, 1], s=13, c=hex_colors[idx], label=action)
        elif plot_type == 'objects':
            for idx, obj in enumerate(most_freq_obj):
                idxs = np.where(names == obj)[0]
                plt.scatter(features_emb[idxs, 0], features_emb[idxs, 1], s=13, c=hex_colors[idx], label=obj)
        if graph_num == 0:
            plt.legend()
        plt.show()
        
def plot_attention_maps(config_path, epoch, pred_type, size=5, split='test', fig_width=15, fig_height=25):
    
    with open(os.path.join(config_path, 'config.pkl'), 'rb') as f:
        cfg_dict = pickle.load(f)
    
    am_path = cfg_dict['am_path']
    filename = cfg_dict['filename']
    annot_path = cfg_dict['annot_path']
    class_map = cfg_dict['class_map']
    T_fm = cfg_dict['out_feature_temp_size']
    OH, OW = cfg_dict['out_feature_size']
    num_layers = cfg_dict['num_layers']
    num_graphs = cfg_dict['num_graphs']
    h, w = cfg_dict['img_size']
    data_path = cfg_dict['data_path']
    
    with open(os.path.join(annot_path, 'annotated_data.pkl'), 'rb') as f:
        annotated_data = pickle.load(f)
    
    with open(os.path.join(am_path, filename, split, 'am_epoch_' + str(epoch) + '.pkl'), 'rb') as f:
        am = pickle.load(f)
        
    for class_label in range(1, cfg_dict['num_actions']):
        
        print(utils.idx2class(class_map, class_label))
        print(15 * '-')
        class_preds = pred_type[class_label]
        num_preds = len(class_preds)
        if num_preds == 0:
            print()
            continue
        if num_preds > size:
            sample = np.random.choice(num_preds, size=size, replace=False)
        else:
            sample = np.random.choice(num_preds, size=num_preds, replace=False)
        preds = [class_preds[s] for s in sample]    
        
        for idx, pred in enumerate(preds):         
            video, instance, score, tube_id, tube = pred
            tbound = annotated_data[split][video]['action_instances'][instance]['tbound']
            W, H = annotated_data[split][video]['(width, height)']
            tube_label = annotated_data[split][video]['action_instances'][instance]['tube_labels'][tube_id]
            center_frame = int(np.random.choice(list(am[video][instance].keys()), 1)) # sample center frame
            frame_seq = np.array([frame_num for frame_num in range(center_frame - 15, center_frame + 16 + 1)])
            frames_in_tbound = []
            for frame_num in frame_seq:
                if (frame_num > tbound[0]) and (frame_num < tbound[-1]):
                    frames_in_tbound.append(frame_num)    
            clip = []
            for frame_num in frame_seq:
                if frame_num in frames_in_tbound:
                    clip.append(frame_num)
                else:
                    if frame_num < center_frame:
                        clip.append(frames_in_tbound[0])
                    else:
                        clip.append(frames_in_tbound[-1])    
            clip = clip[3::4] # get every 4th frame of clip
            
            print('Prediction:', utils.idx2class(class_map, class_label), '|', 
                  'Ground Truth:', utils.idx2class(class_map, tube_label))
            
            fig, ax = plt.subplots(T_fm, num_layers * num_graphs + 1, figsize=(fig_width, fig_height))
            for t in range(T_fm):
                frame_num = clip[t]
                #frame_num = center_frame
                boxes = np.copy(annotated_data[split][video]['action_instances'][instance]['tubes'][tube_id])
                box = boxes[np.where(boxes[:, 0] == frame_num)[0], 1:5][0]
                box[[0, 2]] = (box[[0, 2]] / W) * w
                box[[1, 3]] = (box[[1, 3]] / H) * h
                img = plt.imread(os.path.join(data_path, video, 'frame' + str(frame_num).zfill(6) + '.jpg'))
                rect = plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, edgecolor='r', linewidth=1)
                ax[t][0].add_patch(rect)
                ax[t][0].imshow(img)
                ax_idx = 0
                for j in range(num_layers):
                    for i in range(num_graphs):
                        ax_idx += 1
                        img2 = plt.imread(os.path.join(data_path, video, 'frame' + str(frame_num).zfill(6) + '.jpg'))
                        lngn = str(j) + str(i)
                        att_map = scipy.special.softmax(np.copy(am[video][instance][center_frame][lngn][tube_id]))
                        att_map = att_map.reshape(T_fm, OH, OW)[t] # temporal attention maps
                        #att_map = np.mean(att_map.reshape(T_fm, OH, OW), axis=0) # temporal average attention map
                        res = cv2.resize(att_map, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
                        extent = 0, w, 0, h
                        ax[t][ax_idx].imshow(img2, extent=extent)
                        ax[t][ax_idx].imshow(res, alpha=0.5, cmap='Reds', extent=extent)

            plt.pause(.5)
            print('===' * 20)
        print()
        print()