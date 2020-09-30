import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from models import pytorch_i3d
import torchvision
from torchvision.ops import RoIPool

class GCNLayer(nn.Module):
    def __init__(self, cfg, input_dim, func):
        super(GCNLayer, self).__init__()

        self.cfg = cfg
        self.num_features_gcn = cfg.num_features_gcn
        D = cfg.num_features_mixed4f
        self.input_dim = input_dim
        self.func = func

        self.softmax = nn.Softmax(dim=2)
        self.actor_emb = nn.ModuleList([nn.Linear(self.input_dim, cfg.num_features_gcn) for i in range(cfg.num_graphs)])
        self.context_emb = nn.ModuleList([nn.Conv3d(D + 2, cfg.num_features_gcn, (1, 1, 1)) for i in range(cfg.num_graphs)])
        self.dropout = nn.ModuleList([nn.Dropout(p=cfg.dropout_prob) for i in range(cfg.num_graphs)])
        self.head_linear = nn.ModuleList([nn.Linear(cfg.num_features_gcn, cfg.num_features_gcn, bias=False) for i in range(cfg.num_graphs)])

    def pad_features(self, features, num_actors_list):
        features = torch.split(features, num_actors_list)
        features = pad_sequence(features, batch_first=True)
        return features

    def unpad_features(self, features, num_actors_list, B):
        features_list = [features[b][:num_actors_list[b], :] for b in range(B)]
        features = torch.cat(features_list)
        return features

    def forward(self, actor_features, feature_map, num_actors_list, am_list, return_features):

        B = len(num_actors_list)
        
        if return_features:
            context_features_to_return = []
        
        am_list.append([])
        features_list = []
        for i in range(self.cfg.num_graphs):

            # Actor transformation
            actor_features_emb = self.actor_emb[i](actor_features)
            actor_features_emb = self.pad_features(actor_features_emb, num_actors_list)

            # Context transformation
            context_features = self.context_emb[i](feature_map)
            if return_features:
                context_features_to_return.append(context_features)

            # Dot-product attention
            context_features = context_features.view(B, context_features.shape[1], -1)
            adjacency_matrix = torch.bmm(actor_features_emb, context_features)
            if self.cfg.save_am:
                am_list[-1].append(adjacency_matrix)
            adjacency_matrix = self.softmax(adjacency_matrix)

            mask = torch.zeros(adjacency_matrix.shape).bool().to(adjacency_matrix.device)
            for b in range(B):
                mask[b][num_actors_list[b]:] = True
            adjacency_matrix = adjacency_matrix.masked_fill(mask, 0.)

            # Graph convolutions
            updated_context_features = torch.bmm(adjacency_matrix, context_features.transpose(2, 1)) # message passing
            updated_actor_features = updated_context_features + actor_features_emb
            updated_actor_features = self.unpad_features(updated_actor_features, num_actors_list, B)
            updated_actor_features = self.dropout[i](updated_actor_features)
            updated_actor_features = self.head_linear[i](updated_actor_features) # graph/head-specific linear transformation
            updated_actor_features = F.relu(updated_actor_features)
            features_list.append(updated_actor_features)
        
        if return_features:
            return features_list, context_features_to_return
        
        # Merge output of multiple graphs
        if self.cfg.num_graphs > 1:
            if self.func == 'concat':
                multihead_features = torch.cat(features_list, dim=1)
            elif self.func == 'sum':
                multihead_features = torch.stack(features_list)
                multihead_features = torch.sum(multihead_features, dim=0)
        else:
            multihead_features = features_list[0]
        return multihead_features, am_list

class GCNNet(nn.Module):

    def __init__(self, cfg):
        super(GCNNet, self).__init__()

        self.cfg = cfg
        D = self.cfg.num_features_mixed4f
        crop_size = self.cfg.crop_size

        backbone = pytorch_i3d.InceptionI3d(final_endpoint='Logits') # Set to 'Logits' to build mixed_5b and mixed_5c layers
        backbone.build()
        pretrained_dict = torch.load(os.path.join(cfg.i3d_weights_path, 'rgb_imagenet.pt')) # Load weights pre-trained on Imagenet + Kinetics

        # 1. filter out unnecessary keys
        backbone_weights = {k: v for k, v in pretrained_dict.items() if k in backbone.state_dict()}
        # 2. overwrite entries in the existing state dict
        backbone.state_dict().update(backbone_weights)
        # 3. load the new state dict
        backbone.load_state_dict(backbone_weights)

        mixed_5b = backbone.end_points['Mixed_5b']
        mixed_5c = backbone.end_points['Mixed_5c']

        backbone = pytorch_i3d.InceptionI3d(final_endpoint='Mixed_4f')
        backbone.build()
        backbone_weights = {k: v for k, v in pretrained_dict.items() if k in backbone.state_dict()}
        backbone.state_dict().update(backbone_weights)
        backbone.load_state_dict(backbone_weights)

        self.backbone = backbone
        
        self.roi_pool_op = RoIPool((crop_size[0], crop_size[1]), spatial_scale=1)
        self.mixed_5b = mixed_5b
        self.mixed_5c = mixed_5c
        self.avg_pool_3d = nn.AvgPool3d((cfg.out_feature_temp_size, crop_size[0], crop_size[1]))
        
        self.actor_dropout = nn.Dropout(p=cfg.dropout_prob)
        self.context_dropout = nn.Dropout3d(p=cfg.dropout_prob)
        
        self.gcn_layers = nn.ModuleList(self.construct_layers(cfg))
        
        self.dropout = nn.Dropout(cfg.dropout_prob)
        if cfg.merge_function == 'sum':
            self.linear = nn.Linear(cfg.num_features_gcn, cfg.num_actions)
        elif cfg.merge_function == 'concat':
            self.linear = nn.Linear(cfg.num_graphs * cfg.num_features_gcn, cfg.num_actions)
    
        self.initialize_weights() # initialize gcn layers
        
    def construct_layers(self, cfg):
        layers_list = []
        for layer_num in range(cfg.num_layers):
            if layer_num == 0:
                input_dim = cfg.num_features_mixed5c + 4
            else:
                input_dim = cfg.num_graphs * cfg.num_features_gcn + 4
            if layer_num != (cfg.num_layers - 1): # if not last layer
                func = 'concat'
            else:
                func = cfg.merge_function # if last layer
            layers_list.append(GCNLayer(cfg, input_dim=input_dim, func=func))
        return layers_list
                               
    def initialize_weights(self):
        for layer_num in range(self.cfg.num_layers):
            if layer_num == 0: # first layer
                bound = 0.024 # approximately equal to np.sqrt(1 / self.gcn_layers[0].context_emb[0].weight.shape[1]) - 0.01
            else: 
                bound = 0.034 # approximately equal to np.sqrt(1 / self.gcn_layers[0].context_emb[0].weight.shape[1])
            for graph_num in range(self.cfg.num_graphs):
                nn.init.kaiming_normal_(self.gcn_layers[layer_num].actor_emb[graph_num].weight, nonlinearity='linear')
                nn.init.zeros_(self.gcn_layers[layer_num].actor_emb[graph_num].bias)
                nn.init.kaiming_normal_(self.gcn_layers[layer_num].head_linear[graph_num].weight, nonlinearity='relu')
                nn.init.uniform_(self.gcn_layers[layer_num].context_emb[graph_num].weight, -bound, bound)
                nn.init.zeros_(self.gcn_layers[layer_num].context_emb[graph_num].bias)
        
        nn.init.kaiming_normal_(self.linear.weight, nonlinearity='relu')
        nn.init.zeros_(self.linear.bias)
    
    def get_fm_coords(self, b, t, d):
        coords = torch.linspace(-1, 1, d) # range in [-1, 1]
        x = coords.unsqueeze(0).repeat(d, 1)
        y = coords.unsqueeze(1).repeat(1, d)
        coords = torch.stack((x, y))
        coords = coords.unsqueeze(1).repeat(1, t, 1, 1)
        coords = coords.unsqueeze(0).repeat(b, 1, 1, 1, 1)
        if self.cfg.use_gpu:
            coords = coords.cuda()
        return coords

    def get_actor_coords(self, boxes_input):
        coords = torch.split(boxes_input, 8)
        coords = torch.stack(coords)
        coords = torch.mean(coords, dim=1)
        cw = coords[:, 3] - coords[:, 1]
        ch = coords[:, 4] - coords[:, 2]
        cx = (coords[:, 1] + coords[:, 3]) / 2
        cy = (coords[:, 2] + coords[:, 4]) / 2
        coords = torch.stack((cw, ch, cx, cy), dim=1)
        coords = (coords / 14) * 2 - 1 # normalize in [-1, 1]
        if self.cfg.use_gpu:
            coords = coords.cuda()
        return coords

    def forward(self, batch, *argv):
        imgs, person_boxes, num_boxes_per_frame, action_labels = batch
        
        return_features = False
        if argv:
            if argv[0] == 'return_features':
                return_features = True
        
        B = imgs.shape[0]
        T = imgs.shape[2]
        N = self.cfg.num_person_boxes
        T_fm = self.cfg.out_feature_temp_size
        OH, OW = self.cfg.out_feature_size
        am_list = []

        labels = [action_labels[b][15][:num_boxes_per_frame[b][15]] for b in range(B)]
        labels = torch.cat(labels)

        num_actors_list = []
        for b in range(B):
            num_actors_list.append(num_boxes_per_frame[b][15].item())
        num_actors = sum(num_actors_list)

        feature_map = self.backbone(imgs)

        boxes_idx = []
        boxes_to_map = []
        boxes = torch.reshape(person_boxes, (B * T, N, 4))
        for b in range(B):
            for i in range(20):
                if i < num_boxes_per_frame[b][15]:
                    for frame_idx, frame in enumerate(range(b * T, (b + 1) * T)):
                        if (frame + 1) % 4 == 0:
                            boxes_idx.append((frame + 1) / 4)
                            boxes_to_map.append(boxes[frame][i])
        boxes_idx = torch.FloatTensor(boxes_idx) - 1
        boxes_idx = boxes_idx.to(person_boxes.device)
        boxes_to_map = torch.stack(boxes_to_map)
        boxes_idx = torch.reshape(boxes_idx, (len(boxes_idx), 1))
        boxes_input = torch.cat((boxes_idx, boxes_to_map), dim=1)

        feature_map = feature_map.permute(0, 2, 1, 3, 4)
        feature_map = feature_map.contiguous().view(B * T_fm, self.cfg.num_features_mixed4f, OH, OW)

        pooled_features = self.roi_pool_op(feature_map, boxes_input) # RoI Pooling
        pooled_features = torch.split(pooled_features, T_fm)
        pooled_features = torch.stack(pooled_features)
        pooled_features = pooled_features.permute(0, 2, 1, 3, 4)

        actor_features = self.mixed_5b(pooled_features)
        actor_features = self.mixed_5c(actor_features)
        actor_features = self.avg_pool_3d(actor_features)
        actor_features = actor_features.view(num_actors, self.cfg.num_features_mixed5c,)
        actor_features = self.actor_dropout(actor_features)
        actor_coords = self.get_actor_coords(boxes_input)

        feature_map = feature_map.view(B, T_fm, self.cfg.num_features_mixed4f, OH, OW)
        feature_map = feature_map.permute(0, 2, 1, 3, 4)
        feature_map = self.context_dropout(feature_map)
        fm_coords = self.get_fm_coords(feature_map.shape[0], feature_map.shape[2], feature_map.shape[3])
        feature_map = torch.cat((feature_map, fm_coords), 1) # Attach feature map coordinates (x, y)
        
        if not return_features:
            for j in range(len(self.gcn_layers)):
                actor_features = torch.cat((actor_features, actor_coords), dim=1) # Attach actor coordinates (w, h, x, y)
                actor_features, am_list = self.gcn_layers[j](actor_features, feature_map, num_actors_list, am_list, return_features) # GCN layer
        else: # only to extract actor and object features
            for j in range(len(self.gcn_layers)):
                if j != (len(self.gcn_layers) - 1):
                    actor_features = torch.cat((actor_features, actor_coords), dim=1) # Attach actor coordinates (w, h, x, y)
                    actor_features, _ = self.gcn_layers[j](actor_features, feature_map, num_actors_list, am_list, False) # GCN layer
                else: # only for last layer
                    actor_features = torch.cat((actor_features, actor_coords), dim=1) # Attach actor coordinates (w, h, x, y)
                    actor_features_emb_list, context_features_list = self.gcn_layers[j](actor_features, feature_map, num_actors_list, am_list, return_features) # GCN layer
                    
                    return actor_features_emb_list, context_features_list
                    
        actor_features = self.dropout(actor_features)
        scores = self.linear(actor_features)
        return scores, labels, am_list
