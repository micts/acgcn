import torch
import torch.nn as nn
from models import pytorch_i3d
import torchvision
from torchvision.ops import RoIPool

class BaseNet(nn.Module):
    
    def __init__(self, cfg):
        super(BaseNet, self).__init__()
        self.cfg = cfg
        
        if cfg.use_i3d_tail:
            self.num_features = cfg.num_features_mixed5c
        else:
            self.num_features = cfg.num_features_mixed4f
        K = cfg.crop_size[0]
        T_fm = cfg.out_feature_temp_size
                
        backbone = pytorch_i3d.InceptionI3d(final_endpoint='Logits') # Set to 'Logits' to build mixed_5b and mixed_5c layers
        backbone.build()
        pretrained_dict = torch.load('/project/code/rgb_imagenet.pt') # Load weights pre-trained on Imagenet + Kinetics

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
        self.roi_pool_op = RoIPool((cfg.crop_size[0], cfg.crop_size[1]), spatial_scale=1)
        self.mixed_5b = mixed_5b
        self.mixed_5c = mixed_5c
        self.avg_pool_3d = nn.AvgPool3d(kernel_size=[T_fm, cfg.crop_size[0], cfg.crop_size[1]], stride=(1, 1, 1))
        self.dropout = nn.Dropout(cfg.dropout_prob)
        self.linear = nn.Linear(self.num_features, cfg.num_actions)
         
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, batch):
        imgs, person_boxes, num_boxes_per_frame, action_labels = batch
        
        B = imgs.shape[0]
        T = imgs.shape[2]
        N = self.cfg.num_person_boxes
        T_fm = self.cfg.out_feature_temp_size
        OH, OW = self.cfg.out_feature_size
        
        labels = []
        for b in range(B):
            labels.append(action_labels[b][15][:num_boxes_per_frame[b][15]])
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
            for i in range(self.cfg.num_person_boxes):
                if i < num_boxes_per_frame[b][15]:
                    for frame in range(b * T, (b + 1) * T):
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
        pooled_features_per_actor = torch.split(pooled_features, T_fm)
        features = torch.stack(pooled_features_per_actor)
        features = features.permute(0, 2, 1, 3, 4) 
        
        if self.cfg.use_i3d_tail:
            features = self.mixed_5b(features)
            features = self.mixed_5c(features)
        features = self.avg_pool_3d(features) # spatio-temporal (3d) average pooling
        features = features.view(features.shape[0], self.num_features,)
        
        features = self.dropout(features)
        scores = self.linear(features)
        
        return scores, labels, num_actors_list

        
