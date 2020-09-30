import os
import pickle
import torch
import numpy as np
import scipy
import scipy.special
import utils
import config

def score_tubes(scores, num_actions):
    scored_tubes = {}
    videos = list(scores.keys())
    for video in videos:
        assert video not in scored_tubes
        scored_tubes[video] = {}
        instances = list(scores[video].keys())

        for instance in instances:
            assert instance not in scored_tubes[video]
            center_frames = list(scores[video][instance].keys())
            num_actors = len(scores[video][instance][center_frames[0]]['scores'])
            scored_tubes[video][instance] = np.copy(np.zeros([num_actors, 3], dtype=np.float64))
            all_scores = np.copy(np.zeros([len(center_frames), num_actors, num_actions], dtype=np.float64))
            for frame_idx, center_frame in enumerate(center_frames):
                all_scores[frame_idx, :, :] = scores[video][instance][center_frame]['scores']

            actor_scores = np.mean(all_scores, axis=0)
            tube_predictions = np.argmax(actor_scores, axis=1)
            tube_scores = np.max(actor_scores, axis=1) 

            scored_tubes[video][instance][:, 0] = tube_predictions
            scored_tubes[video][instance][:, 1] = tube_scores
            scored_tubes[video][instance][:, 2] = np.arange(num_actors)
    return scored_tubes

def nms(tubes, scores, class_label, nms_threshold):
    
    if class_label in scores[:, 0]:
        tube_ids = [] # IDs of tubes to keep after nms
        tube_scores = []
        cl_scores = scores[np.where(scores[:, 0] == class_label)[0], :]
        ranked_scores = cl_scores[cl_scores[:, 1].argsort()[::-1]]   
        while len(ranked_scores) > 0:
            hs_tube_id = int(ranked_scores[0, 2]) # id of highest scoring tube
            hs_tube_score = ranked_scores[0, 1]
            hs_tube = tubes[hs_tube_id] # boxes of highest scoring tube
            tube_ids.append(hs_tube_id) # keep id of highest scoring tube
            tube_scores.append(hs_tube_score) # keep score of highest scoring tube
            # Calculate spatio-temporal IoU between highest scoring tube
            # and every other tube of the same class.
            spt_ious = []
            for idx in range(len(ranked_scores)):
                tube_id = int(ranked_scores[idx, 2])
                tube = tubes[tube_id]
                # spatio-temporal IoU
                spt_ious.append(np.mean(utils.get_tube_iou(hs_tube[:, 1:5], tube[:, 1:5])))
            inds = np.where(np.array(spt_ious) < nms_threshold)[0]
            ranked_scores = ranked_scores[inds, :]
        return tube_ids, tube_scores
    else:
        return [], []

def get_tube_predictions(scored_tubes, annot_data, split, num_actions, nms_threshold=0.2):
    tube_preds = {class_label:[] for class_label in range(1, num_actions)}
    videos = list(scored_tubes.keys())
    for video in videos:
        instances = list(scored_tubes[video].keys())
        for instance in instances:
            tubes = annot_data[split][video]['action_instances'][instance]['tubes']
            scores = scored_tubes[video][instance]
            for class_label in range(1, num_actions): # exclude background
                tube_ids, tube_scores = nms(tubes, scores, class_label, nms_threshold)
                if len(tube_ids) > 0:
                    for score_idx, tube_id in enumerate(tube_ids):
                        tube_preds[class_label].append((video, instance, tube_scores[score_idx], tube_id, tubes[tube_id]))
    return tube_preds

def get_gt_tubes(annot_data, split, videos, cfg):
    gt_tubes = {class_label:{} for class_label in range(1, cfg.num_actions)}
    for class_label in range(1, cfg.num_actions):
        for video in videos:
            w, h = annot_data[split][video]['(width, height)']
            instances = list(annot_data[split][video]['action_instances'].keys())
            for instance in instances:
                if cfg.zero_shot:
                    instance_label = cfg.class_map[annot_data[split][video]['action_instances'][instance]['label_name']]
                else:
                    instance_label = annot_data[split][video]['action_instances'][instance]['label']
                tbound = annot_data[split][video]['action_instances'][instance]['tbound']
                if instance_label == class_label:
                    if video not in gt_tubes[class_label]:
                        gt_tubes[class_label][video] = {}
                    #if instance not in gt_tubes[class_label][video]:
                    assert instance not in gt_tubes[class_label][video], (video, instance, class_label, gt_tubes[class_label][video])
                    gt_tubes[class_label][video][instance] = []
                    
                    keyframes_dict = annot_data[split][video]['action_instances'][instance]['keyframes']
                    keyframe_ids = list(keyframes_dict.keys())
                    keyframe_boxes = np.copy(np.stack(list(keyframes_dict.values())))
                    keyframe_boxes[:, [0, 2]] = np.copy(keyframe_boxes[:, [0, 2]]) * w
                    keyframe_boxes[:, [1, 3]] = np.copy(keyframe_boxes[:, [1, 3]]) * h
                    gt_tubes[class_label][video][instance].append(np.hstack((np.array(keyframe_ids).reshape(-1, 1), keyframe_boxes)))
                    for instance2 in instances:
                        if instance != instance2:
                            if cfg.zero_shot:
                                instance_label2 = cfg.class_map[annot_data[split][video]['action_instances'][instance2]['label_name']]
                            else:
                                instance_label2 = annot_data[split][video]['action_instances'][instance2]['label']
                            keyframe_boxes_to_add = []
                            keyframes_to_add = []
                            keyframes_dict2 = annot_data[split][video]['action_instances'][instance2]['keyframes']
                            keyframe_ids2 = list(keyframes_dict2.keys())
                            keyframe_boxes2 = np.copy(np.stack(list(keyframes_dict2.values())))
                            keyframe_boxes2[:, [0, 2]] = np.copy(keyframe_boxes2[:, [0, 2]]) * w
                            keyframe_boxes2[:, [1, 3]] = np.copy(keyframe_boxes2[:, [1, 3]]) * h
                            for idx, keyframe_id2 in enumerate(keyframe_ids2):
                                if (keyframe_id2 >= tbound[0]) and (keyframe_id2 <= tbound[1]):
                                    keyframe_boxes_to_add.append(keyframe_boxes2[idx, :])
                                    keyframes_to_add.append(keyframe_id2)
                            if len(keyframes_to_add) > 0:
                                keyframe_boxes_to_add = np.stack(keyframe_boxes_to_add)
                                if class_label != instance_label2:
                                    if video not in gt_tubes[instance_label2]:
                                        gt_tubes[instance_label2][video] = {}
                                    if instance not in gt_tubes[instance_label2][video]:
                                        gt_tubes[instance_label2][video][instance] = []
                                    gt_tubes[instance_label2][video][instance].append(np.hstack((np.array(keyframes_to_add).reshape(-1, 1), keyframe_boxes_to_add)))
                                else:
                                    gt_tubes[class_label][video][instance].append(np.hstack((np.array(keyframes_to_add).reshape(-1, 1), keyframe_boxes_to_add)))
    return gt_tubes

def average_precision(pr):

    prdif = pr[1:, 1] - pr[:-1, 1]
    prsum = pr[1:, 0] + pr[:-1, 0]

    return np.sum(prdif * prsum * 0.5)

def videomAP(scores, annot_data, split, cfg, iou_threshold=0.5):    
    
    PR = {}

    scored_tubes = score_tubes(scores, cfg.num_actions)
    pred_tubes = get_tube_predictions(scored_tubes, annot_data, split, cfg.num_actions, nms_threshold=0.2)
    gt_tubes = get_gt_tubes(annot_data, split, list(scored_tubes.keys()), cfg)
    
    for class_label in range(1, cfg.num_actions):
        class_pred_tubes = pred_tubes[class_label]
        class_gt_tubes = gt_tubes[class_label]

        pr = np.empty((len(class_pred_tubes) + 1, 2), dtype=np.float32)
        pr[0,0] = 1.0
        pr[0,1] = 0.0
        
        fp = 0
        tp = 0
        fn = 0
        covered_gt_tubes = {}
        for video in class_gt_tubes:
            covered_gt_tubes[video] = {}
            instances = class_gt_tubes[video]
            for instance in instances:
                num_gt_tubes = len(class_gt_tubes[video][instance])
                covered_gt_tubes[video][instance] = num_gt_tubes * [0]
                fn += num_gt_tubes

        for i, j in enumerate(np.argsort(-np.array([pred_tube[2] for pred_tube in class_pred_tubes]))):
            video, instance, score, tube_id, tube = class_pred_tubes[j]

            is_positive = False
            if video in class_gt_tubes:
                if instance in class_gt_tubes[video]:
                    gt_kf_tubes = class_gt_tubes[video][instance]
                    ious = []
                    for gt_tube in gt_kf_tubes:
                        keyframes = gt_tube[:, 0]
                        ious.append(np.mean(utils.get_tube_iou(tube[np.in1d(tube[:, 0], keyframes), 1:5], gt_tube[:, 1:5])))
                    amax = np.argmax(ious)

                    if ious[amax] >= iou_threshold:
                        if covered_gt_tubes[video][instance][amax] == 0:
                            is_positive = True
                            covered_gt_tubes[video][instance][amax] = 1

            if is_positive:
                tp += 1
                fn -= 1
            else:
                fp += 1

            pr[i+1,0] = tp / (tp + fp)
            pr[i+1,1] = tp / (tp + fn)

        PR[utils.idx2class(cfg.class_map, class_label)] = pr

    AP = {class_name:100 * average_precision(PR[class_name]) for class_name in PR}
    mAP = sum(list(AP.values())) / len(AP)
    return mAP, AP

def get_objects_recall(config_path, epoch, split, num_threshold_points=100):
    
    with open(os.path.join(config_path, 'config.pkl'), 'rb') as f:
        cfg_dict = pickle.load(f)    
    
    with open(os.path.join(cfg_dict['am_path'], cfg_dict['filename'], split, 'am_epoch_' + str(epoch) + '_keyframes' + '.pkl'), 'rb') as f:
        am = pickle.load(f)
        
    with open(os.path.join(cfg_dict['annot_path'], 'annotated_data.pkl'), 'rb') as f:
        annotated_data = pickle.load(f)
    
    with open(os.path.join(cfg_dict['annot_path'], 'daly1.1.0.pkl'), 'rb') as f:
        annot = pickle.load(f, encoding='latin1')
    
    obj_annotations = utils.get_obj_annotations(annotated_data, annot)
    
    classes_to_exclude = cfg_dict['classes_to_exclude']
    OH = cfg_dict['out_feature_size'][0]
    OW = cfg_dict['out_feature_size'][1]
    T_fm = cfg_dict['out_feature_temp_size']
    class_map = utils.class2idx_map()
    num_layers = cfg_dict['num_layers']
    num_graphs = cfg_dict['num_graphs']
    
    # collect tubes with IoU > 0.5
    tubes_dict = {}
    for video in annotated_data[split]:
        vid_annot = annotated_data[split][video]
        w, h = vid_annot['(width, height)']
        for instance in vid_annot['action_instances']:
            instance_annot = annotated_data[split][video]['action_instances'][instance]
            keyframes_dict = instance_annot['keyframes']
            keyframe_ids = np.array(list(keyframes_dict.keys()))
            keyframe_boxes = np.copy(np.stack(list(keyframes_dict.values())))
            keyframe_boxes[:, [0, 2]] = np.copy(keyframe_boxes[:, [0, 2]]) * w
            keyframe_boxes[:, [1, 3]] = np.copy(keyframe_boxes[:, [1, 3]]) * h
            for tube_id in instance_annot['tubes']:
                tube = instance_annot['tubes'][tube_id]
                spt_iou = np.mean(utils.get_tube_iou(tube[np.in1d(tube[:, 0], keyframe_ids), 1:5], keyframe_boxes))
                if spt_iou > 0.5:
                    if video not in tubes_dict:
                        tubes_dict[video] = {}
                    if instance not in tubes_dict[video]:
                        tubes_dict[video][instance] = {}
                        tubes_dict[video][instance]['tubes'] = {}
                        tubes_dict[video][instance]['tube_labels'] = []
                    tubes_dict[video][instance]['tubes'][tube_id] = tube
                    tubes_dict[video][instance]['tube_labels'].append(instance_annot['tube_labels'][tube_id])
    
    objects_recall = {}
    thresholds = np.linspace(0, 1, num_threshold_points)
    for class_label in range(1, len(class_map)): # recall curve for each class (exclude background)
        if classes_to_exclude is not None:
            class_name = utils.idx2class(class_map, class_label) 
            if class_name not in classes_to_exclude:
                continue
        # calculate total number of false negatives
        fn = 0
        for video in obj_annotations[split]:
            if (video not in am.keys()) or (video not in tubes_dict):
                continue # no object annotations or no positive tubes
            vid_annot = obj_annotations[split][video]
            for instance in vid_annot['action_instances']:
                if instance not in tubes_dict[video]:
                    continue
                tubes_instance = tubes_dict[video][instance]
                assert len(set(tubes_instance['tube_labels'])) == 1
                instance_label = tubes_instance['tube_labels'][0]
                if class_label != instance_label:
                    continue # skip instances of different class
                keyframes = list(vid_annot['action_instances'][instance].keys())
                for keyframe in keyframes:
                    fn_keyframe = 0
                    for box_idx in range(len(vid_annot['action_instances'][instance][keyframe])):
                        if (vid_annot['action_instances'][instance][keyframe][box_idx][5] == 1) or (vid_annot['action_instances'][instance][keyframe][box_idx][6] == 1):
                            continue
                        fn_keyframe += 1
                    fn += fn_keyframe * len(tubes_instance['tube_labels']) # total number of false negatives
        
        recall_values = np.zeros([len(thresholds), 2])
        for idx, threshold in enumerate(thresholds): # for each threshold
            tp = 0
            fn_ = fn
            for video in obj_annotations[split]:
                if (video not in am.keys()) or (video not in tubes_dict): 
                    continue # no object annotations or no positive tubes
                vid_annot = obj_annotations[split][video]
                W, H = obj_annotations[split][video]['(width, height)']
                for instance in vid_annot['action_instances']:
                    if instance not in tubes_dict[video]:
                        continue # no positive tubes
                    assert len(set(tubes_dict[video][instance]['tube_labels'])) == 1
                    instance_label = tubes_dict[video][instance]['tube_labels'][0]
                    if class_label != instance_label:
                        continue # skip instances of different class
                    keyframes = list(vid_annot['action_instances'][instance].keys())
                            
                    for tube_id in tubes_dict[video][instance]['tubes']: # for each (positive) tube                   
                        for keyframe in keyframes: 
                            att_map_list = []
                            for layer_num in range(num_layers):
                                for graph_num in range(num_graphs):
                                    lngn = str(layer_num) + str(graph_num) # layer number and graph number
                                    att_map = am[video][instance][keyframe][lngn][tube_id]
                                    att_map = att_map.reshape(T_fm, OH, OW)[3]
                                    att_map = att_map.reshape(-1)
                                    att_map = scipy.special.softmax(att_map)
                                    att_map = att_map.reshape(OH, OW)
                                    att_map_list.append(att_map)

                            # get obj annotation for keyframe
                            for box_idx in range(len(vid_annot['action_instances'][instance][keyframe])): # for each object annotation in keyframe
                                obj_box = vid_annot['action_instances'][instance][keyframe][box_idx][0:4]
                                obj_box = obj_box * OH
                                x1 = int(round(obj_box[0]))
                                y1 = int(round(obj_box[1]))
                                x2 = int(round(obj_box[2]))
                                y2 = int(round(obj_box[3]))
                                sum_list = []
                                att_map_idx = 0
                                for layer_num in range(num_layers):
                                    for graph_num in range(num_graphs):
                                        patch = att_map_list[att_map_idx][y1:y2 + 1, x1:x2 + 1]
                                        att_sum = np.sum(patch) # add attention values inside the object bounding box
                                        sum_list.append(att_sum)
                                        att_map_idx += 1
                                is_positive = any(np.array(sum_list) > threshold) # if any of the graphs satisfies condition
                                if is_positive:
                                    tp += 1
                                    fn_ -= 1

            recall_values[idx, 0] = tp / (tp + fn_)
            recall_values[idx, 1] = threshold
        objects_recall[class_label] = recall_values
    return objects_recall

def get_tubes_recall(config_path, epoch, split, num_threshold_points=100):
    
    with open(os.path.join(config_path, 'config.pkl'), 'rb') as f:
        cfg_dict = pickle.load(f)    
    
    scores_path = cfg_dict['scores_path']
    model_name = cfg_dict['model_name']
    filename = cfg_dict['filename']
    
    with open(os.path.join(scores_path, model_name, filename, split, 'scores_epoch_' + str(epoch) + '.pkl'), 'rb') as f:
        scores = pickle.load(f)
    scored_tubes = score_tubes(scores, cfg_dict['num_actions'])
    
    with open(os.path.join(cfg_dict['annot_path'], 'annotated_data.pkl'), 'rb') as f:
        annotated_data = pickle.load(f)
    
    classes_to_exclude = cfg_dict['classes_to_exclude']
    class_map = utils.class2idx_map()
    
    # collect tubes with IoU > 0.5
    tubes_dict = {}
    for video in annotated_data[split]:
        vid_annot = annotated_data[split][video]
        w, h = vid_annot['(width, height)']
        for instance in vid_annot['action_instances']:
            instance_annot = annotated_data[split][video]['action_instances'][instance]
            keyframes_dict = instance_annot['keyframes']
            keyframe_ids = np.array(list(keyframes_dict.keys()))
            keyframe_boxes = np.copy(np.stack(list(keyframes_dict.values())))
            keyframe_boxes[:, [0, 2]] = np.copy(keyframe_boxes[:, [0, 2]]) * w
            keyframe_boxes[:, [1, 3]] = np.copy(keyframe_boxes[:, [1, 3]]) * h
            for tube_id in instance_annot['tubes']:
                tube = instance_annot['tubes'][tube_id]
                spt_iou = np.mean(utils.get_tube_iou(tube[np.in1d(tube[:, 0], keyframe_ids), 1:5], keyframe_boxes))
                if spt_iou > 0.5:
                    if video not in tubes_dict:
                        tubes_dict[video] = {}
                    if instance not in tubes_dict[video]:
                        tubes_dict[video][instance] = {}
                        tubes_dict[video][instance]['tubes'] = {}
                        tubes_dict[video][instance]['tube_labels'] = []
                    tubes_dict[video][instance]['tubes'][tube_id] = tube
                    tubes_dict[video][instance]['tube_labels'].append(instance_annot['tube_labels'][tube_id])
    
    tubes_recall = []
    for class_label in range(1, len(class_map)): # for each class
        running_corrects = 0
        running_total = 0
        for video in scored_tubes:
            if video not in tubes_dict:
                continue
            for instance in scored_tubes[video]:
                if instance not in tubes_dict[video]:
                    continue
                tubes_instance = tubes_dict[video][instance]
                assert len(set(tubes_instance['tube_labels'])) == 1
                instance_label = tubes_instance['tube_labels'][0]
                if class_label != instance_label:
                    continue
                tube_ids = np.array(list(tubes_dict[video][instance]['tubes'].keys()))
                predicted_labels = scored_tubes[video][instance][tube_ids, 0]
                gt_labels = np.array(tubes_dict[video][instance]['tube_labels'], dtype=predicted_labels.dtype)
                running_corrects += np.sum(predicted_labels == gt_labels)
                running_total += len(gt_labels)
        tubes_recall.append(running_corrects / running_total)
        
    return tubes_recall

def return_tp_fp(config_path, epoch, split, iou_threshold=0.5):    
    
    cfg = config.Config('gcn', 0, 0, None, False)
    with open(os.path.join(config_path, 'config.pkl'), 'rb') as f:
        cfg_dict = pickle.load(f)
    cfg = utils.overwrite_config(cfg, cfg_dict)

    scores_path = cfg.scores_path
    model_name = cfg.model_name
    filename = cfg.filename
    
    with open(os.path.join(cfg.annot_path, 'annotated_data.pkl'), 'rb') as f:
        annot_data = pickle.load(f)
    
    with open(os.path.join(scores_path, model_name, filename, split, 'scores_epoch_' + str(epoch) + '.pkl'), 'rb') as f:
        scores = pickle.load(f)
        
    scored_tubes = score_tubes(scores, cfg.num_actions)
    pred_tubes = get_tube_predictions(scored_tubes, annot_data, split, cfg.num_actions, nms_threshold=0.2)
    gt_tubes = get_gt_tubes(annot_data, split, list(scored_tubes.keys()), cfg)
    
    tp = {}
    fp = {}
    for class_label in range(1, cfg.num_actions):
        tp[class_label] = []
        fp[class_label] = []
        class_pred_tubes = pred_tubes[class_label]
        class_gt_tubes = gt_tubes[class_label]

        covered_gt_tubes = {}
        for video in class_gt_tubes:
            covered_gt_tubes[video] = {}
            instances = class_gt_tubes[video]
            for instance in instances:
                num_gt_tubes = len(class_gt_tubes[video][instance])
                covered_gt_tubes[video][instance] = num_gt_tubes * [0]

        for i, j in enumerate(np.argsort(-np.array([pred_tube[2] for pred_tube in class_pred_tubes]))):
            video, instance, score, tube_id, tube = class_pred_tubes[j]

            is_positive = False
            if video in class_gt_tubes:
                if instance in class_gt_tubes[video]:
                    gt_kf_tubes = class_gt_tubes[video][instance]
                    ious = []
                    for gt_tube in gt_kf_tubes:
                        keyframes = gt_tube[:, 0]
                        ious.append(np.mean(utils.get_tube_iou(tube[np.in1d(tube[:, 0], keyframes), 1:5], gt_tube[:, 1:5])))
                    amax = np.argmax(ious)

                    if ious[amax] >= iou_threshold:
                        if covered_gt_tubes[video][instance][amax] == 0:
                            is_positive = True
                            covered_gt_tubes[video][instance][amax] = 1

            if is_positive:
                tp[class_label].append((video, instance, score, tube_id, tube))
            else:
                fp[class_label].append((video, instance, score, tube_id, tube))

    return tp, fp