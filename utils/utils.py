import os
import pickle
import numpy as np

def get_iou(a, b, epsilon=1e-16):
    """ 
    Adapted from http://ronny.rest/tutorials/module/localization_001/iou/
    
    Given two boxes `a` and `b` defined as a list of four numbers:
            [x1,y1,x2,y2]
        where:
            x1,y1 represent the upper left corner
            x2,y2 represent the lower right corner
        It returns the Intersect of Union score for these two boxes.

    Args:
        a:          (list of 4 numbers) [x1,y1,x2,y2]
        b:          (list of 4 numbers) [x1,y1,x2,y2]
        epsilon:    (float) Small value to prevent division by zero

    Returns:
        (float) The Intersect of Union score.
    """
    # coordinates of intersection box
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    # area of overlap - area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)
    # handle case where there is NO overlap
    if (width<0) or (height <0):
        return 0.0
    area_overlap = width * height

    # combined area
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_combined = area_a + area_b - area_overlap

    # ratio of area of overlap over combined area
    iou = area_overlap / (area_combined+epsilon)
    return iou

def get_tube_iou(a, b, epsilon=1e-16):
    """ 
    Adapted from http://ronny.rest/tutorials/module/localization_001/iou/
    
    Given two arrays `a` and `b` where each row contains a bounding
        box defined as a list of four numbers:
            [x1,y1,x2,y2]
        where:
            x1,y1 represent the upper left corner
            x2,y2 represent the lower right corner
        It returns the Intersect of Union scores for each corresponding
        pair of boxes.

    Args:
        a:          (numpy array) each row containing [x1,y1,x2,y2] coordinates
        b:          (numpy array) each row containing [x1,y1,x2,y2] coordinates
        epsilon:    (float) Small value to prevent division by zero

    Returns:
        (numpy array) The Intersect of Union scores for each pair of bounding
        boxes.
    """
    # coordinates of intersection boxes
    x1 = np.array([a[:, 0], b[:, 0]]).max(axis=0)
    y1 = np.array([a[:, 1], b[:, 1]]).max(axis=0)
    x2 = np.array([a[:, 2], b[:, 2]]).min(axis=0)
    y2 = np.array([a[:, 3], b[:, 3]]).min(axis=0)

    # areas of overlap - area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)

    # handle case where there is NO overlap
    width[width < 0] = 0
    height[height < 0] = 0

    area_overlap = width * height

    # combined areas
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    area_combined = area_a + area_b - area_overlap

    # ratio of area of overalp over combined area
    iou = area_overlap / (area_combined + epsilon)
    return iou

def gen_labels_keyframes(save_path=None):
        
    print('Generating labels for tracks...')
    
    with open('../DALY/all_videos.pkl', 'rb') as f:
        all_videos = pickle.load(f)
    
    with open('../DALY/daly1.1.0.pkl', 'rb') as f:
        annot = pickle.load(f, encoding='latin1')
    
    with open('../DALY/humantubes_DALY.pkl', 'rb') as f:
        tubes = pickle.load(f, encoding='latin1')
    
    with open('../DALY/validation_videos.pkl', 'rb') as f:
        validation_videos = pickle.load(f)
    
    with open('../DALY/frames_size.pkl', 'rb') as f:
        frames_size = pickle.load(f)
    
    action_dict = {}
    for idx, video in enumerate(all_videos):
        vid_annot = annot['annot'][video]['annot']
        # correct inconsistencies between humantubes_DALY.pkl and daly1.1.0.pkl
        if video == 'TnXBUmIYmyI.mp4':
            actions = ['FoldingTextile', 'CleaningFloor']
        elif video == 'sjgkGNQyOWg.mp4':
            actions = ['TakingPhotosOrVideos', 'CleaningWindows', 'CleaningFloor']
        else:
            actions = [action for action in vid_annot]
        action_dict[video] = actions       

    iou_d = {}
    ious_ind = []
    annot_data = {}
    annot_data['training'] = {}
    annot_data['validation'] = {}
    annot_data['test'] = {}
    for idx, video in enumerate(all_videos):

        if video in annot['splits'][0]: # test videos
            split = 'test'
        elif video in validation_videos:
            split = 'validation'
        else:
            split = 'training'

        video_id = video.split('.')[0]
        vid_annot = annot['annot'][video]['annot']

        W, H = frames_size[video]

        annot_data[split][video] = {}
        annot_data[split][video]['(width, height)'] = (W, H)
        annot_data[split][video]['action_instances'] = {}

        actions = action_dict[video]
        
        instance_idx = 0
        for action_idx, action in enumerate(actions):
            annot_instance_idx = 0
            for instance in vid_annot[action]:
                if (instance['flags']['isReflection'] == True) or (instance['flags']['isAmbiguous'] == True):
                    annot_instance_idx += 1
                    continue
                    
                annot_data[split][video]['action_instances'][instance_idx] = {}
                annot_data[split][video]['action_instances'][instance_idx]['frames'] = {}
                annot_data[split][video]['action_instances'][instance_idx]['tubes'] = {}
                annot_data[split][video]['action_instances'][instance_idx]['tube_labels'] = []
                annot_data[split][video]['action_instances'][instance_idx]['tube_labelnames'] = []
                annot_data[split][video]['action_instances'][instance_idx]['tube_ious'] = []
                annot_data[split][video]['action_instances'][instance_idx]['keyframes'] = {}
                tubes_instance = tubes[0][video][instance_idx]
                
                start_frame_tubes = int(tubes_instance[0][0, 0]) # first frame of tube/instance
                end_frame_tubes = int(tubes_instance[0][-1, 0]) # last frame of tube/instance
                begin_time = vid_annot[action][annot_instance_idx]['beginTime']
                end_time = vid_annot[action][annot_instance_idx]['endTime']
                annot_data[split][video]['action_instances'][instance_idx]['time_bound'] = (begin_time, end_time)
                annot_data[split][video]['action_instances'][instance_idx]['tbound'] = (start_frame_tubes, end_frame_tubes)
                annot_data[split][video]['action_instances'][instance_idx]['label_name'] = action
                annot_data[split][video]['action_instances'][instance_idx]['label'] = utils.class2idx(action)
                
                annot_keylist = vid_annot[action][annot_instance_idx]['keyframes']
                keyframes = []
                for annot_keyframe in annot_keylist:
                    keyframes.append(annot_keyframe['frameNumber'])
                    assert (annot_keyframe['frameNumber'] >= start_frame_tubes) and (annot_keyframe['frameNumber'] <= end_frame_tubes)
                    annot_data[split][video]['action_instances'][instance_idx]['keyframes'][annot_keyframe['frameNumber']] = annot_keyframe['boundingBox'][0]
                
                for tube_idx, tube in enumerate(tubes_instance):
                    annot_data[split][video]['action_instances'][instance_idx]['tubes'][tube_idx] = tube 
                    for frame in tube:
                        frame_num = int(frame[0])
                        if frame_num not in annot_data[split][video]['action_instances'][instance_idx]['frames']:
                            annot_data[split][video]['action_instances'][instance_idx]['frames'][frame_num] = []
                        annot_data[split][video]['action_instances'][instance_idx]['frames'][frame_num].append(frame[1:5])
                
                    tube_iou_class = [0, 'Background']
                    for action_idx2, action2 in enumerate(actions):
                        for instance2_idx, instance2 in enumerate(vid_annot[action2]):
                            if (instance2['flags']['isReflection'] == True) or (instance2['flags']['isAmbiguous'] == True):
                                continue
                            #vid_annot[action][instance2]
                            annot_keylist2 = vid_annot[action2][instance2_idx]['keyframes']
                            keyframes2 = []
                            keyframe_boxes2 = []
                            for annot_keyframe2 in annot_keylist2:
                                if (annot_keyframe2['frameNumber'] >= start_frame_tubes) and (annot_keyframe2['frameNumber'] <= end_frame_tubes):
                                    keyframes2.append(annot_keyframe2['frameNumber'])
                                    keyframe_box2 = np.copy(annot_keyframe2['boundingBox'][0])
                                    keyframe_box2 = np.array([keyframe_box2[0] * W, keyframe_box2[1] * H, keyframe_box2[2] * W, keyframe_box2[3] * H])
                                    keyframe_boxes2.append(keyframe_box2)
                            if len(keyframes2) > 0:
                                keyframe_boxes2 = np.vstack(keyframe_boxes2)
                                keyframes2 = np.array(keyframes2)
                                spt_iou = np.mean(utils.get_tube_iou(tube[np.in1d(tube[:, 0], keyframes2), 1:5], keyframe_boxes2))
                                if spt_iou > 0.5:
                                    if spt_iou > tube_iou_class[0]:
                                        tube_iou_class[0] = spt_iou
                                        tube_iou_class[1] = action2
                                else:
                                    if spt_iou > tube_iou_class[0]:
                                        tube_iou_class[0] = spt_iou
                    annot_data[split][video]['action_instances'][instance_idx]['tube_labels'].append(utils.class2idx(tube_iou_class[1]))
                    annot_data[split][video]['action_instances'][instance_idx]['tube_labelnames'].append(tube_iou_class[1])
                    annot_data[split][video]['action_instances'][instance_idx]['tube_ious'].append(tube_iou_class[0])
                                                                    
                instance_idx += 1
                annot_instance_idx += 1
    
    # annot_data_ = copy.deepcopy(annot_data)
    # splits = list(annot_data.keys())
    # for split in splits:
    #     for video in annot_data[split]:
    #         instances = annot_data[split][video]['action_instances']
    #         for instance in instances:
    #             frames = list(annot_data[split][video]['action_instances'][instance]['frames'].keys())
    #             for frame in frames:
    #                 boxes = np.copy(np.vstack(annot_data[split][video]['action_instances'][instance]['frames'][frame]))
    #                 annot_data_[split][video]['action_instances'][instance]['frames'][frame] = boxes
    #             labels = np.copy(np.hstack(annot_data[split][video]['action_instances'][instance]['tube_labels']))
    #             annot_data_[split][video]['action_instances'][instance]['tube_labels'] = labels
    
    if save_path != None:
        print('Saving dictionary under', save_path)
        with open(os.path.join(save_path, 'annotated_data.pkl'), 'wb') as f:
            pickle.dump(annot_data, f)
        print('Saving finished.')
        
    return(annot_data)


def split_train_val(write_to_pkl=False):
    
    """
    Construct a new training set and a validation set from the initial training set.
    A video is assigned to either the training set or the validation set.
    """
    
    with open('/project/DALY/daly1.1.0.pkl', 'rb') as f:
        annot = pickle.load(f, encoding='latin1')
    
    all_training_videos = {}
    for idx in range(1, 11):
        all_training_videos[idx2class(idx)] = []

    for video in annot['annot']:
        if video in annot['splits'][0]:
            continue
        suggested_class = annot['annot'][video]['suggestedClass']
        all_training_videos[suggested_class].append(video)
        
    i = 0
    training_videos = []    
    validation_videos = []
    np.random.seed(86723) # 17343
    for action_class in all_training_videos:
        videos = all_training_videos[action_class]
        n_class_videos = len(videos)
        assert n_class_videos == 31
        val_videos_idxs = np.random.choice(np.arange(31), size=10, replace=False) # size=11
        for idx in range(31):        
            if idx in val_videos_idxs:
                validation_videos.append(videos[idx])
            else:
                training_videos.append(videos[idx])            
                
    if write_to_pkl:
        with open('../DALY/training_videos.pkl', 'wb') as f:
            pickle.dump(training_videos, f)
        with open('../DALY/validation_videos.pkl', 'wb') as f:
            pickle.dump(validation_videos, f)
        with open('../DALY/all_training_videos.pkl', 'wb') as f:
            pickle.dump(training_videos + validation_videos, f)
                
    return(training_videos, validation_videos)

def class2idx_map(classes_to_exclude=None):
    actions = ['Background',
               'ApplyingMakeUpOnLips', 
               'BrushingTeeth', 
               'CleaningFloor', 
               'CleaningWindows', 
               'Drinking', 
               'FoldingTextile', 
               'Ironing', 
               'Phoning', 
               'PlayingHarmonica', 
               'TakingPhotosOrVideos']

    if classes_to_exclude is not None:
        for class_name in classes_to_exclude:
            actions.remove(class_name)
    
    class_map = {action:idx for idx, action in enumerate(actions)}
    return class_map

def class2idx(class_map, class_name):    
    return class_map[class_name]

def idx2class(class_map, class_index):
    assert (class_index >=0) and (class_index <= 10), 'Class index should be in the interval [0, 10]'
    for class_name, idx in class_map.items():
        if idx == class_index:
            return(class_name)

def get_max_num_tubes(annot_data):
    """
    Calculates maximum number of tubes across all action instances.
    """
    max_num_tubes = 0
    for split in annot_data:
        for video in annot_data[split]:
            action_instances = annot_data[split][video]['action_instances']
            for instance in action_instances:
                num_tubes = len(action_instances[instance]['tubes'])
                if num_tubes > max_num_tubes:
                    max_num_tubes = num_tubes
    return(max_num_tubes)

def print_config(cfg):
    print('\n', '==========Configuration Parameters==========')
    for k, v in cfg.__dict__.items():
        if k == 'class_map':
            if k == 'class_map':
                print(k + ': {')
                for idx, (class_name, class_idx) in enumerate(v.items()):
                    if idx == (len(v.keys()) - 1):
                        print(class_name + ': ', str(class_idx) + '}', sep='')
                    else:
                        print(class_name + ': ', str(class_idx), sep='')
        else:
            print(k, ': ', v, sep='')
    print('')

def overwrite_config(cfg, cfg_dict):
    for k in cfg.__dict__:
        if k in cfg_dict:
            cfg.__dict__[k] = cfg_dict[k]
    return(cfg)

def save_config(cfg):
    with open(os.path.join(cfg.results_path, cfg.model_name, cfg.filename, 'config.pkl'), 'wb') as f:
        pickle.dump(cfg.__dict__, f)

def save_scores(scores, labels, num_actors_list, scores_dict, video_names, instances, center_frames):
    
    # Get the indices for each actor to index the arrays 'scores' and 'labels' 
    actor_idx = np.cumsum(num_actors_list)
    actor_idx = np.hstack((0, actor_idx))
    
    # We can infer the batch size based on the length of 'num_actors_list'.
    # We want to split scores per mini-batch, since each mini-batch might
    # correspond to a different video, instance, and center frame.
    batch_size = len(num_actors_list) 
    for b in range(batch_size):
        video = video_names[b]
        instance = instances[b]
        center_frame = center_frames[b]
        
        if video not in scores_dict:
            scores_dict[video] = {}
        if instance not in scores_dict[video]:
            scores_dict[video][instance] = {}
            
        # We sample center frames uniformly without replacement, 
        # so in each epoch and for each phase ('train', 'val'), 
        # a center frame should access scores_dict only once.
        # If a center frame exists already in scores_dict, raise an error.
        #assert center_frame not in scores_dict[video][instance]
        
        # Index 'scores' and 'labels' based on actor indices (actor_idx) produced by cumsum.
        scores_dict[video][instance][center_frame] = {}
        scores_dict[video][instance][center_frame]['scores'] = scores[actor_idx[b]:actor_idx[b + 1], :]
        scores_dict[video][instance][center_frame]['labels'] = labels[actor_idx[b]:actor_idx[b + 1]]
    return scores_dict

def get_obj_annotations(annotated_data, annot):
                
    with open('../DALY/all_videos.pkl', 'rb') as f:
        all_videos = pickle.load(f)
        
    with open('../DALY/daly1.1.0.pkl', 'rb') as f:
        annot = pickle.load(f, encoding='latin1')
    
    with open('../DALY/validation_videos.pkl', 'rb') as f:
        validation_videos = pickle.load(f)
        
    action_dict = {}
    for idx, video in enumerate(all_videos):
        vid_annot = annot['annot'][video]['annot']
        # correct inconsistencies between annotated_data and daly1.1.0.pkl
        if video == 'TnXBUmIYmyI.mp4':
            actions = ['FoldingTextile', 'CleaningFloor']
        elif video == 'sjgkGNQyOWg.mp4':
            actions = ['TakingPhotosOrVideos', 'CleaningWindows', 'CleaningFloor']
        else:
            actions = [action for action in vid_annot]
        action_dict[video] = actions       

    obj_annotations = {}
    obj_annotations['training'] = {}
    obj_annotations['validation'] = {}
    obj_annotations['test'] = {}
    for idx, video in enumerate(all_videos):

        if video in annot['splits'][0]: # test videos
            split = 'test'
        elif video in validation_videos:
            split = 'validation'
        else:
            split = 'training'

        video_id = video.split('.')[0]
        vid_annot = annot['annot'][video]['annot']
            
        obj_annotations[split][video] = {}
        obj_annotations[split][video]['(width, height)'] = annotated_data[split][video]['(width, height)']
        obj_annotations[split][video]['action_instances'] = {}

        actions = action_dict[video]

        instance_idx = 0
        for action_idx, action in enumerate(actions):
            for instance in vid_annot[action]:
                if (instance['flags']['isReflection'] == True) or (instance['flags']['isAmbiguous'] == True):
                    continue
                is_object = False
                for keyframe in instance['keyframes']:
                    if len(keyframe['objects']) > 0:
                        is_object = True
                if is_object == False:
                    instance_idx += 1
                    continue
                                
                obj_annotations[split][video]['action_instances'][instance_idx] = {}
                for keyframe in instance['keyframes']:
                    if len(keyframe['objects']) > 0:
                        keyframe_num = keyframe['frameNumber']            
                        obj_annotations[split][video]['action_instances'][instance_idx][keyframe_num] = keyframe['objects']
                    
                instance_idx += 1
    return obj_annotations

def save_am(am_list, am_dict, video_info):

    video_names, instances, center_frames = video_info[0], video_info[1], video_info[2]
    
    for video_name in video_names:
        if video_name not in am_dict:
            am_dict[video_name] = {}
    for idx, instance in enumerate(instances):
        if instance not in am_dict[video_names[idx]]:
            am_dict[video_names[idx]][instance] = {}
    for idx, center_frame in enumerate(center_frames):
        if center_frame not in am_dict[video_names[idx]][instances[idx]]:
            am_dict[video_names[idx]][instances[idx]][center_frame] = {}
    
    for layer in range(len(am_list)):
        for graph in range(len(am_list[layer])):
            for idx, adj_matrix in enumerate(am_list[layer][graph]):
                am_dict[video_names[idx]][instances[idx]][center_frames[idx]][str(layer) + str(graph)] = adj_matrix.cpu().numpy()
    return(am_dict)
