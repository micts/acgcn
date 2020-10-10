from utils import utils

class Config(object):

    def __init__(self,
                 mode,
                 model_name,
                 num_layers,
                 num_graphs,
                 merge_function,
                 model_checkpoint,
                 total_epochs,
                 warmup_epochs,
                 init_lr,
                 max_lr,
                 batch_size,
                 zero_shot):

        self.img_size = 224, 224 # size of input frame (h, w)
        self.out_feature_size = 14, 14 # size of output feature map (Mixed_4f) (h, w)
        self.out_feature_temp_size = 8 # temporal output size of feature map (Mixed_4f)
        self.num_person_boxes = 20 # max number of tracks in action instance
        self.num_in_frames = 32 # number of input frames

        self.model_name = model_name # 'baseline' or 'gcn'

        self.data_path = '../DALY/frames/' # 'data/DALY/frames/
        self.annot_path = 'data/DALY/annotations/'
        self.results_path = 'results/'
        self.scores_path = 'scores/'
        self.am_path = 'am/'
        self.features_path = 'extracted_features/'
        self.i3d_weights_path = 'models/'
        self.filename = ''

        self.use_gpu = True
        if self.use_gpu:
            self.device_list = '0'

        self.num_actions = 11 # (10 + Background)

        self.label_tracks = False # whether to compute track annotations (True), or load them instead (False)

        self.training_batch_size = batch_size # 3
        self.test_batch_size = batch_size # 3

        self.momentum = 0.9
        self.weight_decay = 0

        # learning rate with cosine annealing schedule
        # 'warmup' refers to linear warmp-up
        self.start_epoch = 0
        self.total_epochs = total_epochs # 450 # 'total_steps' is inferred as (per epoch) 'num_steps' * 'total_epochs'
        self.warmup_epochs = warmup_epochs # 0 # 'warmup_steps' is inferred as (per epoch) 'num_steps' * 'warmup_epochs' (0-indexed)
        self.init_lr = init_lr # 4.5e-6
        self.max_lr = max_lr # 4.7e-5
        self.min_lr = 0


        # if self.warmup_epochs == 0:
        #     if self.init_lr is not None:
        #         warnings.warn("Warning: {warmup_epochs} is 0, while init_lr is greater than 0.\n Defaulting init_lr to None".format(s))
        # if self.init_lr is None:
        #     if self.warmup_epochs > 0:
        #         warnings.warn("Warning: {warmup_epochs} is 0, while init_lr is greater than 0.\n Defaulting init_lr to None".format(s))
                # raise warning

        # if self.model_name == 'baseline':
        #     self.total_epochs = 150
        #     self.warmup_epochs = 0
        #     self.init_lr = None
        #     self.max_lr = 2.5e-4
        #     self.min_lr = 0

        self.num_features_mixed4f = 832 # number of output channels of Mixed_4f
        self.num_features_mixed5c = 1024 # number of output channels of Mixed_5c
        self.num_features_gcn = 256
        self.crop_size = 7, 7 # output size of RoI Pooling
        self.dropout_prob = 0.5
        self.num_layers = num_layers # number of gcn layers
        self.num_graphs = num_graphs # number of graphs per layer
        self.merge_function = merge_function # function to merge output of multiple graphs in final layer: 'sum' or 'concat'

        if self.model_name == 'baseline':
            self.use_i3d_tail = True

        self.zero_shot = zero_shot
        self.classes_to_exclude = None
        if self.model_name == 'gcn':
            if self.zero_shot:
                self.classes_to_exclude = ['Ironing', 'TakingPhotosOrVideos'] # classes to exclude during training
                self.num_actions = self.num_actions - len(self.classes_to_exclude)
        elif self.model_name == 'baseline':
            self.zero_shot = False

        self.class_map = utils.class2idx_map(self.classes_to_exclude)

        self.save_log = True # whether to save the model, state_dict, and loss every x-number of epochs
        self.set_bn_eval = False # If set to True, freeze batch normalization layers
        self.save_scores = False # whether to save output (softmax) scores every x-number of epochs
        self.save_am = False # whether to save the adjacency matrix of every clip
        if self.model_name == 'baseline':
            self.save_am = False
        self.plot_grad_flow = False
        self.num_epochs_to_val = 10

        self.load_checkpoint = False # Load weights from checkpoint to resume training
        if self.load_checkpoint:
            self.checkpoint_path = ''
