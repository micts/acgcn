import sys
import argparse
import warnings
import time
from config import config
from tools import train

# test this change
# test also this change
# test this too
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', default='training', help='Mode: \'training\' or \'inference\'')
parser.add_argument('-mn', '--model_name', required=True, help='Model name: \'baseline\' or \'gcn\'')
parser.add_argument('-nl', '--num_layers', type=int, help='Number of gcn layers')
parser.add_argument('-ng', '--num_graphs', type=int, help='Number of graphs per layer')
parser.add_argument('-mf', '--merge_function', help='Function to merge output of multiple graphs in final layer: \'sum\' or \'concat\'')
parser.add_argument('-mc', '--model_checkpoint', help='Path to model\'s saved weights (model checkpoint). To be used for inference')
parser.add_argument('-te', '--total_epochs', type=int, help='Total number of epochs')
parser.add_argument('-we', '--warmup_epochs', type=int, default=0, help='Number of epochs to apply linear learning rate warm-up. Valid only when --init_lr is positive float.')
parser.add_argument('-ilr', '--init_lr', type=float, help='Initial learning rate. Valid only when --warmup_epochs > 0.')
parser.add_argument('-mlr', '--max_lr', type=float, help='Maximum learning rate. Equivalent to --init_learning rate when --warmup_epochs=0')
parser.add_argument('-bs', '--batch_size', type=int, default=3, help='Batch size')
parser.add_argument('-zs', '--zero_shot', action='store_true', default=False, help='Exclude certain classes during training; refer to config.py to modify the classes to be exluded')

args = parser.parse_args()

if args.warmup_epochs == 0:
    if args.init_lr is not None:
        warnings.warn("Warning: warmup_epochs is 0, while init_lr is greater than 0.\n Defaulting init_lr to None.")
        args.init_lr = None
        time.sleep(3)
if args.init_lr is None:
    if args.warmup_epochs > 0:
        warnings.warn("Warning: init_lr is None, while warmup_epochs is greater than 0.\n Defaulting warmup_epochs to 0.")
        args.warmup_epochs = 0
        time.sleep(3)


cfg = config.Config(args.mode,
                    args.model_name,
                    args.num_layers,
                    args.num_graphs,
                    args.merge_function,
                    args.model_checkpoint,
                    args.total_epochs,
                    args.warmup_epochs,
                    args.init_lr,
                    args.max_lr,
                    args.batch_size,
                    args.zero_shot)

if args.mode == 'training':
    train.train(cfg)
elif args.mode == 'inference':
    inference.inference(cfg)
else:
    raise ValueError("{} is not a valid mode. Possible values for mode are: \'training\' or \'inference\'.".format(args.mode))
