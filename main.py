import sys
import argparse
import config
import train

parser = argparse.ArgumentParser()
parser.add_argument('-mn', '--model_name', default='gcn', help='model name: \'baseline\' or \'gcn\'')
parser.add_argument('-nl', '--num_layers', type=int, default=2, help='number of gcn layers')
parser.add_argument('-ng', '--num_graphs', type=int, default=2, help='number of graphs per layer')
parser.add_argument('-mf', '--merge_function', default='sum', help='function to merge output of multiple graphs in final layer: \'sum\' or \'concat\'')
parser.add_argument('-zs', '--zero_shot', action='store_true', default=False, help='exclude certain classes during training; refer to config.py to modify the classes to be exluded')

args = parser.parse_args()

cfg = config.Config(args.model_name, args.num_layers, args.num_graphs, args.merge_function, args.zero_shot)
train.train(cfg)
