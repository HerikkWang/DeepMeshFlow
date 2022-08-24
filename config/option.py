import argparse
import yaml
import os

# user setting
parser = argparse.ArgumentParser(description="Options for Deep Image Stitching Training.")
parser.add_argument("--gpu", type=str, default='1', help="choose which gpu to use")
parser.add_argument("--cpu", action="store_true", help="use cpu training")
parser.add_argument("--seed", type=int, default=1, help="set random seed")
parser.add_argument("-e", dest="exp_name", type=str, help="experiment name", default="test")
parser.add_argument("-m", dest="model_name", type=str, help="model name", default="test")
parser.add_argument("-r", dest="resume", action="store_true", help="whether resume training using checkpoints/{exp_name}/{model_name}_latest.pth")
parser.add_argument("--nw", dest="num_workers", type=int, default=8, help="number of workers in dataloader")
parser.add_argument("-c", dest="config", type=str, default="config.yaml", help="config file name")
# parser.add_argument("--bs", type=int, dest="batch_size", default=8, help="set batch size")
# parser.add_argument("--lr", type=float, default=1e-4, help="set learning rate")
args = parser.parse_args()
# yaml config
# with open(os.path.join("config", args.config), 'r') as f:
with open("/home/wyq/DeepMeshFlow/config/config.yaml", 'r') as f:
    config = yaml.load(f.read(), yaml.Loader)
for i in config:
    args.__dict__[i] = config[i]
# print(type(args.lr))
# args.__dict__['test'] = 'test'
# print(args.__dict__)