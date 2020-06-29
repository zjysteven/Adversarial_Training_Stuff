import os, argparse
import numpy as np
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import random


def get_args():
    parser = argparse.ArgumentParser(description='Plots for monitoring the training', add_help=True)
    parser.add_argument('--arch', default='WideResNet', type=str)
    parser.add_argument('--depth', default=34, type=int)
    parser.add_argument('--width', default=10, type=int)
    parser.add_argument('--seed', default=233, type=int)
    parser.add_argument('--which', default='cat', choices=['madry', 'cat', 'fast'])
    parser.add_argument('--subdir', type=str)
    parser.add_argument('--no-plot-eps', action='store_false', dest='plot_eps')
    parser.add_argument('--no-plot-loss', action='store_false', dest='plot_loss')
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    root = '{:s}{:d}_{:d}'.format(args.arch, args.depth, args.seed)
    root = os.path.join(root, args.which, args.subdir)

    eps_root = os.path.join(root, 'eps')
    loss_root = os.path.join(root, 'loss')

    if args.plot_eps:
        assert os.path.exists(eps_root), "eps directory [%s] doesn't exist..." % eps_root
    if args.plot_loss:
        assert os.path.exists(loss_root), "loss directory [%s] doesn't exist..." % loss_root


to_load = sorted(glob.glob(os.path.join(root, '*.npy')), key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-1]))
save_path = root + '_plot.png'

random.seed(0)
sample_num = 2
sample_idx = random.sample(range(50000), sample_num)
epochs = len(to_load)

eps_vs_epoch = np.zeros((sample_num, epochs))

for ii, path in enumerate(tqdm(to_load)):
    eps_vs_epoch[:, ii] = np.load(path)[sample_idx]


fig, ax = plt.subplots()
for ii in range(sample_num):
    ax.plot(range(epochs), eps_vs_epoch[ii, :])

plt.savefig(save_path)
plt.close()
