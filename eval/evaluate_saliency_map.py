import sys, os, json, argparse, logging, random
sys.path.append(os.getcwd())
import pandas as pd
import warnings
from tqdm import tqdm

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from advertorch.attacks import GradientSignAttack, LinfBasicIterativeAttack, LinfPGDAttack
from advertorch.attacks.utils import multiple_mini_batch_attack, attack_whole_dataset

import arguments
import utils
from saliency_map import *


def get_args():
    parser = argparse.ArgumentParser(description='Evaluation of Models with Advertorch', add_help=True)
    arguments.model_args(parser)
    arguments.data_args(parser)
    arguments.base_eval_args(parser)
    arguments.saliency_map_args(parser)
    args = parser.parse_args()
    return args


def main():
    # get args
    args = get_args()

    # set up gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    assert torch.cuda.is_available()

    # load model
    model = utils.setup(args, train=False, model_file=args.model_file)
    testloader = utils.get_loader(args, train=args.trainset, batch_size=args.batch_size, shuffle=False)
    saliency_map_loader = utils.get_loader(args, train=args.trainset, batch_size=1, shuffle=False)

    # set up save root
    output_root = args.model_file.replace('state_dicts', 'saliency_maps')
    output_root = output_root.replace('.pth', '')
    subroot = '%s_%s' % ('train' if args.trainset else 'test', 
        'clean' if not args.adversarial else 'adv')
    output_root = os.path.join(output_root, subroot)

    if not os.path.exists(output_root):
        os.makedirs(output_root)

    # set up adversary, if we want
    if args.adversarial:
        adversary = LinfPGDAttack(
            model, loss_fn=nn.CrossEntropyLoss(), eps=args.eps/255.,
            nb_iter=args.steps, eps_iter=args.alpha/255., rand_init=True, 
            clip_min=0., clip_max=1., targeted=False)
    else:
        adversary = None
    
    # test classification
    correct = []
    if adversary:
        adv_samples = []
        labels = []
    for x, y in testloader:
        x, y = x.cuda(), y.cuda()
        if adversary:
            adv = adversary.perturb(x)
            adv_samples.append(adv)
            labels.append(y)
            outputs = model(adv)
        else:
            outputs = model(x)
        _, preds = outputs.max(1)
        correct.append(preds.eq(y))
    correct = torch.cat(correct).cpu().numpy()
    np.save(
        os.path.join(output_root, 'classification.npy'), 
        correct
    )

    # generate saliency map
    smooth_grad = SmoothGrad(stdev_spread=args.stdev_spread, 
        n_samples=args.n_samples, magnitude=args.magnitude)

    all_grad_maps = []

    if adversary:
        adv_samples = torch.cat(adv_samples)
        labels = torch.cat(labels)
        adv_dataset = TensorDataset(adv_samples, labels)
        loader = DataLoader(adv_dataset, batch_size=1, shuffle=False)

        for x, y in tqdm(loader):        
            # idx=y means that we are interested in the 
            # grad w.r.t. the true class
            grad_maps = smooth_grad(model, x, idx=y)
            all_grad_maps.append(grad_maps)
    else:
        for x, y in tqdm(saliency_map_loader):
            x, y = x.cuda(), y.cuda()
        
            # idx=y means that we are interested in the 
            # grad w.r.t. the true class
            grad_maps = smooth_grad(model, x, idx=y)
            all_grad_maps.append(grad_maps)
    
    all_grad_maps = torch.cat(all_grad_maps).cpu().numpy()
    #np.save(
    #    os.path.join(output_root, 'raw_grad.npy'), 
    #    all_grad_maps
    #)

    # normalize grad
    normalize_fn = VisualizeImageGrayscale if args.absolute else VisualizeImageDiverging
    normalized = []
    for grad in all_grad_maps:
        normalized.append(normalize_fn(grad))
    filename = 'normalized_saliency_map'
    if args.absolute:
        filename += '_abs'
    np.save(
        os.path.join(output_root, '%s.npy'%filename),
        np.array(normalized)
    )


if __name__ == '__main__':
    main()








