import sys, os, json, argparse, logging, random
sys.path.append('..')
from tqdm import tqdm
import pandas as pd
import numpy as np
from pprint import pprint

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from advertorch.attacks import GradientSignAttack, LinfBasicIterativeAttack, LinfPGDAttack
from advertorch.attacks.utils import multiple_mini_batch_attack, attack_whole_dataset
from advertorch.utils import to_one_hot

import arguments
import utils


def get_args():
    parser = argparse.ArgumentParser(description='Evaluation of Models with Advertorch', add_help=True)
    arguments.model_args(parser)
    arguments.data_args(parser)
    arguments.base_eval_args(parser)
    arguments.fosc_eval_args(parser)
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

    # set seed
    torch.manual_seed(0)

    # get data loader
    total_sample_num = 50000 if args.trainset else 10000
    if args.subset_num > 0:
        random.seed(0)
        subset_idx = random.sample(range(total_sample_num), args.subset_num)
        testloader = utils.get_loader(args, train=args.trainset, batch_size=args.batch_size, shuffle=False, subset_idx=subset_idx)
    else:
        testloader = utils.get_loader(args, train=args.trainset, batch_size=args.batch_size, shuffle=False)

    # initialization
    eps = args.eps/255.
    alpha = args.alpha/255.
    steps = args.steps

    sample_num = args.subset_num if args.subset_num > 0 else total_sample_num

    def get_dict(clean=False):
        size = (sample_num, steps) if not clean else (sample_num,)
        out = {}

        out['losses'] = torch.zeros(size)
        out['confs'] = torch.zeros(size)
        out['correct'] = torch.zeros(size)
        if not clean:
            out['gradnorm'] = torch.zeros(size)
        else:
            out['grad'] = torch.zeros((sample_num, 3, 32, 32))
        if not clean:
            out['perturbation'] = torch.zeros(
                (sample_num, 3, 32, 32, steps)
            )
        
        return out

    max_dict = get_dict()
    min_dict = get_dict()
    clean_dict = get_dict(True)
    
    criterion = nn.CrossEntropyLoss(reduction='none')
    total = 0
    for inp, lbl in tqdm(testloader, desc='Batch', leave=False, position=0):
        inp, lbl = inp.cuda(), lbl.cuda()
        
        # clean
        with torch.no_grad():
            outputs = model(inp)
            loss = criterion(outputs, lbl)
            clean_dict['losses'][total:total+inp.shape[0]] = loss.detach().cpu()
            probs = F.softmax(outputs, dim=-1)
            _, preds = probs.max(1)
            clean_dict['correct'][total:total+inp.shape[0]] = preds.eq(lbl).float().cpu()
            clean_dict['confs'][total:total+inp.shape[0]] = torch.gather(probs, 1, lbl.view(-1, 1)).detach().view(-1).cpu()

        x_nat = inp.clone().detach()

        ################################################################################################################## 
        x_adv = inp.clone().detach()

        # iteratively perturb data
        for i in range(steps):
            # Calculate gradient w.r.t. data
            x_adv.requires_grad = True
            outputs = model(x_adv)

            loss = criterion(outputs, lbl)
            model.zero_grad()

            loss.sum().backward()
            grad = x_adv.grad.data

            if i == 0:
                clean_dict['grad'][total:total+inp.shape[0]] = grad.cpu()

            if i > 0:
                # record correct and confidence
                probs = F.softmax(outputs, dim=-1)
                _, preds = probs.max(1)
                max_dict['correct'][total:total+inp.shape[0], i-1] = preds.eq(lbl).float().cpu()
                max_dict['confs'][total:total+inp.shape[0], i-1] = torch.gather(probs, 1, lbl.view(-1, 1)).detach().view(-1).cpu()           

                # record loss
                max_dict['losses'][total:total+inp.shape[0], i-1] = loss.detach().cpu()

                # record fosc
                grad_flatten = grad.view(grad.shape[0], -1)
                grad_norm_l2 = torch.norm(grad_flatten, 2, dim=1)
                max_dict['gradnorm'][total:total+inp.shape[0], i-1] = grad_norm_l2.cpu()
                
            with torch.no_grad():
                sign_data_grad = grad.sign()
                # maximize loss
                x_adv = x_adv + alpha * sign_data_grad
                x_adv = torch.max(torch.min(x_adv, x_nat+eps), x_nat-eps)
                x_adv = torch.clamp(x_adv, 0., 1.)
            
            max_dict['perturbation'][total:total+inp.shape[0],:,:,:,i-1] = (x_adv.detach() - x_nat).cpu()

        # record statistics for the final x_adv
        x_adv.requires_grad = True
        outputs = model(x_adv)
        loss = criterion(outputs, lbl)
        model.zero_grad()

        loss.sum().backward()
        grad = x_adv.grad.data

        # record correct and confidence
        probs = F.softmax(outputs, dim=-1)
        _, preds = probs.max(1)
        max_dict['correct'][total:total+inp.shape[0], -1] = preds.eq(lbl).float().cpu()
        max_dict['confs'][total:total+inp.shape[0], -1] = torch.gather(probs, 1, lbl.view(-1, 1)).detach().view(-1).cpu()              

        # record loss
        max_dict['losses'][total:total+inp.shape[0], -1] = loss.detach().cpu()

        # record fosc
        grad_flatten = grad.view(grad.shape[0], -1)
        grad_norm_l2 = torch.norm(grad_flatten, 2, dim=1)
        max_dict['gradnorm'][total:total+inp.shape[0], -1] = grad_norm_l2.cpu()
        
        max_dict['perturbation'][total:total+inp.shape[0],:,:,:,-1] = (x_adv.detach() - inp).cpu()
        ##################################################################################################################

        ################################################################################################################## 
        x_adv = inp.clone().detach()

        # iteratively perturb data
        for i in range(steps):
            # Calculate gradient w.r.t. data
            x_adv.requires_grad = True
            outputs = model(x_adv)

            loss = criterion(outputs, lbl)
            model.zero_grad()

            loss.sum().backward()
            grad = x_adv.grad.data

            if i > 0:
                # record correct and confidence
                probs = F.softmax(outputs, dim=-1)
                _, preds = probs.max(1)
                min_dict['correct'][total:total+inp.shape[0], i-1] = preds.eq(lbl).float().cpu()
                min_dict['confs'][total:total+inp.shape[0], i-1] = torch.gather(probs, 1, lbl.view(-1, 1)).detach().view(-1).cpu()           

                # record loss
                min_dict['losses'][total:total+inp.shape[0], i-1] = loss.detach().cpu()

                # record fosc
                grad_flatten = grad.view(grad.shape[0], -1)
                grad_norm_l2 = torch.norm(grad_flatten, 2, dim=1)
                min_dict['gradnorm'][total:total+inp.shape[0], i-1] = grad_norm_l2.cpu()
                
            with torch.no_grad():
                sign_data_grad = grad.sign()
                # minimize loss
                x_adv = x_adv - alpha * sign_data_grad
                x_adv = torch.max(torch.min(x_adv, x_nat+eps), x_nat-eps)
                x_adv = torch.clamp(x_adv, 0., 1.)
            
            min_dict['perturbation'][total:total+inp.shape[0],:,:,:,i-1] = (x_adv.detach() - x_nat).cpu()

        # record statistics for the final x_adv
        x_adv.requires_grad = True
        outputs = model(x_adv)
        loss = criterion(outputs, lbl)
        model.zero_grad()

        loss.sum().backward()
        grad = x_adv.grad.data

        # record correct and confidence
        probs = F.softmax(outputs, dim=-1)
        _, preds = probs.max(1)
        min_dict['correct'][total:total+inp.shape[0], -1] = preds.eq(lbl).float().cpu()
        min_dict['confs'][total:total+inp.shape[0], -1] = torch.gather(probs, 1, lbl.view(-1, 1)).detach().view(-1).cpu()              

        # record loss
        min_dict['losses'][total:total+inp.shape[0], -1] = loss.detach().cpu()

        # record fosc
        grad_flatten = grad.view(grad.shape[0], -1)
        grad_norm_l2 = torch.norm(grad_flatten, 2, dim=1)
        min_dict['gradnorm'][total:total+inp.shape[0], -1] = grad_norm_l2.cpu()
        
        min_dict['perturbation'][total:total+inp.shape[0],:,:,:,-1] = (x_adv.detach() - inp).cpu()
        ##################################################################################################################

        total += inp.shape[0]
    
    assert total == sample_num

    """
    to_print = np.zeros((7, steps))
    to_print[0] = np.array([100.*x/total for x in adv_correct.sum(dim=0).cpu().numpy()])
    to_print[1] = confs.mean(dim=0).numpy()[1:]
    to_print[2] = losses.mean(dim=0).numpy()
    to_print[3] = fosc.mean(dim=0).numpy()
    to_print[4] = gnorm.mean(dim=0).numpy()[1:]
    to_print[5] = tensor_norm.mean(dim=0).numpy()
    to_print[6] = cossim.mean(dim=0).numpy()
    np.set_printoptions(suppress=True,
        formatter={'float_kind':'{:2.3f}'.format}, linewidth=130)

    print('Model:\t%s'%args.model_file)
    print('Clean acc: {:.2%}'.format(correct.sum().item()/total))
    pprint(to_print)
    print('\n')
    """

    if args.save:
        # save results
        save_root = args.model_file.replace('state_dicts', 'explore')
        save_root = save_root.replace('.pth', '')
        base_root = '%s_%d_eps_%d_alpha_%d_steps_%d' % (
            'train' if args.trainset else 'test',
            sample_num, args.eps, args.alpha, args.steps
        )
        save_root = os.path.join(save_root, base_root)
        if not os.path.exists(save_root):
            os.makedirs(save_root)

        save_fn = lambda filename, dict_name, dict, key: np.save(
            os.path.join(save_root, '.'.join((
                '_'.join((dict_name, filename)), 'npy'))
                ),
            dict[key].numpy())

        save_fn('conf', 'clean', clean_dict, 'confs')
        save_fn('loss', 'clean', clean_dict, 'losses')
        save_fn('correct', 'clean', clean_dict, 'correct')
        save_fn('grad', 'clean', clean_dict, 'grad')

        save_fn('conf', 'max', max_dict, 'confs')
        save_fn('loss', 'max', max_dict, 'losses')
        save_fn('correct', 'max', max_dict, 'correct')
        save_fn('gradnorm', 'max', max_dict, 'gradnorm')
        save_fn('perturbation', 'max', max_dict, 'perturbation')

        save_fn('conf', 'min', min_dict, 'confs')
        save_fn('loss', 'min', min_dict, 'losses')
        save_fn('correct', 'min', min_dict, 'correct')
        save_fn('gradnorm', 'min', min_dict, 'gradnorm')
        save_fn('perturbation', 'min', min_dict, 'perturbation')
       
        


if __name__ == '__main__':
    main()








