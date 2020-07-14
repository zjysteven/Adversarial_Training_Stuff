import os, json, argparse, logging, random
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

    size = (args.subset_num if args.subset_num > 0 else total_sample_num, args.steps)
    fosc = torch.zeros(size)
    losses = torch.zeros(size)
    confs = torch.zeros((size[0], size[1]+1))

    gnorm = torch.zeros(size)
    tensor_norm = torch.zeros(size)
    cossim = torch.zeros(size)

    total = 0
    correct = 0
    adv_correct = [0 for _ in range(steps)]
    criterion = nn.CrossEntropyLoss(reduction='none')
    cos = nn.CosineSimilarity(dim=1)

    for inp, lbl in tqdm(testloader, desc='Batch', leave=False, position=0):
        inp, lbl = inp.cuda(), lbl.cuda()
        
        # clean correct
        with torch.no_grad():
            outputs = model(inp)
            probs = F.softmax(outputs, dim=-1)
            _, preds = probs.max(1)
            correct += preds.eq(lbl).sum().item()
            confs[total:total+inp.shape[0], 0] = torch.gather(probs, 1, lbl.view(-1, 1)).detach().view(-1).cpu()

        x_nat = inp.clone().detach()
        x_adv = None
    
        # random start
        x_adv = inp.clone().detach() + torch.FloatTensor(inp.shape).uniform_(-eps, eps).cuda()
        x_adv = torch.clamp(x_adv, 0., 1.) # respect image bounds

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
                adv_correct[i-1] += preds.eq(lbl).sum().item()
                confs[total:total+inp.shape[0], i] = torch.gather(probs, 1, lbl.view(-1, 1)).detach().view(-1).cpu()           

                # record loss
                losses[total:total+inp.shape[0], i-1] = loss.detach().cpu()

                # record fosc
                grad_flatten = grad.view(grad.shape[0], -1)
                grad_norm = torch.norm(grad_flatten, 1, dim=1)
                diff = (x_adv.clone().detach() - inp).view(inp.shape[0], -1)
                fosc_value = eps * grad_norm - (grad_flatten * diff).sum(dim=1)
                fosc[total:total+inp.shape[0], i-1] = fosc_value.cpu()

                # another way to compute fosc
                tensor = (eps * grad.sign() - (x_adv.clone().detach() - inp)).view(grad.shape[0], -1)
                tensor_norm_l2 = torch.norm(tensor, 2, dim=1)
                grad_norm_l2 = torch.norm(grad_flatten, 2, dim=1)
                cosine = cos(tensor, grad_flatten)
                fosc_value_ = tensor_norm_l2 * grad_norm_l2 * cosine
                assert torch.all(torch.abs(fosc_value_ - fosc_value) < 1e-4)

                gnorm[total:total+inp.shape[0], i-1] = grad_norm_l2.cpu()
                tensor_norm[total:total+inp.shape[0], i-1] = tensor_norm_l2.cpu()
                cossim[total:total+inp.shape[0], i-1] = cosine.cpu()
                
            with torch.no_grad():
                sign_data_grad = grad.sign()
                x_adv = x_adv + alpha * sign_data_grad
                x_adv = torch.max(torch.min(x_adv, x_nat+eps), x_nat-eps)
                x_adv = torch.clamp(x_adv, 0., 1.)

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
        adv_correct[-1] += preds.eq(lbl).sum().item()
        confs[total:total+inp.shape[0], -1] = torch.gather(probs, 1, lbl.view(-1, 1)).detach().view(-1).cpu()              

        # record loss
        losses[total:total+inp.shape[0], -1] = loss.detach().cpu()

        # record fosc
        grad_flatten = grad.view(grad.shape[0], -1)
        grad_norm = torch.norm(grad_flatten, 1, dim=1)
        diff = (x_adv.clone().detach() - inp).view(inp.shape[0], -1)
        fosc_value = eps * grad_norm - (grad_flatten * diff).sum(dim=1)
        fosc[total:total+inp.shape[0], -1] = fosc_value.cpu()
        
        # another way to compute fosc
        tensor = (eps * grad.sign() - (x_adv.clone().detach() - inp)).view(grad.shape[0], -1)
        tensor_norm_l2 = torch.norm(tensor, 2, dim=1)
        grad_norm_l2 = torch.norm(grad_flatten, 2, dim=1)
        cosine = cos(tensor, grad_flatten)
        fosc_value_ = tensor_norm_l2 * grad_norm_l2 * cosine
        assert torch.all(torch.abs(fosc_value_ - fosc_value) < 1e-4)

        gnorm[total:total+inp.shape[0], -1] = grad_norm_l2.cpu()
        tensor_norm[total:total+inp.shape[0], -1] = tensor_norm_l2.cpu()
        cossim[total:total+inp.shape[0], -1] = cosine.cpu()

        total += inp.shape[0]
    
    assert total == size[0]

    to_print = np.zeros((7, steps))
    to_print[0] = np.array([100.*x/total for x in adv_correct])
    to_print[1] = confs.mean(dim=0).numpy()[1:]
    to_print[2] = losses.mean(dim=0).numpy()
    to_print[3] = fosc.mean(dim=0).numpy()
    to_print[4] = gnorm.mean(dim=0).numpy()
    to_print[5] = tensor_norm.mean(dim=0).numpy()
    to_print[6] = cossim.mean(dim=0).numpy()
    np.set_printoptions(suppress=True,
        formatter={'float_kind':'{:2.3f}'.format}, linewidth=130)

    print('Model:\t%s'%args.model_file)
    print('Clean acc: {:.2%}'.format(correct/total))
    pprint(to_print)
    print('\n')

    if args.save:
        robustness = [100.*correct/total]
        robustness.extend([100.*x/total for x in adv_correct])
        assert len(robustness) == args.steps + 1

        # save results
        save_root = args.model_file.replace('state_dicts', 'fosc_eval')
        save_root = save_root.replace('.pth', '')
        if not os.path.exists(save_root):
            os.makedirs(save_root)

        save_fn = lambda filename, mat: np.save(
            os.path.join(save_root, '.'.join((filename, 'npy'))),
            mat)

        base_filename = '%s_%d_eps_%d_alpha_%d_steps_%d' % (
            'train' if args.trainset else 'test',
            size[0], args.eps, args.alpha, args.steps
        )

        save_fn(base_filename+'_conf', confs.numpy())
        save_fn(base_filename+'_loss', losses.numpy())
        save_fn(base_filename+'_fosc', fosc.numpy())
        save_fn(base_filename+'_acc', np.array(robustness))
        save_fn(base_filename+'_grad_norm', gnorm.numpy())
        save_fn(base_filename+'_tensor_norm', tensor_norm.numpy())
        save_fn(base_filename+'_cossim', cossim.numpy())


if __name__ == '__main__':
    main()








