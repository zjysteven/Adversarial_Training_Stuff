import sys, os, json, argparse, logging, random
sys.path.append(os.getcwd())
import pandas as pd
import warnings
from tqdm import tqdm

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from advertorch.attacks import GradientSignAttack, LinfBasicIterativeAttack, LinfPGDAttack
from advertorch.attacks.utils import multiple_mini_batch_attack, attack_whole_dataset

import arguments
import utils


def get_args():
    parser = argparse.ArgumentParser(description='Evaluation of Models with Advertorch', add_help=True)
    arguments.model_args(parser)
    arguments.data_args(parser)
    arguments.base_eval_args(parser)
    arguments.wbox_eval_args(parser)
    args = parser.parse_args()
    return args


def main():
    # get args
    args = get_args()
    if args.save_adv:
        assert args.benchmark, 'We only save adversarial examples when benchmarking the robustness'

    # set up gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    assert torch.cuda.is_available()

    # load model
    model = utils.setup(args, train=False, model_file=args.model_file)

    total_sample_num = 50000 if args.trainset else 10000
    # get data loaders
    if args.subset_num > 0:
        random.seed(0)
        subset_idx = random.sample(range(total_sample_num), args.subset_num)
        testloader = utils.get_loader(args, train=args.trainset, batch_size=args.batch_size, shuffle=False, subset_idx=subset_idx)
    else:
        #testloader = utils.get_loader(args, train=args.trainset, batch_size=1000, shuffle=False, augmentation=True)
        testloader = utils.get_loader(args, train=args.trainset, batch_size=args.batch_size, shuffle=False)

    loss_fn = nn.CrossEntropyLoss() if args.loss_fn == 'xent' else utils.CarliniWagnerLoss(conf=args.cw_conf)

    rob = {}
    rob['sample'] = 'train' if args.trainset else 'test'
    rob['sample_num'] = args.subset_num if args.subset_num else total_sample_num
    rob['loss_fn'] = 'xent' if args.loss_fn == 'xent' else 'cw_{:.1f}'.format(args.cw_conf)

    if args.save_to_csv or args.save_adv:
        output_root = args.model_file.replace('state_dicts', 'wbox_results')
        output_root = output_root.replace('.pth', '')

        if not os.path.exists(output_root):
            os.makedirs(output_root)

    if args.convergence_check:
        eps = 0.03
        steps_list = [50, 500, 1000]
        random_start = 1

        rob['random_start'] = random_start
        rob['eps'] = eps

        # FGSM
        test_iter = tqdm(testloader, desc='FGSM', leave=False, position=0)
        adversary = GradientSignAttack(
            model, loss_fn=nn.CrossEntropyLoss(), eps=eps, 
            clip_min=0., clip_max=1., targeted=False)
        _, label, pred, advpred = attack_whole_dataset(adversary, test_iter, device="cuda")
        tqdm.write("Accuracy: {:.2f}%, FGSM Accuracy: {:.2f}%".format(
            100. * (label == pred).sum().item() / len(label),
            100. * (label == advpred).sum().item() / len(label)))
        rob['clean'] = 100. * (label == pred).sum().item() / len(label)
        rob['fgsm'] = 100. * (label == advpred).sum().item() / len(label)
        
        #zero_acc_flag = False
        for steps in tqdm(steps_list, desc='PGD steps', leave=False, position=0):            
            correct_or_not = []

            for i in tqdm(range(random_start), desc='Random Start', leave=False, position=1):
                torch.manual_seed(i)
                test_iter = tqdm(testloader, desc='Batch', leave=False, position=2)

                adversary = LinfPGDAttack(
                    model, loss_fn=loss_fn, eps=eps,
                    nb_iter=steps, eps_iter=eps/5, rand_init=True, clip_min=0., clip_max=1.,
                    targeted=False)
                
                _, label, pred, advpred = attack_whole_dataset(adversary, test_iter, device="cuda")

                correct_or_not.append(label == advpred)
            
            correct_or_not = torch.stack(correct_or_not, dim=-1).all(dim=-1)

            tqdm.write("Accuracy: {:.2f}%, steps: {:d}, PGD Accuracy: {:.2f}%".format(
                100. * (label == pred).sum().item() / len(label),
                steps,
                100. * correct_or_not.sum().item() / len(label)))
            
            rob[str(steps)] = 100. * correct_or_not.sum().item() / len(label)
        
        # save to file
        if args.save_to_csv:
            output = os.path.join(output_root, 'convergence.csv')

            df = pd.DataFrame(rob, index=[0])
            if args.append_out and os.path.isfile(output):
                with open(output, 'a') as f:
                    f.write('\n')
                #with open(output, 'a') as f:
                #    df.to_csv(f, sep=',', header=f.tell()==0, index=False)
                df.to_csv(output, sep=',', mode='a', header=False, index=False, float_format='%.2f')
            else:
                df.to_csv(output, sep=',', index=False, float_format='%.2f')
    elif args.benchmark:
        rob['random_start'] = args.random_start
        rob['steps'] = args.steps
        eps = 8./255.
        alpha = 2./255.

        correct_or_not = []

        for i in tqdm(range(args.random_start), desc='Random Start', leave=True, position=1):
            torch.manual_seed(i)
            test_iter = tqdm(testloader, desc='Batch', leave=False, position=2)

            adversary = LinfPGDAttack(
                model, loss_fn=loss_fn, eps=eps,
                nb_iter=args.steps, eps_iter=alpha, rand_init=True, clip_min=0., clip_max=1.,
                targeted=False)
                
            adv, label, pred, advpred = attack_whole_dataset(adversary, test_iter, device="cuda")
            correct_or_not.append(label == advpred)
            
        correct_or_not = torch.stack(correct_or_not, dim=-1).all(dim=-1)

        tqdm.write("Accuracy: {:.2f}%, eps: {:.2f}, PGD Accuracy: {:.2f}%".format(
            100. * (label == pred).sum().item() / len(label),
            eps,
            100. * correct_or_not.sum().item() / len(label)))
            
        rob['clean'] = 100. * (label == pred).sum().item() / len(label)
        rob['adv'] = 100. * correct_or_not.sum().item() / len(label)

        # save to file
        if args.save_to_csv:
            output = os.path.join(output_root, 'benchmark.csv')
            df = pd.DataFrame(rob, index=[0])
            if args.append_out and os.path.isfile(output):
                with open(output, 'a') as f:
                    f.write('\n')
                #with open(output, 'a') as f:
                #    df.to_csv(f, sep=',', header=f.tell()==0, index=False)
                df.to_csv(output, sep=',', mode='a', header=False, index=False, float_format='%.2f')
            else:
                df.to_csv(output, sep=',', index=False, float_format='%.2f')
        
        if args.save_adv:
            if args.random_start > 1:
                warnings.warn('Found multiple random starts, \
                    only saving the adversarial examples generated in the last round')
            output = output_root.replace('wbox_results', 'adv_examples')
            if not os.path.exists(output):
                os.makedirs(output)
            filename = '{:s}_{:d}_steps_{:d}.pt'.format(
                'train' if args.trainset else 'test',
                adv.shape[0],
                args.steps
            )
            torch.save({
                'adv_inputs': adv.detach().cpu(), 
                'labels': label.cpu(),
            }, os.path.join(output, filename))
    else:
        eps_list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]

        rob['random_start'] = args.random_start
        rob['steps'] = args.steps
        
        zero_acc_flag = False
        for eps in tqdm(eps_list, desc='PGD eps', leave=True, position=0):
            if zero_acc_flag and args.early_stop:
                rob[str(eps)] = 0
                continue
            
            correct_or_not = []

            for i in tqdm(range(args.random_start), desc='Random Start', leave=False, position=1):
                torch.manual_seed(i)
                test_iter = tqdm(testloader, desc='Batch', leave=False, position=2)

                adversary = LinfPGDAttack(
                    model, loss_fn=loss_fn, eps=eps,
                    nb_iter=args.steps, eps_iter=eps/5, rand_init=True, clip_min=0., clip_max=1.,
                    targeted=False)
                
                _, label, pred, advpred = attack_whole_dataset(adversary, test_iter, device="cuda")

                correct_or_not.append(label == advpred)
            
            correct_or_not = torch.stack(correct_or_not, dim=-1).all(dim=-1)

            tqdm.write("Accuracy: {:.2f}%, eps: {:.2f}, PGD Accuracy: {:.2f}%".format(
                100. * (label == pred).sum().item() / len(label),
                eps,
                100. * correct_or_not.sum().item() / len(label)))
            
            rob['clean'] = 100. * (label == pred).sum().item() / len(label)
            rob[str(eps)] = 100. * correct_or_not.sum().item() / len(label)

            if rob[str(eps)] < 0.01:
                zero_acc_flag = True
        
        # save to file
        if args.save_to_csv:
            output = os.path.join(output_root, 'robustness_vs_eps.csv')

            df = pd.DataFrame(rob, index=[0])
            if args.append_out and os.path.isfile(output):
                with open(output, 'a') as f:
                    f.write('\n')
                #with open(output, 'a') as f:
                #    df.to_csv(f, sep=',', header=f.tell()==0, index=False)
                df.to_csv(output, sep=',', mode='a', header=False, index=False, float_format='%.2f')
            else:
                df.to_csv(output, sep=',', index=False, float_format='%.2f')
    
    


if __name__ == '__main__':
    main()








