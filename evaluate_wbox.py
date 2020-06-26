import os, json, argparse, hashlib, logging, random
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
from tqdm import tqdm
import pandas as pd

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from advertorch.attacks import GradientSignAttack, LinfBasicIterativeAttack, LinfPGDAttack
from advertorch.attacks.utils import multiple_mini_batch_attack, attack_whole_dataset
from advertorch.utils import to_one_hot

import arguments
import utils


class CarliniWagnerLoss(nn.Module):
    """
    Carlini-Wagner Loss: objective function #6.
    Paper: https://arxiv.org/pdf/1608.04644.pdf
    """

    def __init__(self, conf=50.):
        super(CarliniWagnerLoss, self).__init__()
        self.conf = conf

    def forward(self, input, target):
        """
        :param input: pre-softmax/logits.
        :param target: true labels.
        :return: CW loss value.
        """
        num_classes = input.size(1)
        label_mask = to_one_hot(target, num_classes=num_classes).float()
        correct_logit = torch.sum(label_mask * input, dim=1)
        wrong_logit = torch.max((1. - label_mask) * input, dim=1)[0]
        loss = -F.relu(correct_logit - wrong_logit + self.conf).sum()
        return loss


def get_args():
    parser = argparse.ArgumentParser(description='Evaluation of Models with Advertorch', add_help=True)
    arguments.model_args(parser)
    arguments.data_args(parser)
    arguments.wbox_eval_args(parser)
    args = parser.parse_args()
    return args


def main():
    # get args
    args = get_args()

    # set up gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    assert torch.cuda.is_available()

    # load model
    model = utils.get_model(args, train=False, model_file=args.model_file)

    # get data loaders
    if args.subset_num > 0:
        random.seed(0)
        subset_idx = random.sample(range(10000), args.subset_num)
        testloader = utils.get_testloader(args, batch_size=200, shuffle=False, subset_idx=subset_idx)
    else:
        testloader = utils.get_testloader(args, batch_size=200, shuffle=False)

    """
    # BIM
    test_iter = tqdm(testloader, desc='BIM', leave=False, position=0)
    BIM = LinfBasicIterativeAttack(
            ensemble, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.01, 
            nb_iter=20, eps_iter=0.01/5, clip_min=0., clip_max=1., targeted=False)

    _, label, pred, advpred = attack_whole_dataset(BIM, test_iter, device="cuda")

    print("Accuracy: {:.2f}%, BIM Accuracy: {:.2f}%".format(
        100. * (label == pred).sum().item() / len(label),
        100. * (label == advpred).sum().item() / len(label)))
    """

    loss_fn = nn.CrossEntropyLoss() if args.loss_fn == 'xent' else CarliniWagnerLoss(conf=args.cw_conf)

    rob = {}
    rob['sample_num'] = args.subset_num if args.subset_num else 10000
    rob['loss_fn'] = 'xent' if args.loss_fn == 'xent' else 'cw_{:.1f}'.format(args.cw_conf)

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
        print("Accuracy: {:.2f}%, FGSM Accuracy: {:.2f}%".format(
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

            #if rob[str(eps)] < 0.1:
            #    zero_acc_flag = True
            #    break
        
        # save to file
        if args.save_to_csv:
            output_root = os.path.join('wbox_results', args.model_file.split('/')[1], 'convergence_check')
            #if args.subset_num:
            #    output_root = os.path.join(output_root, 'subset')
            if not os.path.exists(output_root):
                os.makedirs(output_root)
            output_filename = args.model_file.split('/')[-2]
            #if args.subset_num:
            #    output_filename += '_subset'
            output = os.path.join(output_root, '.'.join((output_filename, 'csv')))

            df = pd.DataFrame(rob, index=[0])
            if args.append_out and os.path.isfile(output):
                with open(output, 'a') as f:
                    f.write('\n')
                #with open(output, 'a') as f:
                #    df.to_csv(f, sep=',', header=f.tell()==0, index=False)
                df.to_csv(output, sep=',', mode='a', header=False, index=False, float_format='%.2f')
            else:
                df.to_csv(output, sep=',', index=False, float_format='%.2f')

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
            output_root = os.path.join('wbox_results', args.model_file.split('/')[1])
            #if args.subset_num:
            #    output_root = os.path.join(output_root, 'subset')
            if not os.path.exists(output_root):
                os.makedirs(output_root)
            output_filename = args.model_file.split('/')[-2] + '_epoch_%s' % (args.model_file.split('/')[-1].split('_')[-1].split('.')[0])
            #if args.subset_num:
            #    output_filename += '_subset'
            output = os.path.join(output_root, '.'.join((output_filename, 'csv')))

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








