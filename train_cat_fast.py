import os, json, argparse, hashlib, logging, random
from tqdm import tqdm
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions.dirichlet as d
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

import arguments
import utils
from attacks import Linf_PGD, Linf_PGD_for_CAT_minhao


# https://arxiv.org/pdf/2002.06789.pdf
class CAT():
    def __init__(self, model, optimizer, scheduler, writer, save_path=None, **kwargs):
        self.model = model
        self.epochs = kwargs['epochs']
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.writer = writer
        self.save_path = save_path
        
        self.criterion = utils.CE_with_soft_label() if kwargs['label_smoothing'] else nn.CrossEntropyLoss()
        self.num_classes = 10
        self.batch_size = kwargs['batch_size']

        # PGD configs
        self.attack_cfg = {'alpha': kwargs['alpha'],
                           'steps': 1,
                           'is_targeted': False,
                           'rand_start': kwargs['rs'],
                           'criterion': self.criterion
                          }
        self.fixed_alpha = kwargs['fixed_alpha']
        if not self.fixed_alpha:
            self.adapt_alpha = kwargs['adapt_alpha']
        self.max_eps = kwargs['eps']
        self.eta = kwargs['eta']
        self.c = kwargs['c']
        self.dirichlet = d.Dirichlet(torch.ones((self.num_classes)))

        self.test_robust = kwargs['test_robust']
        self.prepare_data(**kwargs)
        self.eps = torch.zeros((len(self.trainset))).cuda()
        self.save_eps = kwargs['save_eps']
        self.label_smoothing = kwargs['label_smoothing']

    def prepare_data(self, **kwargs):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.trainset = utils.CIFAR10_cat(root=kwargs['data_dir'], train=True,
                                    transform=transform_train,
                                    download=True)
        self.testset = datasets.CIFAR10(root=kwargs['data_dir'], train=False,
                                    transform=transform_test,
                                    download=True)
        self.trainloader = DataLoader(self.trainset, num_workers=4, batch_size=kwargs['batch_size'], shuffle=True, pin_memory=True)
        self.testloader = DataLoader(self.testset, num_workers=4, batch_size=100, shuffle=False, pin_memory=True)

        if self.test_robust:
            subset_idx = random.sample(range(10000), 1000)
            subset = Subset(datasets.CIFAR10(root=kwargs['data_dir'], train=False,
                                    transform=transform_test,
                                    download=True), subset_idx)
            self.rob_testloader = DataLoader(subset, num_workers=4, batch_size=100, shuffle=False, pin_memory=True)
        else:
            self.rob_testloader = None

    def get_epoch_iterator(self):
        iterator = tqdm(list(range(1, self.epochs+1)), total=self.epochs, desc='Epoch',
                        leave=False, position=1)
        return iterator

    def get_batch_iterator(self):
        iterator = tqdm(self.trainloader, desc='Batch', leave=False, position=2)
        return iterator

    def run(self):
        epoch_iter = self.get_epoch_iterator()
        for epoch in epoch_iter:
            self.train(epoch)
            self.test(epoch)
            self.save(epoch)

    def train(self, epoch):
        self.model.train()

        losses = 0
        #current_lr = self.scheduler.get_last_lr()[0]
        #print(current_lr)
        #exit()
        
        batch_iter = self.get_batch_iterator()
        for inputs, targets, idx in batch_iter:
            inputs, targets = inputs.cuda(), targets.cuda()

            # fetch eps for current batch of samples
            eps_per_sample = self.eps[idx]

            # generate soft label
            if self.label_smoothing:
                one_hot_targets = torch.zeros((inputs.size(0), self.num_classes)).cuda()
                one_hot_targets.scatter_(1, targets.unsqueeze(1), 1)

                s1 = (1 - self.c*eps_per_sample).view(-1,1) * one_hot_targets
                s2 = (self.c*eps_per_sample).view(-1,1) * self.dirichlet.sample([inputs.size(0)]).cuda()
                soft_targets = s1 + s2
            
            # temporarily increase eps
            eps_per_sample += self.eta

            # set alpha
            if not self.fixed_alpha:
                self.attack_cfg['alpha'] = self.adapt_alpha*eps_per_sample/self.attack_cfg['steps']

            # generate adversarial examples
            adv_inputs = Linf_PGD_for_CAT_minhao(self.model, inputs, soft_targets if self.label_smoothing else targets, eps=eps_per_sample, **self.attack_cfg)

            # for those already successful adversarial examples
            # do not increase eps
            #outputs = self.model(adv_inputs)
            #_, predicted = outputs.max(1)
            #wrong_idx = ~(predicted.eq(targets))
            #eps_per_sample[wrong_idx] -= self.eta
            eps_per_sample = utils.Linf_distance(adv_inputs, inputs)

            # make sure eps do not exceed max eps
            eps_per_sample = torch.clamp(eps_per_sample, 0., self.max_eps)

            # update eps
            self.eps[idx] = eps_per_sample

            # generate soft label again
            if self.label_smoothing:
                s1 = (1 - self.c*eps_per_sample).view(-1,1) * one_hot_targets
                s2 = (self.c*eps_per_sample).view(-1,1) * self.dirichlet.sample([inputs.size(0)]).cuda()
                soft_targets = s1 + s2

            # update parameters
            outputs = self.model(adv_inputs)
            # use soft label
            loss = self.criterion(outputs, soft_targets if self.label_smoothing else targets)
            losses += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()            

        self.scheduler.step()

        non_zero_eps = torch.sum(self.eps>0).item()

        print_message = 'Epoch [{:3d}] | Adv Loss: {:.4f}, non-zero eps: {:d}'.format(epoch, losses/len(batch_iter), non_zero_eps)
        tqdm.write(print_message)

        self.scheduler.step()
        self.writer.add_scalar('train/adv_loss', losses/len(batch_iter), epoch)
        self.writer.add_scalar('train/non_zero_eps', non_zero_eps, epoch)

    def test(self, epoch):
        self.model.eval()
        
        total = 0
        clean_loss = 0; clean_correct = 0
        
        with torch.no_grad():
            for inputs, targets in self.testloader:
                inputs, targets = inputs.cuda(), targets.cuda()

                outputs = self.model(inputs)
                clean_loss += nn.CrossEntropyLoss()(outputs, targets).item()
                _, predicted = outputs.max(1)
                clean_correct += predicted.eq(targets).sum().item()

                total += inputs.size(0)

        self.writer.add_scalar('test/clean_loss', clean_loss/len(self.testloader), epoch)
        self.writer.add_scalar('test/clean_acc', 100*clean_correct/total, epoch)

        print_message = 'Evaluation  | Clean Loss {loss:.4f}\tClean Acc {acc:.2%}'.format(
            loss=clean_loss/len(self.testloader), acc=clean_correct/total
        )

        if self.rob_testloader:
            total = 0
            adv_loss = 0; adv_correct = 0

            for inputs, targets in self.rob_testloader:
                inputs, targets = inputs.cuda(), targets.cuda()

                adv_inputs = Linf_PGD(self.model, inputs, targets, eps=8./255., alpha=2./255., steps=10)
                
                with torch.no_grad():
                    outputs = self.model(adv_inputs)
                    adv_loss += nn.CrossEntropyLoss()(outputs, targets).item()
                
                _, predicted = outputs.max(1)
                adv_correct += predicted.eq(targets).sum().item()

                total += inputs.size(0)
            
            self.writer.add_scalar('test/adv_loss', adv_loss/len(self.rob_testloader), epoch)
            self.writer.add_scalar('test/adv_acc', 100*adv_correct/total, epoch)

            print_message += '\tAdv Loss {loss:.4f}\tAdv Acc {acc:.2%}'.format(
                loss=adv_loss/len(self.testloader), acc=adv_correct/total
            )
        
        tqdm.write(print_message)

    def save(self, epoch):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, os.path.join(self.save_path, 'model_'+str(epoch)+'.pth'))
        if self.save_eps:
            to_save = self.eps.cpu().numpy()
            save_path = os.path.join(self.save_path, 'eps')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            np.save(os.path.join(save_path, 'epoch_%d.npy'%epoch), to_save)



def get_args():
    parser = argparse.ArgumentParser(description='CIFAR10 Customized Adversarial Training of Ensemble', add_help=True)
    arguments.model_args(parser)
    arguments.data_args(parser)
    arguments.base_train_args(parser)
    arguments.cat_train_args(parser)
    args = parser.parse_args()
    return args


def main():
    # get args
    args = get_args()

    # set up gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    assert torch.cuda.is_available()

    # set up writer, logger, and save directory for models
    #filename = 'model'
    save_root = os.path.join('checkpoints', 'cat_minhao_fgsm', '{:s}{:d}-{:d}_seed_{:d}_epochs_{:d}_{:s}'.format(args.arch, args.depth, args.width, args.seed, args.epochs, args.lr_sch))
    if args.lr_sch == 'cyclic':
        save_root += '_{:.1f}'.format(args.lr_max)
    if args.fixed_alpha:
        save_root += '_fixed_alpha_{:.5f}'.format(args.alpha)
    else:
        save_root += '_adapt_alpha_{:.2f}x'.format(args.adapt_alpha)
    if args.rs:
        save_root += '_rs'
    if not args.label_smoothing:
        save_root += '_no_ls'
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    else:
        print('*********************************')
        print('* The checkpoint already exists *')
        print('*********************************')

    #save_path = os.path.join(save_root, filename)
    writer = SummaryWriter(save_root.replace('checkpoints', 'runs'))

    # dump configurations for potential future references
    with open(os.path.join(save_root, 'cfg.json'), 'w') as fp:
        json.dump(vars(args), fp, indent=4)
    with open(os.path.join(save_root.replace('checkpoints', 'runs'), 'cfg.json'), 'w') as fp:
        json.dump(vars(args), fp, indent=4)

    # set up random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # initialize models
    model = utils.get_model(args, train=True)

    # get optimizers and schedulers
    optimizer, scheduler = utils.get_optimizer_and_scheduler(args, model)

    # train the ensemble
    trainer = CAT(model, optimizer, scheduler, writer, save_root, **vars(args))
    trainer.run()


if __name__ == '__main__':
    main()








