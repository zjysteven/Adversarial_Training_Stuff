import os, json, argparse, logging, random
from tqdm import tqdm
import numpy as np
try:
    from apex import amp
except ModuleNotFoundError:
    pass

import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

import arguments
import utils


class Natural():
    def __init__(self, args, model, optimizer, scheduler, writer, save_path=None, **kwargs):
        self.model = model
        self.epochs = args.epochs
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.writer = writer
        self.save_path = save_path
        
        self.criterion = nn.CrossEntropyLoss()
        self.prepare_data(args)
        self.use_amp = args.amp

    def prepare_data(self, args):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.trainset = utils.CIFAR10_with_idx(root=args.data_dir, train=True,
                                    transform=transform_train,
                                    download=True)
        self.testset = datasets.CIFAR10(root=args.data_dir, train=False,
                                    transform=transform_test,
                                    download=True)
        self.trainloader = DataLoader(self.trainset, num_workers=4, batch_size=args.batch_size, shuffle=True, pin_memory=True)
        self.testloader = DataLoader(self.testset, num_workers=4, batch_size=100, shuffle=False, pin_memory=True)

    def get_epoch_iterator(self):
        iterator = tqdm(list(range(1, self.epochs+1)), total=self.epochs, desc='Epoch',
                        leave=True, position=1)
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
        correct = 0

        batch_iter = self.get_batch_iterator()
        for inputs, targets, idx in batch_iter:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = self.model(inputs)

            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

            loss = self.criterion(outputs, targets)
            losses += loss.item()

            self.optimizer.zero_grad()
            if self.use_amp:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            self.optimizer.step() 
            self.scheduler.step()        

        print_message = 'Epoch [{:3d}] | Loss: {:.4f}, Acc: {:.2%}'.format(epoch, losses/len(batch_iter), correct/len(self.trainset))
        tqdm.write(print_message)

        self.writer.add_scalar('train/clean_loss', losses/len(batch_iter), epoch)
        self.writer.add_scalar('train/clean_acc', correct/len(self.trainset), epoch)

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
        tqdm.write(print_message)

    def _save(self, attr_name, epoch):
        to_save = eval('self.'+attr_name).numpy()
        save_path = os.path.join(self.save_path, attr_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(os.path.join(save_path, 'epoch_%d.npy'%epoch), to_save)

    def save(self, epoch):
        try:
            state_dict = self.model.model.state_dict()
        except AttributeError:
            state_dict = self.model.module.model.state_dict()
        save_path = os.path.join(self.save_path, 'state_dicts')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save({
            'epoch': epoch,
            'model_state_dict': state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, os.path.join(save_path, 'epoch_'+str(epoch)+'.pth'))


def get_args():
    parser = argparse.ArgumentParser(description='CIFAR10 Natural Training', add_help=True)
    arguments.model_args(parser)
    arguments.data_args(parser)
    arguments.base_train_args(parser)
    arguments.amp_args(parser)
    args = parser.parse_args()
    return args


def main():
    # get args
    args = get_args()

    # set up gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    assert torch.cuda.is_available()

    # set up writer, logger, and save directory for models
    arch = '{:s}{:d}'.format(args.arch, args.depth)
    save_root = os.path.join('checkpoints', arch, 'natural', 'seed_'+str(args.seed))
    subfolder = 'epochs_{:d}_batch_{:d}_lr_{:s}'.format(args.epochs, args.batch_size, args.lr_sch)
    if args.lr_sch == 'cyclic':
        subfolder += '_{:.1f}'.format(args.lr_max)
    if args.amp:
        subfolder += '_%s' % args.opt_level
    save_root = os.path.join(save_root, subfolder)
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    else:
        print('*********************************')
        print('* The checkpoint already exists *')
        print('*********************************')

    writer = SummaryWriter(save_root.replace('checkpoints', 'runs'))

    # dump configurations for potential future references
    with open(os.path.join(save_root, 'cfg.json'), 'w') as fp:
        json.dump(vars(args), fp, indent=4)
    with open(os.path.join(save_root.replace('checkpoints', 'runs'), 'cfg.json'), 'w') as fp:
        json.dump(vars(args), fp, indent=4)

    # set up random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # initialize model, optimizer, and scheduler
    model, optimizer, scheduler = utils.setup(args, train=True)

    # train the model
    trainer = Natural(args, model, optimizer, scheduler, writer, save_root)
    trainer.run()


if __name__ == '__main__':
    main()








