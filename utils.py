import os, random
from PIL import Image
from collections import OrderedDict
try:
    from apex import amp
except ModuleNotFoundError:
    pass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR10

from advertorch.utils import NormalizeByChannelMeanStd
from models.wrn import WideResNet
import models.resnet_imagenet as resnet_imagenet
import models.resnet_cifar as resnet_cifar


######################################
# Set up model, optimizer, scheduler #
######################################
def setup(args, train=True, model_file=None):
    # initialize model
    if args.arch == 'wrn':
        model = WideResNet(depth=args.depth, widen_factor=args.width)
    elif args.arch == 'resnet':
        if args.depth in [18, 34, 50, 101, 152]:
            model = resnet_imagenet.resnet(depth=args.depth)
        elif args.depth in [20, 32]:
            model = resnet_cifar.resnet(depth=args.depth)
        else:
            raise ValueError('Depth %d is not valid for ResNet...' % args.depth)
    else:
        raise ValueError('Architecture [%s] is not supported yet...' % args.arch)
    
    if model_file:
        ckpt = torch.load(model_file)
        state_dict = ckpt['model_state_dict']
        try:
            model.load_state_dict(state_dict)
        except KeyError:
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)

    # set up the normalizer
    mean = torch.tensor([0.4914, 0.4822, 0.4465], dtype=torch.float32)
    std = torch.tensor([0.2023, 0.1994, 0.2010], dtype=torch.float32)
    normalizer = NormalizeByChannelMeanStd(mean=mean, std=std)
    model = ModelWrapper(model, normalizer).cuda()

    if train:
        model.train()
    else:
        model.eval()
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        return model

    # set up optimizer and scheduler
    model, optimizer, scheduler = get_optimizer_and_scheduler(args, model)

    # nn.DataParallel
    if torch.cuda.device_count() > 1:
        if args.amp:
            assert args.opt_level == 'O1', "Haven't tested opt_level %s with nn.DataParallel..." % args.opt_level
        model = nn.DataParallel(model)

    return model, optimizer, scheduler


def get_optimizer_and_scheduler(args, model):
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
        weight_decay=args.weight_decay)

    if args.amp:
        amp_args = dict(opt_level=args.opt_level, loss_scale=args.loss_scale, verbosity=False)
        if args.opt_level == 'O2':
            amp_args['master_weights'] = args.master_weights
        model, optimizer = amp.initialize(model, optimizer, **amp_args)
    
    if args.lr_sch == 'multistep':
        if len(args.sch_intervals) > 0:
            sch_intervals = [int(e * (50000 // args.batch_size + 1)) for e in args.sch_intervals]
        else:
            lr_steps = args.epochs * (50000 // args.batch_size + 1)
            sch_intervals = [lr_steps//2, (3*lr_steps)//4]
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=sch_intervals, gamma=args.lr_gamma)
    elif args.lr_sch == 'cyclic':
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.lr_min, max_lr=args.lr_max,
            step_size_up=lr_steps//2, step_size_down=lr_steps//2)
    
    return model, optimizer, scheduler


class ModelWrapper(nn.Module):
    def __init__(self, model, normalizer):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.normalizer = normalizer

    def forward(self, x):
        x = self.normalizer(x)
        return self.model(x)


###################################
# Set up data loader              #
###################################
def get_train_loaders(args):
    kwargs = {'num_workers': 4,
              'batch_size': args.batch_size,
              'shuffle': True,
              'pin_memory': True}
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    trainset = datasets.CIFAR10(root=args.data_dir, train=True,
                                transform=transform_train,
                                download=True)
    testset = datasets.CIFAR10(root=args.data_dir, train=False,
                                transform=transform_test,
                                download=True)
    trainloader = DataLoader(trainset, **kwargs)
    testloader = DataLoader(testset, num_workers=4, batch_size=100, shuffle=False, pin_memory=True)
    return trainloader, testloader


def get_test_loader(args, batch_size=100, shuffle=False, subset_idx=None):
    kwargs = {'num_workers': 4,
              'batch_size': batch_size,
              'shuffle': shuffle,
              'pin_memory': True}
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    if subset_idx is not None:
        testset = Subset(datasets.CIFAR10(root=args.data_dir, train=False,
                                transform=transform_test,
                                download=False), subset_idx)
    else:
        testset = datasets.CIFAR10(root=args.data_dir, train=False,
                                transform=transform_test,
                                download=False)
    testloader = DataLoader(testset, **kwargs)
    return testloader


###################################
# CAT related                     #
###################################
class CIFAR10_with_idx(CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        super(CIFAR10_with_idx, self).__init__(root, train=train, transform=transform,
                                      target_transform=target_transform, download=download)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, index) where target is index of the target class,
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


def Linf_distance(x_adv, x):
    diff = (x_adv - x).view(x.size(0), -1)
    out = torch.max(torch.abs(diff), 1)[0]
    return out
    

# https://github.com/pytorch/pytorch/issues/7455
class CE_with_soft_label(nn.Module):
    def __init__(self, reduction="mean"):
        super(CE_with_soft_label, self).__init__()
        self.reduction = reduction

    def forward(self, logits, soft_targets):
        preds = logits.log_softmax(dim=-1)
        assert preds.shape == soft_targets.shape

        loss = torch.sum(-soft_targets * preds, dim=-1)

        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError("Reduction type '{:s}' is not supported!".format(self.reduction))


if __name__ == '__main__':
    logits = torch.Tensor([[10, -10, 20]])
    target = torch.LongTensor([2])
    one_hot_target = torch.Tensor([[0,0,1]])

    loss1 = CE_with_soft_label()(logits, one_hot_target)
    loss2 = nn.CrossEntropyLoss()(logits, target)

    print(loss1)
    print(loss2)