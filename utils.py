import os, random, copy
from PIL import Image
from collections import OrderedDict
import numpy as np
try:
    from apex import amp
except ModuleNotFoundError:
    pass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR10

from advertorch.utils import NormalizeByChannelMeanStd, to_one_hot
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
    elif args.arch == 'pre_resnet':
        if args.depth in [18, 34, 50, 101, 152]:
            model = resnet_imagenet.preact_resnet(depth=args.depth)
        else:
            raise ValueError('Depth %d is not valid for PreActResNet...' % args.depth)
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
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=args.momentum,
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

    
    # adapted from https://github.com/MadryLab/robustness/blob/master/robustness/model_utils.py
    def register_layers(self, layer_names):
        '''
        Args:
            layers (list of layer names): list of layer names that are of interest
        
        self.layers (dict of layer names and functions): where each function,
                when applied to submod, returns a desired layer. For example, one
                element could be `'layer1': lambda model: model.layer1`.
        '''
        # layers must be in order
        layer_dict = {}

        def hook(module, _, output):
            module.register_buffer('activations', output)
            
        last_name = None
        for name in layer_names:
            temp = name.split('.')
            temp_new = []
            for i, t in enumerate(temp):
                if t.isdigit():
                    temp_new[i-1] += '[%s]' % t
                else:
                    temp_new.append(t)
            name_new = '.'.join(temp_new)

            layer_fn = lambda model, new_name: eval('.'.join(('model', new_name)))
            layer = layer_fn(self.model, name_new)
            
            layer.register_forward_hook(hook)
            layer_dict[name_new] = layer_fn
            
        setattr(self, 'layers', layer_dict)
    
    """
    def register_layers(self, layers):
        setattr(self, 'layers', layers)

        for layer_func in layers:
            layer = layer_func(self.model)
            def hook(module, _, output):
                module.register_buffer('activations', output)

            layer.register_forward_hook(hook)
    """
    def extract_features(self, inp):        
        x = self.normalizer(inp)
        out = self.model(x)
        activs = {layer_name: layer_fn(self.model, layer_name).activations for layer_name, layer_fn in self.layers.items()}
        #activs = [layer_fn(self.model).activations for layer_fn in self.layers]
        activs['output'] = out
        return activs


###################################
# Set up data loader              #
###################################
def get_train_loaders(args):
    kwargs = {'num_workers': 4,
              'batch_size': args.batch_size,
              'shuffle': True,
              'pin_memory': True}
    if args.data_aug:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    else:
        transform_train = transforms.Compose([
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


def get_loader(args, train=False, batch_size=100, shuffle=False, subset_idx=None, augmentation=False):
    kwargs = {'num_workers': 4,
              'batch_size': batch_size,
              'shuffle': shuffle,
              'pin_memory': True}
    if augmentation:
        transform_test = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    else:
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
    if subset_idx is not None:
        testset = Subset(datasets.CIFAR10(root=args.data_dir, train=train,
                                transform=transform_test,
                                download=False), subset_idx)
    else:
        testset = datasets.CIFAR10(root=args.data_dir, train=train,
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


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


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


class CarliniWagnerLoss(nn.Module):
    """
    Carlini-Wagner Loss: objective function #6.
    Paper: https://arxiv.org/pdf/1608.04644.pdf
    """

    def __init__(self, conf=50., reduction="sum"):
        super(CarliniWagnerLoss, self).__init__()
        self.conf = conf
        self.reduction = reduction

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
        loss = -F.relu(correct_logit - wrong_logit + self.conf)

        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError("Reduction type '{:s}' is not supported!".format(self.reduction))


def trades_loss(model,
                x_natural,
                y,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                distance='l_inf'):
    # define KL-loss
    #criterion_kl = nn.KLDivLoss(size_average=False)
    criterion_kl = nn.KLDivLoss(reduction='batchmean')
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(model(x_natural), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1),
                                           F.softmax(model(x_natural), dim=1))
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    #x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    x_adv = x_adv.clone().detach()
    # zero gradient
    #optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_natural)
    loss_natural = F.cross_entropy(logits, y)
    #loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
    #                                                F.softmax(model(x_natural), dim=1))
    loss_robust = criterion_kl(F.log_softmax(model(x_adv), dim=1), F.softmax(model(x_natural), dim=1))
    loss = loss_natural + beta * loss_robust
    return loss


if __name__ == '__main__':
    logits = torch.Tensor([[10, -10, 20]])
    target = torch.LongTensor([2])
    one_hot_target = torch.Tensor([[0,0,1]])

    loss1 = CE_with_soft_label()(logits, one_hot_target)
    loss2 = nn.CrossEntropyLoss()(logits, target)

    print(loss1)
    print(loss2)