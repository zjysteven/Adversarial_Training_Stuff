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

import sys
sys.path.append(os.getcwd())
import arguments
import utils
from attacks import Linf_PGD


class Madry():
    def __init__(self, args, model, optimizer, scheduler, trainloader, testloader, 
                 writer, save_path=None, **kwargs):
        self.model = model
        self.epochs = args.epochs
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.trainloader = trainloader
        self.testloader = testloader

        self.writer = writer
        self.save_path = save_path
        
        self.criterion = nn.CrossEntropyLoss(reduction='none')

        # PGD configs
        self.attack_cfg = {'eps': args.eps/255., 
                           'alpha': args.alpha/255.,
                           'steps': args.steps,
                           'is_targeted': False,
                           'rand_start': True,
                           'inner_max': 'madry',
                           'input_diversity': args.input_diversity,
                           'prob': args.id_prob
                          }
        
        self.test_robust = args.test_robust
        #self.prepare_data(args)

        """
        self.eps = torch.zeros((len(self.trainset), 3, 32, 32))
        self.loss = torch.zeros((len(self.trainset)))
        self.correct = torch.zeros((len(self.trainset)))
        self.fosc = torch.zeros((len(self.trainset)))
        self.minus_delta_loss = torch.zeros((len(self.trainset), 2))
        self.minus_delta_correct = torch.zeros((len(self.trainset), 2))
        """
        self.save_eps = args.save_eps
        self.save_loss = args.save_loss
        self.save_fosc = args.save_fosc
        self.save_correct = args.save_correct
        self.save_minus_delta = args.save_minus_delta
        

        self.use_amp = args.amp

        """
        self.increase_steps = args.increase_steps
        self.increase_eps = args.increase_eps
        self.linear_eps = args.linear_eps
        if self.increase_steps:
            assert len(args.more_steps) > 0
            assert len(args.more_steps) == len(args.steps_intervals)
            self.more_steps = np.array(args.more_steps)
            self.steps_intervals = np.array(args.steps_intervals)
        if self.increase_eps:
            assert len(args.more_eps) > 0
            assert len(args.more_eps) == len(args.eps_intervals)
            self.more_eps = np.array(args.more_eps)
            self.eps_intervals = np.array(args.eps_intervals)
        if self.linear_eps:
            self.eps_list = np.linspace(0, args.eps, args.epochs)
        """

        self.clean_coeff = args.clean_coeff

    def prepare_data(self, args):
        t = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]
        if args.cutout:
            t.append(utils.Cutout(n_holes=args.cutout_n_holes, length=args.cutout_length))
        transform_train = transforms.Compose(t)
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

        if self.test_robust:
            subset_idx = random.sample(range(10000), 1000)
            subset = Subset(datasets.CIFAR10(root=args.data_dir, train=False,
                                    transform=transform_test,
                                    download=True), subset_idx)
            self.rob_testloader = DataLoader(subset, num_workers=4, batch_size=100, shuffle=False, pin_memory=True)
        else:
            self.rob_testloader = None

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

        current_lr = self.scheduler.get_last_lr()[0]
        """
        if self.increase_steps:
            current_steps_idx = self.steps_intervals < epoch
            if np.any(current_steps_idx):
                current_steps = self.more_steps[current_steps_idx][-1]
                self.attack_cfg['steps'] = current_steps
        
        if self.increase_eps:
            current_eps_idx = self.eps_intervals < epoch
            if np.any(current_eps_idx):
                current_eps = self.more_eps[current_eps_idx][-1]
                self.attack_cfg['eps'] = float(current_eps / 255.)
        
        if self.linear_eps:
            self.attack_cfg['eps'] = float(self.eps_list[epoch-1] / 255.)
        """
        losses = 0
        correct = 0
        total = 0

        batch_iter = self.get_batch_iterator()
        for batch_data in batch_iter:
            if len(batch_data) == 3:
                inputs, targets, idx = batch_data
            elif len(batch_data) == 2:
                inputs, targets = batch_data

            inputs, targets = inputs.cuda(), targets.cuda()

            if self.clean_coeff < 1:
                adv_return = Linf_PGD(self.model, inputs, targets, 
                    **self.attack_cfg, use_amp=self.use_amp, optimizer=self.optimizer,
                    fosc=self.save_fosc)

                if self.save_fosc:
                    adv_inputs, fosc_val = adv_return
                    #self.fosc[idx] = fosc_val.cpu()
                else:
                    adv_inputs = adv_return
                """
                if self.save_minus_delta:
                    self.model.eval()
                    outputs = self.model(2*inputs-adv_inputs)
                    _, preds = outputs.max(1)
                    loss = self.criterion(outputs, targets)
                    self.minus_delta_loss[idx, 0] = loss.detach().cpu()
                    self.minus_delta_correct[idx, 0] = preds.eq(targets).float().cpu()
                    self.model.train()
                """

            if self.clean_coeff > 0 and self.clean_coeff < 1:
                outputs = self.model(adv_inputs)
                adv_loss = self.criterion(outputs, targets)
                clean_loss = self.criterion(self.model(inputs), targets)
                loss = self.clean_coeff * clean_loss + \
                    (1 - self.clean_coeff) * adv_loss
            elif self.clean_coeff == 0:
                outputs = self.model(adv_inputs)
                loss = self.criterion(outputs, targets)
            elif self.clean_coeff == 1:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

            #if self.save_loss:
            #    self.loss[idx] = loss.detach().cpu()
            loss = loss.mean()
            losses += loss.item()

            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            #if self.save_correct:
            #    self.correct[idx] = predicted.eq(targets).float().cpu()

            self.optimizer.zero_grad()
            if self.use_amp:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            self.optimizer.step() 
            self.scheduler.step()

            total += inputs.size(0)

            """
            if self.save_eps:
                self.eps[idx] = (adv_inputs-inputs).cpu()

            if self.save_minus_delta:
                self.model.eval()
                outputs = self.model(2*inputs-adv_inputs)
                _, preds = outputs.max(1)
                loss = self.criterion(outputs, targets)
                self.minus_delta_loss[idx, 1] = loss.detach().cpu()
                self.minus_delta_correct[idx, 1] = preds.eq(targets).float().cpu()
                self.model.train() 
            """

        """
        if self.save_minus_delta:
            print_message = 'Epoch [{:3d}] | Adv Loss: {:.4f}, Adv Acc: {:.2%}, Minus delta loss: {:.4f} / {:.4f}, Minus delta acc: {:.2%} / {:.2%}'.format(
                epoch, losses/len(batch_iter), correct/len(self.trainset), 
                self.minus_delta_loss[:,0].mean(), self.minus_delta_loss[:,1].mean(),
                self.minus_delta_correct[:,0].mean(), self.minus_delta_correct[:,1].mean())
        else:
            print_message = 'Epoch [{:3d}] | Adv Loss: {:.4f}, Adv Acc: {:.2%}'.format(epoch, losses/len(batch_iter), correct/len(self.trainset))
        """
        print_message = 'Epoch [{:3d}] | Adv Loss: {:.4f}, Adv Acc: {:.2%}'.format(epoch, losses/len(batch_iter), correct/total)
        tqdm.write(print_message)

        self.writer.add_scalar('train/adv_loss', losses/len(batch_iter), epoch)
        self.writer.add_scalar('train/adv_acc', correct/total, epoch)
        self.writer.add_scalar('lr', current_lr, epoch)
        self.writer.add_scalar('steps', self.attack_cfg['steps'], epoch)
        self.writer.add_scalar('eps', self.attack_cfg['eps']*255, epoch)

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
        
        """
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

            print_message += '\tAdv Loss {loss:.4f}\tAdv Acc {acc:.2%}\n'.format(
                loss=adv_loss/len(self.testloader), acc=adv_correct/total
            )
        """

        if self.test_robust:
            total = 0
            adv_loss = 0; adv_correct = 0
            batch_num = min(len(self.testloader), 10)

            for idx, (inputs, targets) in enumerate(self.testloader):
                if idx == batch_num:
                    break

                inputs, targets = inputs.cuda(), targets.cuda()
                adv_inputs = Linf_PGD(self.model, inputs, targets, eps=8./255., alpha=2./255., steps=10)
                
                with torch.no_grad():
                    outputs = self.model(adv_inputs)
                    adv_loss += nn.CrossEntropyLoss()(outputs, targets).item()
                
                _, predicted = outputs.max(1)
                adv_correct += predicted.eq(targets).sum().item()

                total += inputs.size(0)
            
            self.writer.add_scalar('test/adv_loss', adv_loss/batch_num, epoch)
            self.writer.add_scalar('test/adv_acc', 100*adv_correct/total, epoch)

            print_message += '\tAdv Loss {loss:.4f}\tAdv Acc {acc:.2%}\n'.format(
                loss=adv_loss/len(self.testloader), acc=adv_correct/total
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

        if self.save_eps:
            self._save('eps', epoch)

        if self.save_loss:
            self._save('loss', epoch)

        if self.save_fosc:
            self._save('fosc', epoch)
        
        if self.save_minus_delta:
            self._save('minus_delta_loss', epoch)
            self._save('minus_delta_correct', epoch)
        
        if self.save_correct:
            self._save('correct', epoch)


def get_args():
    parser = argparse.ArgumentParser(description='CIFAR10 Madry Adversarial Training', add_help=True)
    arguments.model_args(parser)
    arguments.data_args(parser)
    arguments.base_train_args(parser)
    arguments.madry_advt_args(parser)
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
    if args.arch == 'wrn':
        arch = '{:s}{:d}_{:d}'.format(args.arch, args.depth, args.width)
    else:
        arch = '{:s}{:d}'.format(args.arch, args.depth)
    save_root = os.path.join('checkpoints', args.dataset, arch, 'madry', 'seed_'+str(args.seed))
    subfolder = 'epochs_{:d}_batch_{:d}_lr_{:s}'.format(args.epochs, args.batch_size, args.lr_sch)
    if args.lr_sch == 'cyclic':
        subfolder += '_{:.1f}'.format(args.lr_max)
    subfolder += '_eps_%d_alpha_%d_steps_%d' % (args.eps, args.alpha, args.steps)
    if args.val:
        subfolder += '_val'
    if args.clean_coeff > 0:
        subfolder += '_clean_coeff_%.1f' % args.clean_coeff
    if args.increase_steps:
        subfolder += '_[%s]@[%s]' % (','.join(str(e) for e in args.more_steps), ','.join(str(e) for e in args.steps_intervals))
    if args.increase_eps:
        subfolder += '_increase_eps'
    if args.linear_eps:
        subfolder += '_linear_eps'
    if args.input_diversity:
        subfolder += '_id_prob_%.1f' % args.id_prob
    if args.cutout:
        subfolder += '_cutout_%d_%d' % (args.cutout_n_holes, args.cutout_length)
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
    np.random.seed(args.seed)

    # initialize model, optimizer, and scheduler
    model, optimizer, scheduler, trainloader, testloader = utils.setup(args, train=True)

    # train the model
    trainer = Madry(args, model, optimizer, scheduler, trainloader, testloader, writer, save_root)
    trainer.run()


if __name__ == '__main__':
    main()








