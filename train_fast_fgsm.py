import os, json, argparse, hashlib, logging, random
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
from tqdm import tqdm

import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import arguments
import utils
from attacks import Linf_PGD


class Fast_AdvT():
    def __init__(self, model, optimizer, scheduler,
                 trainloader, testloader,
                 writer, save_path=None, rob_testloader=None, **kwargs):
        self.model = model
        self.epochs = kwargs['epochs']
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.trainloader = trainloader
        self.testloader = testloader

        self.writer = writer
        self.save_path = save_path
        
        self.criterion = nn.CrossEntropyLoss()

        # PGD configs
        self.attack_cfg = {'eps': kwargs['eps']/255., 
                           'alpha': kwargs['alpha']/255.,
                           'steps': 1,
                           'is_targeted': False,
                           'rand_start': True
                          }
        self.rob_testloader = rob_testloader

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

        current_lr = self.scheduler.get_last_lr()[0]
        
        batch_iter = self.get_batch_iterator()
        for inputs, targets in batch_iter:
            inputs, targets = inputs.cuda(), targets.cuda()

            adv_inputs = Linf_PGD(self.model, inputs, targets, **self.attack_cfg)

            outputs = self.model(adv_inputs)
            loss = self.criterion(outputs, targets)
            losses += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()            

        print_message = 'Epoch [{:3d}] | Adv Loss: {:.4f}'.format(epoch, losses/len(batch_iter))
        tqdm.write(print_message)

        self.scheduler.step()
        self.writer.add_scalar('train/adv_loss', losses/len(batch_iter), epoch)
        self.writer.add_scalar('lr', current_lr, epoch)

    def test(self, epoch):
        self.model.eval()
        
        total = 0
        clean_loss = 0; clean_correct = 0
        
        with torch.no_grad():
            for inputs, targets in self.testloader:
                inputs, targets = inputs.cuda(), targets.cuda()

                outputs = self.model(inputs)
                clean_loss += self.criterion(outputs, targets).item()
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

                adv_inputs = Linf_PGD(self.model, inputs, targets,  eps=8./255., alpha=2./255., steps=10)
                
                with torch.no_grad():
                    outputs = self.model(adv_inputs)
                    adv_loss += self.criterion(outputs, targets).item()
                
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
        }, self.save_path+'_'+str(epoch)+'.pth')


def get_args():
    parser = argparse.ArgumentParser(description='CIFAR10 Adversarial Training of Ensemble', add_help=True)
    arguments.model_args(parser)
    arguments.data_args(parser)
    arguments.base_train_args(parser)
    arguments.fast_adv_train_args(parser)
    args = parser.parse_args()
    return args


def main():
    # get args
    args = get_args()

    # set up gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    assert torch.cuda.is_available()

    # set up writer, logger, and save directory for models
    filename = 'model'
    save_root = os.path.join('checkpoints', 'fast_advt', '{:s}{:d}-{:d}_seed_{:d}_alpha_{:d}_epochs_{:d}_{:s}'.format(args.arch, args.depth, args.width, args.seed, args.alpha, args.epochs, args.lr_sch))
    if args.lr_sch == 'cyclic':
        save_root += '_{:.1f}'.format(args.lr_max)
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    else:
        print('*********************************')
        print('* The checkpoint already exists *')
        print('*********************************')

    save_path = os.path.join(save_root, filename)
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

    # get data loaders
    trainloader, testloader = utils.get_loaders(args)
    if args.test_robust:
        subset_idx = random.sample(range(10000), 1000)
        rob_testloader = utils.get_testloader(args, batch_size=100, shuffle=False, subset_idx=subset_idx)
    else:
        rob_testloader = None

    # get optimizers and schedulers
    optimizer, scheduler = utils.get_optimizer_and_scheduler(args, model)

    # train the ensemble
    trainer = Fast_AdvT(model, optimizer, scheduler,
        trainloader, testloader, writer, save_path, rob_testloader, **vars(args))
    trainer.run()


if __name__ == '__main__':
    main()