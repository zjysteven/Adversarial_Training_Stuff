import torch
import torch.nn as nn
import torch.nn.functional as F
import random
try:
    from apex import amp
except ModuleNotFoundError:
    pass

"""
def gradient_wrt_data(model, inputs, targets, criterion, use_amp=False, optimizer=None):
    inputs.requires_grad = True
    outputs = model(inputs)

    loss = criterion(outputs, targets)
    model.zero_grad()

    if use_amp:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()

    data_grad = inputs.grad.data
    return data_grad.clone().detach()
"""

def gradient_wrt_data(model, inputs, targets, criterion, use_amp=False, optimizer=None, input_diversity=False, id_prob=0.5):
    inputs.requires_grad = True
    if input_diversity:
        outputs = model(input_diversity_fn(inputs, id_prob))
    else:
        outputs = model(inputs)

    loss = criterion(outputs, targets)
    model.zero_grad()

    if use_amp:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()

    data_grad = inputs.grad.data
    return data_grad.clone().detach()


def Linf_PGD(model, dat, lbl, eps, alpha, steps, 
             is_targeted=False, rand_start=True, criterion=nn.CrossEntropyLoss(), inner_max='madry', 
             return_mask=False, use_amp=False, optimizer=None, fosc=False, input_diversity=False, prob=0.5):
    assert type(eps) == type(alpha), 'eps and alpha type should match'
    assert isinstance(eps, float) or isinstance(eps, torch.Tensor), 'eps type is not valid'
    
    # set to eval mode, but also record the initial mode for future recover
    mode = model.training
    model.eval()

    x_nat = dat.clone().detach()
    x_adv = None
    
    # random start
    if rand_start:
        if isinstance(eps, float):
            x_adv = dat.clone().detach() + torch.FloatTensor(dat.shape).uniform_(-eps, eps).cuda()
        elif isinstance(eps, torch.Tensor):
            start_noise = torch.zeros_like(dat)
            for ii in range(dat.shape[0]):
                start_noise[ii].uniform_(-eps[ii], eps[ii])
            x_adv = dat.clone().detach() + start_noise.float()
    else:
        x_adv = dat.clone().detach()
    x_adv = torch.clamp(x_adv, 0., 1.) # respect image bounds

    if isinstance(eps, torch.Tensor):
        eps = eps.view(-1,1,1,1)
        alpha = alpha.view(-1,1,1,1)

    # iteratively perturb data
    for i in range(steps):
        # Calculate gradient w.r.t. data
        grad = gradient_wrt_data(model, x_adv, lbl, criterion, use_amp, optimizer, input_diversity, prob)
        with torch.no_grad():
            # Get the sign of the gradient
            sign_data_grad = grad.sign()
            if is_targeted:
                # perturb the data to MINIMIZE loss on tgt class
                x_adv = x_adv - alpha * sign_data_grad
            else:
                # perturb the data to MAXIMIZE loss on gt class
                x_adv = x_adv + alpha * sign_data_grad
            # Clip the perturbations w.r.t. the original data so we still satisfy l_infinity
            #x_adv = torch.clamp(x_adv, x_nat-eps, x_nat+eps) # Tensor min/max not supported yet
            x_adv = torch.max(torch.min(x_adv, x_nat+eps), x_nat-eps)
            # Make sure we are still in bounds
            x_adv = torch.clamp(x_adv, 0., 1.)
    
    if inner_max in ['madry', 'cat_paper']:
        if fosc:
            grad = gradient_wrt_data(model, x_adv, lbl, nn.CrossEntropyLoss(reduction='sum'))
            grad_flatten = grad.view(grad.shape[0], -1)
            grad_norm = torch.norm(grad_flatten, 1, dim=1)
            diff = (x_adv.clone().detach() - x_nat).view(x_nat.shape[0], -1)
            fosc_value = eps * grad_norm - (grad_flatten * diff).sum(dim=1)
            model.train(mode)
            return x_adv.clone().detach(), fosc_value
        else:
            model.train(mode)
            return x_adv.clone().detach()
    elif inner_max == 'cat_code':
        if len(lbl.shape) == 2:
            mask = (torch.max(model(x_adv),dim=1)[1] == torch.max(lbl,dim=1)[1])
        else:
            mask = (torch.max(model(x_adv),dim=1)[1] == lbl)
        if mask is None: # does this really do anything?
            return x_nat
        x_nat.data[mask] = x_adv.data[mask]
        model.train(mode)
        if return_mask:
            return x_nat, mask
        return x_nat


def input_diversity_fn(image, prob, low=28, high=32):
    if random.random() > prob:
        return image
    rnd = random.randint(low, high)
    rescaled = F.interpolate(image, size=[rnd, rnd], mode='bilinear')
    h_rem = high - rnd
    w_rem = high - rnd
    pad_top = random.randint(0, h_rem)
    pad_bottom = h_rem - pad_top
    pad_left = random.randint(0, w_rem)
    pad_right = w_rem - pad_left
    padded = F.pad(rescaled, [pad_top, pad_bottom, pad_left, pad_right], 'constant', 0)
    return padded

"""
def Linf_PGD_Diverse_Input(model, dat, lbl, eps, alpha, steps, 
                           is_targeted=False, rand_start=False, criterion=nn.CrossEntropyLoss(), 
                           prob=0.5, momentum=False, mu=1, use_amp=False, optimizer=None)
    x_nat = dat.clone().detach()
    x_adv = None
    if rand_start:
        x_adv = dat.clone().detach() + torch.FloatTensor(dat.shape).uniform_(-eps, eps).cuda()
    else:
        x_adv = dat.clone().detach()
    x_adv = torch.clamp(x_adv, 0., 1.) # respect image bounds
    g = torch.zeros_like(x_adv)
    # Iteratively Perturb data
    for i in range(steps):
        # Calculate gradient w.r.t. data
        grad = gradient_wrt_data(model, x_adv, lbl, criterion, use_amp, optimizer, input_diversity=True, id_prob=prob)
        with torch.no_grad():
            if momentum:
                # Compute sample wise L1 norm of gradient
                flat_grad = grad.view(grad.shape[0], -1)
                l1_grad = torch.norm(flat_grad, 1, dim=1)
                grad = grad / torch.clamp(l1_grad, min=1e-12).view(grad.shape[0],1,1,1)
                # Accumulate the gradient
                new_grad = mu * g + grad # calc new grad with momentum term
                g = new_grad
            else:
                new_grad = grad
            # Get the sign of the gradient
            sign_data_grad = new_grad.sign()
            # Get the sign of the gradient
            sign_data_grad = grad.sign()
            if is_targeted:
                x_adv = x_adv - alpha * sign_data_grad # perturb the data to MINIMIZE loss on tgt class
            else:
                x_adv = x_adv + alpha * sign_data_grad # perturb the data to MAXIMIZE loss on gt class
            # Clip the perturbations w.r.t. the original data so we still satisfy l_infinity
            #x_adv = torch.clamp(x_adv, x_nat-eps, x_nat+eps) # Tensor min/max not supported yet
            x_adv = torch.max(torch.min(x_adv, x_nat+eps), x_nat-eps)
            # Make sure we are still in bounds
            x_adv = torch.clamp(x_adv, 0., 1.)
    return x_adv.clone().detach()
"""

def calc_fosc(model, adv, nat, lbl):
    # https://github.com/YisenWang/dynamic_adv_training/blob/master/pgd_attack.py
    x_adv = adv.clone().detach()
    x_nat = nat.clone().detach()
    # set loss reduction to sum, otherwise the fosc value is too small
    grad = gradient_wrt_data(model, x_adv, lbl, nn.CrossEntropyLoss(reduction='sum'))
    grad_flatten = grad.view(grad.shape[0], -1)
    grad_norm = torch.norm(grad_flatten, 1, dim=1)
    diff = (x_adv - x_nat).view(x_nat.shape[0], -1)
    fosc_value = eps * grad_norm - (grad_flatten * diff).sum(dim=1)
    return fosc_value