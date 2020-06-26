import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from apex import amp
except ModuleNotFoundError:
    pass


def gradient_wrt_data(model, inputs, targets, criterion):
    inputs.requires_grad = True
    outputs = model(inputs)

    loss = criterion(outputs, targets)
    model.zero_grad()
    loss.backward()

    data_grad = inputs.grad.data
    return data_grad.clone().detach()


def gradient_wrt_data_apex(model, inputs, targets, criterion, optimizer):
    inputs.requires_grad = True
    outputs = model(inputs)

    loss = criterion(outputs, targets)
    model.zero_grad()
    with amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward()
    #loss.backward()

    data_grad = inputs.grad.data
    return data_grad.clone().detach()


def Linf_PGD(model, dat, lbl, eps, alpha, steps, is_targeted=False, rand_start=True, criterion=nn.CrossEntropyLoss()):
    x_nat = dat.clone().detach()
    x_adv = None
    if rand_start:
        x_adv = dat.clone().detach() + torch.FloatTensor(dat.shape).uniform_(-eps, eps).cuda()
    else:
        x_adv = dat.clone().detach()
    x_adv = torch.clamp(x_adv, 0., 1.) # respect image bounds
    # Iteratively Perturb data
    for i in range(steps):
        # Calculate gradient w.r.t. data
        grad = gradient_wrt_data(model, x_adv, lbl, criterion)
        with torch.no_grad():
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


def Linf_PGD_for_CAT_my(model, dat, lbl, eps, alpha, steps, is_targeted=False, rand_start=True, criterion=nn.CrossEntropyLoss()):
    x_nat = dat.clone().detach()
    x_adv = None
    
    if rand_start:
        #x_adv = dat.clone().detach() + torch.FloatTensor(dat.shape).uniform_(-eps, eps).cuda()
        start_noise = torch.zeros_like(dat)
        for ii in range(dat.shape[0]):
            start_noise[ii].uniform_(-eps[ii], eps[ii])
        x_adv = dat.clone().detach() + start_noise.float()
    else:
        x_adv = dat.clone().detach()
    x_adv = torch.clamp(x_adv, 0., 1.) # respect image bounds
    
    # Iteratively Perturb data
    for i in range(steps):
        # Calculate gradient w.r.t. data
        grad = gradient_wrt_data(model, x_adv, lbl, criterion)
        with torch.no_grad():
            # Get the sign of the gradient
            sign_data_grad = grad.sign()
            if is_targeted:
                x_adv = x_adv - alpha.view(-1,1,1,1) * sign_data_grad # perturb the data to MINIMIZE loss on tgt class
            else:
                x_adv = x_adv + alpha.view(-1,1,1,1) * sign_data_grad # perturb the data to MAXIMIZE loss on gt class
            # Clip the perturbations w.r.t. the original data so we still satisfy l_infinity
            #x_adv = torch.clamp(x_adv, x_nat-eps, x_nat+eps) # Tensor min/max not supported yet
            x_adv = torch.max(torch.min(x_adv, x_nat+eps.view(-1,1,1,1)), x_nat-eps.view(-1,1,1,1))
            # Make sure we are still in bounds
            x_adv = torch.clamp(x_adv, 0., 1.)
    return x_adv.clone().detach()


def Linf_PGD_for_CAT_minhao(model, dat, lbl, eps, alpha, steps, is_targeted=False, rand_start=False, return_mask=False, criterion=nn.CrossEntropyLoss()):
    x_nat = dat.clone().detach()
    x_adv = None

    if rand_start:
        #x_adv = dat.clone().detach() + torch.FloatTensor(dat.shape).uniform_(-eps, eps).cuda()
        start_noise = torch.zeros_like(dat)
        for ii in range(dat.shape[0]):
            start_noise[ii].uniform_(-eps[ii], eps[ii])
        x_adv = dat.clone().detach() + start_noise.float()
    else:
        x_adv = dat.clone().detach()
    
    # Iteratively Perturb data
    for i in range(steps):
        # Calculate gradient w.r.t. data
        grad = gradient_wrt_data(model, x_adv, lbl, criterion)
        with torch.no_grad():
            # Get the sign of the gradient
            sign_data_grad = grad.sign()
            if is_targeted:
                x_adv = x_adv - alpha.view(-1,1,1,1) * sign_data_grad # perturb the data to MINIMIZE loss on tgt class
            else:
                x_adv = x_adv + alpha.view(-1,1,1,1) * sign_data_grad # perturb the data to MAXIMIZE loss on gt class
            # Clip the perturbations w.r.t. the original data so we still satisfy l_infinity
            #x_adv = torch.clamp(x_adv, x_nat-eps, x_nat+eps) # Tensor min/max not supported yet
            x_adv = torch.max(torch.min(x_adv, x_nat+eps.view(-1,1,1,1)), x_nat-eps.view(-1,1,1,1))
            # Make sure we are still in bounds
            x_adv = torch.clamp(x_adv, 0., 1.)
    
    if len(lbl.shape) == 2:
        mask = (torch.max(model(x_adv),dim=1)[1] == torch.max(lbl,dim=1)[1])
    else:
        mask = (torch.max(model(x_adv),dim=1)[1] == lbl)
    if mask is None: # does this really do anything?
        return x_nat
    x_nat.data[mask] = x_adv.data[mask]
    if return_mask:
        return x_nat, mask
    return x_nat


def Linf_PGD_for_CAT_minhao_apex(model, optimizer, dat, lbl, eps, alpha, steps, is_targeted=False, rand_start=False, return_mask=False, criterion=nn.CrossEntropyLoss()):
    x_nat = dat.clone().detach()
    x_adv = None

    if rand_start:
        #x_adv = dat.clone().detach() + torch.FloatTensor(dat.shape).uniform_(-eps, eps).cuda()
        start_noise = torch.zeros_like(dat)
        for ii in range(dat.shape[0]):
            start_noise[ii].uniform_(-eps[ii], eps[ii])
        x_adv = dat.clone().detach() + start_noise.float()
    else:
        x_adv = dat.clone().detach()
    
    # Iteratively Perturb data
    for i in range(steps):
        # Calculate gradient w.r.t. data
        grad = gradient_wrt_data_apex(model, x_adv, lbl, criterion, optimizer)
        with torch.no_grad():
            # Get the sign of the gradient
            sign_data_grad = grad.sign()
            if is_targeted:
                x_adv = x_adv - alpha.view(-1,1,1,1) * sign_data_grad # perturb the data to MINIMIZE loss on tgt class
            else:
                x_adv = x_adv + alpha.view(-1,1,1,1) * sign_data_grad # perturb the data to MAXIMIZE loss on gt class
            # Clip the perturbations w.r.t. the original data so we still satisfy l_infinity
            #x_adv = torch.clamp(x_adv, x_nat-eps, x_nat+eps) # Tensor min/max not supported yet
            x_adv = torch.max(torch.min(x_adv, x_nat+eps.view(-1,1,1,1)), x_nat-eps.view(-1,1,1,1))
            # Make sure we are still in bounds
            x_adv = torch.clamp(x_adv, 0., 1.)
    
    if len(lbl.shape) == 2:
        mask = (torch.max(model(x_adv),dim=1)[1] == torch.max(lbl,dim=1)[1])
    else:
        mask = (torch.max(model(x_adv),dim=1)[1] == lbl)
    if mask is None: # does this really do anything?
        return x_nat
    x_nat.data[mask] = x_adv.data[mask]
    if return_mask:
        return x_nat, mask
    return x_nat