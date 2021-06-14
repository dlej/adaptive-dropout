from abc import abstractmethod

import torch
from torch import Tensor
from torch.nn import functional as F, Linear, Module, ModuleList
from torch.nn.parameter import Parameter


class MaskedLinear(Module):

    def __init__(self, in_features, out_features, bias=True, tol=1e-2):

        super(MaskedLinear, self).__init__()

        self.linear = Linear(in_features, out_features, bias=bias)
        self.alpha = Parameter(Tensor(out_features, in_features), requires_grad=False)
        self.tol = tol

        self.reset_parameters()
    
    def reset_parameters(self):

        self.alpha.fill_(1)

    def forward(self, input):

        mask = self.sample_mask()

        return F.linear(input, self.linear.weight * mask, self.linear.bias)
    
    def sample_mask(self):

        if self.training:
            mask = self.sample_training_mask()
        else:
            mask = torch.ones_like(self.linear.weight)
            mask[self.alpha < self.tol]

        return mask
    
    @abstractmethod
    def sample_training_mask(self):
        pass

    @abstractmethod
    def update_alpha(self):
        pass

    def sparsity(self):

        return torch.sum(self.alpha < self.tol).item(), torch.numel(self.alpha)


class PassThroughMaskLinear(MaskedLinear):

    def __init__(self, in_features, out_features, bias=True, tol=1e-2):

        super(PassThroughMaskLinear, self).__init__(in_features, out_features, bias=bias, tol=tol)
    
    def sample_training_mask(self):

        mask = torch.ones_like(self.linear.weight)
        return mask
    
    @torch.no_grad()
    def update_alpha(self, new_alpha):

        self.alpha.copy_(new_alpha)


class LogEtaLamdaLinear(MaskedLinear):

    def __init__(self, in_features, out_features, bias=True, tol=1e-2, eta_hat=None):

        super(LogEtaLamdaLinear, self).__init__(in_features, out_features, bias=bias, tol=tol)

        if eta_hat is None:
            self.log_eta_lamda = Parameter(torch.zeros_like(self.linear.weight) + 5)
        else:
            self.log_eta_lamda = Parameter(torch.zeros_like(self.linear.weight))
            with torch.no_grad():
                self.log_eta_lamda.copy_(torch.log(eta_hat(self.linear.weight)))
    
    def sample_training_mask(self):

        mask = torch.ones_like(self.linear.weight)
        return mask

    @torch.no_grad()
    def update_alpha(self):

        self.alpha.copy_(torch.sigmoid(self.log_eta_lamda))


class GaussianDropoutLinear(LogEtaLamdaLinear):

    def __init__(self, in_features, out_features, bias=True, tol=1e-2, eta_hat=None):

        super(GaussianDropoutLinear, self).__init__(in_features, out_features, bias=bias, tol=tol, eta_hat=eta_hat)
    
    def sample_training_mask(self):

        return 1 + torch.randn_like(self.log_eta_lamda) * torch.exp(-self.log_eta_lamda / 2)


class ScaledBernoulliDropoutLinear(MaskedLinear):

    def __init__(self, in_features, out_features, bias=True, tol=1e-2):

        super(ScaledBernoulliDropoutLinear, self).__init__(in_features, out_features, bias=bias, tol=tol)
    
    def sample_training_mask(self):

        mask = torch.zeros_like(self.alpha)
        which = torch.rand_like(self.alpha) < self.alpha
        mask[which] = 1 / self.alpha[which]

        return mask
    
    @torch.no_grad()
    def update_alpha(self, eta_hat, lamda):

        eta = eta_hat(self.linear.weight) 
        self.alpha.copy_(eta / (eta + lamda))


class MaskedLinearNet(Module):

    def __init__(self, dims, masked_linear_module, masked_linear_params={}, bias=True, tol=1e-2):

        super(MaskedLinearNet, self).__init__()

        self.linears = ModuleList()

        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            self.linears.append(masked_linear_module(in_dim, out_dim, bias=bias, tol=tol, **masked_linear_params))
        
    def forward(self, input):

        x = input.view(input.shape[0], -1)

        for linear in self.linears[:-1]:
            x = F.relu(linear(x))
        
        return self.linears[-1](x)

    @abstractmethod
    def update_alphas(self):
        pass
    
    def sparsity(self):

        return [linear.sparsity() for linear in self.linears]


class PenalizedNet(MaskedLinearNet):

    def __init__(self, dims, penalty_module, penalty_params={}, bias=True, tol=1e-2, lamda=1e-3):

        super(PenalizedNet, self).__init__(dims, PassThroughMaskLinear, bias=bias, tol=tol)

        self.penalty_ = penalty_module(**penalty_params)
        self.lamda = lamda
    
    @torch.no_grad()
    def update_alphas(self):

        for linear in self.linears:
            eta = self.penalty_.eta_hat(linear.linear.weight)
            linear.update_alpha(eta / (eta + self.lamda))

    def penalty(self):

        return self.lamda * sum(self.penalty_(linear.linear.weight) for linear in self.linears)


class GaussianDropoutNet(MaskedLinearNet):

    def __init__(self, dims, dual_penalty_module, dual_penalty_params={}, eta_hat_init=None, bias=True, tol=1e-2, lamda=1e-3):

        self.lamda = lamda

        if eta_hat_init is None:
            masked_linear_params = {}
        else:
            self.eta_hat_init = eta_hat_init
            masked_linear_params = dict(eta_hat=self.eta_hat)

        super(GaussianDropoutNet, self).__init__(dims, GaussianDropoutLinear, masked_linear_params=masked_linear_params, bias=bias, tol=tol)

        self.dual_penalty = dual_penalty_module(**dual_penalty_params)

    def eta_hat(self, w):

        return self.eta_hat_init(w) / self.lamda

    def update_alphas(self):

        for linear in self.linears:
            linear.update_alpha()
    
    def penalty(self):

        return self.lamda * 2 * sum(self.dual_penalty(2 * torch.exp(linear.log_eta_lamda)) for linear in self.linears)


class EtaTrickNet(MaskedLinearNet):

    def __init__(self, dims, dual_penalty_module, dual_penalty_params={}, eta_hat_init=None, eta_weight_decay=1e-3, bias=True, tol=1e-2, lamda=1e-3):

        self.lamda = lamda

        if eta_hat_init is None:
            masked_linear_params = {}
        else:
            self.eta_hat_init = eta_hat_init
            masked_linear_params = dict(eta_hat=self.eta_hat)

        super(EtaTrickNet, self).__init__(dims, LogEtaLamdaLinear, masked_linear_params=masked_linear_params, bias=bias, tol=tol)

        self.dual_penalty = dual_penalty_module(**dual_penalty_params)
        self.eta_weight_decay = eta_weight_decay

    def eta_hat(self, w):

        return self.eta_hat_init(w) / self.lamda

    def update_alphas(self):

        for linear in self.linears:
            linear.update_alpha()
    
    def penalty(self):

        l2s = []
        fs_eta = []
        wds = []
        
        for linear in self.linears:
            eta = self.lamda * torch.exp(linear.log_eta_lamda)
            l2s.append(torch.sum(linear.linear.weight ** 2 / eta))
            fs_eta.append(self.dual_penalty(eta))
            wds.append(self.eta_weight_decay * torch.sum(linear.log_eta_lamda))
        
        return self.lamda / 2 * (sum(l2s) + sum(fs_eta) + sum(wds))


class AdaptiveScaledBernoulliDropoutNet(MaskedLinearNet):

    def __init__(self, dims, eta_hat, bias=True, tol=1e-2, lamda=1e-3):

        super(AdaptiveScaledBernoulliDropoutNet, self).__init__(dims, ScaledBernoulliDropoutLinear, bias=bias, tol=tol)

        self.eta_hat = eta_hat
        self.lamda = lamda

    def update_alphas(self):

        for linear in self.linears:
            linear.update_alpha(self.eta_hat, self.lamda)


class AdaptiveTikhonovNet(MaskedLinearNet):

    def __init__(self, dims, eta_hat, bias=True, tol=1e-2, lamda=1e-3, alpha_min=1e-3):

        super(AdaptiveTikhonovNet, self).__init__(dims, PassThroughMaskLinear, bias=bias, tol=tol)

        self.eta_hat = eta_hat
        self.lamda = lamda
        self.alpha_min = alpha_min

    def penalty(self):

        penalties = []

        for linear in self.linears:
            alpha = torch.clamp(linear.alpha, min=self.alpha_min)
            eta_lamda = alpha / (1 - alpha)

            penalties.append(torch.sum(linear.linear.weight ** 2 / eta_lamda) / 2)
        
        return sum(penalties)
    
    def update_alphas(self):

        for linear in self.linears:
            eta = self.eta_hat(linear.linear.weight)
            linear.update_alpha(eta / (eta + self.lamda))


class AdaptiveProxNet(MaskedLinearNet):

    def __init__(self, dims, eta_hat, bias=True, tol=1e-2, lamda=1e-3):

        super(AdaptiveProxNet, self).__init__(dims, PassThroughMaskLinear, bias=bias, tol=tol)

        self.eta_hat = eta_hat
        self.lamda = lamda
        self.lr = 1

    @torch.no_grad()
    def prox(self):

        for linear in self.linears:
            linear.linear.weight.copy_(linear.alpha * linear.linear.weight)

    @torch.no_grad()
    def update_alphas(self):

        for linear in self.linears:
            eta = self.eta_hat(linear.linear.weight)
            linear.update_alpha(eta / (eta + self.lr * self.lamda))
    
    @torch.no_grad()
    def update_alphas_Adam(self, adam):

        self.lr = adam.param_groups[0]['lr']
        beta_2 = adam.param_groups[0]['betas'][1]

        for linear in self.linears:

            state = adam.state[linear.linear.weight]
            sqrt_v_hat = torch.sqrt(state['exp_avg_sq'] / (1 - beta_2 ** state['step']))
            eta = self.eta_hat(linear.linear.weight)
            linear.update_alpha(eta / (eta + self.lr / sqrt_v_hat * self.lamda))


class PenalizedCriterion(Module):

    def __init__(self, criterion_class, criterion_params={}):

        super(PenalizedCriterion, self).__init__()
        self.criterion = criterion_class(**criterion_params)
        self.penalty = None
    
    def forward(self, input, target):

        loss = self.criterion(input, target)

        if self.penalty is not None:
            loss += self.penalty()
        
        return loss
        
    def set_penalty_from_net(self, net):

        self.penalty = net.penalty


class LogSumPenalty(Module):

    def __init__(self, epsilon=1):

        super(LogSumPenalty, self).__init__()

        self.epsilon = epsilon
    
    def forward(self, input):

        return torch.sum(torch.log(torch.abs(input) + self.epsilon))
    
    def eta_hat(self, w):

        w = torch.abs(w)
        return w * (w + self.epsilon)


class LogSumDualPenalty(Module):

    def __init__(self, epsilon=1):

        super(LogSumDualPenalty, self).__init__()

        self.epsilon = epsilon
    
    def forward(self, input):

        eta = input
        sqrt = torch.sqrt(self.epsilon ** 2 + 4 * eta)

        return 2 * torch.sum(torch.log((sqrt + self.epsilon) / 2) - (sqrt - self.epsilon) ** 2 / (4 * eta))
