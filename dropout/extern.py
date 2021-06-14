import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter


# adapted from https://colab.research.google.com/github/bayesgroup/deepbayes-2019/blob/master/seminars/day6/SparseVD-solution.ipynb
class LinearSVDO(nn.Module):

    def __init__(self, in_features, out_features, threshold):
        super(LinearSVDO, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.threshold = threshold

        self.W = Parameter(torch.Tensor(out_features, in_features))
        self.log_alpha = Parameter(torch.Tensor(out_features, in_features))
        self.bias = Parameter(torch.Tensor(1, out_features))
        
        self.reset_parameters()

    def reset_parameters(self):
        self.bias.data.zero_()
        self.W.data.normal_(0, 0.02)
        self.log_alpha.data.fill_(-5)        
        
    def forward(self, x):
        self.log_sigma = self.log_alpha / 2.0 + torch.log(1e-16 + torch.abs(self.W))
        self.log_sigma = torch.clamp(self.log_sigma, -10, 10) 
        
        if self.training:
            lrt_mean =  F.linear(x, self.W) + self.bias
            lrt_std = torch.sqrt(F.linear(x * x, torch.exp(self.log_sigma * 2.0)) + 1e-8)
            eps = lrt_std.data.new(lrt_std.size()).normal_()
            return lrt_mean + lrt_std * eps
    
        return F.linear(x, self.W * (self.log_alpha < self.threshold).float()) + self.bias
    
    def sparsity(self):

        return torch.sum(self.log_alpha > self.threshold).item(), torch.numel(self.log_alpha)
        
    def kl_reg(self):
        # Return KL here -- a scalar 
        k1, k2, k3 = torch.Tensor([0.63576]).cuda(), torch.Tensor([1.8732]).cuda(), torch.Tensor([1.48695]).cuda()
        kl = k1 * torch.sigmoid(k2 + k3 * self.log_alpha) - 0.5 * torch.log1p(torch.exp(-self.log_alpha))
        a = - torch.sum(kl)
        return a


class LinearSVDOAdditiveReparam(nn.Module):

    def __init__(self, in_features, out_features, threshold):
        super(LinearSVDOAdditiveReparam, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.threshold = threshold

        self.W = Parameter(torch.Tensor(out_features, in_features))
        self.log_sigma = Parameter(torch.Tensor(out_features, in_features))
        self.bias = Parameter(torch.Tensor(1, out_features))
        
        self.reset_parameters()

    def reset_parameters(self):
        self.bias.data.zero_()
        self.W.data.normal_(0, 0.02)
        self.log_sigma.data.fill_(-5)        
        
    def forward(self, x):
        self.log_alpha = self.log_sigma * 2.0 - 2.0 * torch.log(1e-16 + torch.abs(self.W))
        self.log_alpha = torch.clamp(self.log_alpha, -10, 10) 
        
        if self.training:
            lrt_mean =  F.linear(x, self.W) + self.bias
            lrt_std = torch.sqrt(F.linear(x * x, torch.exp(self.log_sigma * 2.0)) + 1e-8)
            eps = lrt_std.data.new(lrt_std.size()).normal_()
            return lrt_mean + lrt_std * eps
    
        return F.linear(x, self.W * (self.log_alpha < self.threshold).float()) + self.bias
    
    def sparsity(self):

        return torch.sum(self.log_alpha > self.threshold).item(), torch.numel(self.log_alpha)
        
    def kl_reg(self):
        # Return KL here -- a scalar 
        k1, k2, k3 = torch.Tensor([0.63576]).cuda(), torch.Tensor([1.8732]).cuda(), torch.Tensor([1.48695]).cuda()
        kl = k1 * torch.sigmoid(k2 + k3 * self.log_alpha) - 0.5 * torch.log1p(torch.exp(-self.log_alpha))
        a = - torch.sum(kl)
        return a


class LinearSVDONoReparam(nn.Module):

    def __init__(self, in_features, out_features, threshold):
        super(LinearSVDONoReparam, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.threshold = threshold

        self.W = Parameter(torch.Tensor(out_features, in_features))
        self.log_alpha = Parameter(torch.Tensor(out_features, in_features))
        self.bias = Parameter(torch.Tensor(1, out_features))
        
        self.reset_parameters()

    def reset_parameters(self):
        self.bias.data.zero_()
        self.W.data.normal_(0, 0.02)
        self.log_alpha.data.fill_(-5)        
        
    def forward(self, x):
        
        if self.training:
            mask = 1 + torch.randn_like(self.log_alpha) * torch.exp(self.log_alpha / 2)
            return F.linear(x, self.W * mask) + self.bias
    
        return F.linear(x, self.W * (self.log_alpha < self.threshold).float()) + self.bias
    
    def sparsity(self):

        return torch.sum(self.log_alpha > self.threshold).item(), torch.numel(self.log_alpha)
        
    def kl_reg(self):
        # Return KL here -- a scalar 
        k1, k2, k3 = torch.Tensor([0.63576]).cuda(), torch.Tensor([1.8732]).cuda(), torch.Tensor([1.48695]).cuda()
        kl = k1 * torch.sigmoid(k2 + k3 * self.log_alpha) - 0.5 * torch.log1p(torch.exp(-self.log_alpha))
        a = - torch.sum(kl)
        return a


class VarDropNet(nn.Module):

    def __init__(self, threshold, reparam='additive', dims=None, lamda=None, tol=None):
        super(VarDropNet, self).__init__()
        if reparam == 'additive':
            linear_svdo = LinearSVDOAdditiveReparam
        elif reparam == 'local':
            linear_svdo = LinearSVDO
        else:
            linear_svdo = LinearSVDONoReparam
        self.fc1 = linear_svdo(28*28, 300, threshold)
        self.fc2 = linear_svdo(300,  100, threshold)
        self.fc3 = linear_svdo(100,  10, threshold)
        self.threshold=threshold

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x
    
    def sparsity(self):

        return [fc.sparsity() for fc in [self.fc1, self.fc2, self.fc3]]


# Define New Loss Function -- SGVLB 
class SGVLB(nn.Module):
    def __init__(self):
        super(SGVLB, self).__init__()
    
    # modified to set via callback
    def set_params(self, net, train_size):
        self.train_size = train_size
        self.net = net

    def forward(self, input, target, kl_weight=1.0):
        assert not target.requires_grad
        kl = 0.0
        for module in self.net.children():
            if hasattr(module, 'kl_reg'):
                kl = kl + module.kl_reg()
        return F.cross_entropy(input, target) * self.train_size + kl_weight * kl