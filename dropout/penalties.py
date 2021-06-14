from abc import ABC, abstractmethod

import numpy as np
from numpy.lib.function_base import interp
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize_scalar, brentq
from scipy.special import dawsn

# for solving for variational duals
F_IN_MAX = 1e6
OMEGA_IN_MAX = 1e6

# for inverting hard concrete distribution for L0 regularization
LOG_ALPHA_MIN = -4
LOG_ALPHA_MAX = 4


class Penalty(object):

    def __call__(self, x):

        return self.omega(x)


class Penalty1D(Penalty):

    def omega_upper_bound(self, w, eta):

        return w ** 2 / eta + self.f(eta)
    
    def f_lower_bound(self, eta, w):

        return 2 * self.omega(w) - w ** 2 / eta
    
    def minimize_eta(self, w, bounds=None):

        if bounds is None:
            bounds = self.eta_bounds()
        res = minimize_scalar(lambda eta: self.omega_upper_bound(w, eta), bounds=bounds, method='bounded')
        return res
    
    def eta_bounds(self):

        return (0, F_IN_MAX)
    
    def eta_hat(self, w):

        res = self.minimize_eta(w)
        return res.x

    def maximize_w(self, eta, bounds=None):

        if bounds is None:
            bounds = self.w_bounds()
        res = minimize_scalar(lambda w: -self.f_lower_bound(eta, w), bounds=bounds, method='bounded')
        return res
        
    def w_bounds(self):
        return (0, OMEGA_IN_MAX)
    
    def w_hat(self, eta):

        res = self.maximize_w(eta)
        return res.x


class FPenalty1D(Penalty1D):

    def __init__(self, f, f_bounds=(0, F_IN_MAX)):

        self.f = f
        self._f_bounds = f_bounds
    
    def f_bounds(self):

        return self._f_bounds
    
    def omega(self, w):

        return self.minimize_eta(w).fun / 2
        

class OmegaPenalty1D(Penalty1D):

    def __init__(self, omega):

        self.omega = omega

    def f(self, eta):

        return -self.maximize_w(eta).fun


class LpPenalty1D(Penalty1D):

    def __init__(self, p):

        self.p = p
        self.q = p / (2 - p)
    
    def omega(self, w):

        return 1 / self.p * w ** self.p
    
    def f(self, eta):

        return 1 / self.q * eta ** self.q
    
    def eta_hat(self, w):

        return w ** (2 - self.p)


class LogSumPenalty1D(Penalty1D):

    def __init__(self, epsilon):

        self.epsilon = epsilon
    
    def omega(self, w):

        return np.log(w + self.epsilon)
    
    def f(self, eta):

        sqrt = np.sqrt(self.epsilon ** 2 + 4 * eta)
        return 2 * np.log((sqrt + self.epsilon) / 2) - (sqrt - self.epsilon) ** 2 / (4 * eta)
    
    def eta_hat(self, w):

        return w * (w + self.epsilon)


class Log2SumPenalty1D(Penalty1D):

    def __init__(self, epsilon):

        self.epsilon = epsilon
    
    def omega(self, w):

        return np.log(w ** 2 + self.epsilon)
    
    def f(self, eta):

        return 2 * np.log(2 * eta) - self.epsilon / eta - 2
    
    def eta_hat(self, w):

        return (w ** 2 + self.epsilon) / 2


class SCADPenalty1D(Penalty1D):

    def __init__(self, a, lamda):

        self.a = a
        self.lamda = lamda
    
    def omega(self, w):

        if w < self.lamda:

            return w

        elif w <= self.a * self.lamda:

            return (2 * self.a * self.lamda * w - w ** 2 - self.lamda ** 2) / (2 * (self.a - 1) * self.lamda)

        else:

            return (self.a + 1) * self.lamda / 2
    
    def f(self, eta):

        if eta <= self.lamda:

            return eta

        else:

            return self.lamda * ((self.a + 1) * eta - self.lamda) / ((self.a - 1) * self.lamda + eta)
    
    def eta_hat(self, w):

        if w < self.lamda:

            return w
        
        elif w <= self.a * self.lamda:

            return (self.a - 1) * self.lamda * w / (self.a * self.lamda - w)
        
        else:

            return F_IN_MAX


class MCPPenalty1D(Penalty1D):

    def __init__(self, a, lamda):

        self.alamda = a * lamda
    
    def omega(self, w):

        if w <= self.alamda:

            return w - w ** 2 / (2 * self.alamda)
        
        else:

            return self.alamda / 2
    
    def f(self, eta):

        return self.alamda * eta / (eta + self.alamda)
    
    def eta_hat(self, w):

        if w < self.alamda:

            return self.alamda * w / (self.alamda - w)
        
        else:

            return F_IN_MAX
            

class StandoutPenalty1D(OmegaPenalty1D):

    def __init__(self, lamda=1, w2=1):

        self.lamda = lamda
        self.w2 = w2

    def omega(self, w):

        return self.w2 ** 2 / self.lamda * (1 - (w + 1) * np.exp(-w))


class VarDropPenalty1D(FPenalty1D):

    k1 = 0.63576
    k2 = 1.87320
    k3 = 1.48695

    def __init__(self, lamda):

        self.lamda = lamda
    
    def f(self, eta):

        return 2 / self.lamda * (self.k1 * sigmoid(self.k3 * np.log(eta) - self.k2) + 0.5 * np.log(1 + eta))


class VarDropPenaltyExact1D(FPenalty1D):

    def __init__(self, n_etas=1000):

        self.n_etas = n_etas
        self.etas = np.logspace(-3, np.log10(F_IN_MAX), n_etas)
        dfs = dawsn(np.sqrt(self.etas / 2)) / np.sqrt(self.etas / 2)
        self.fs = cumulative_trapezoid(dfs, self.etas, initial=0)
    
    def f(self, eta):

        return interp(eta, self.etas, self.fs)


class L0LouizosPenalty1D(Penalty1D):

    def __init__(self, lamda=1, beta=2/3, gamma=-0.1, zeta=1.1, n_alphas=500, n_us=10_000):

        self.lamda = lamda
        self.beta = beta
        self.gamma = gamma
        self.zeta = zeta

        self._as = np.logspace(LOG_ALPHA_MIN, LOG_ALPHA_MAX, n_alphas)
        us = np.random.rand(n_us)
        # make sure we have no zeros
        us = np.maximum(us, np.finfo(us.dtype).eps)

        zs = sigmoid((np.log(us[None, :]) - np.log(1 - us[None, :]) + np.log(self._as[:, None])) / self.beta)
        ss = np.clip(zs * (self.zeta - self.gamma) + self.gamma, 0, 1)

        self.Es = np.mean(ss, axis=1)
        self.E2s = np.mean(ss ** 2, axis=1)

        self.etas = 1 / (lamda * (self.E2s / self.Es ** 2 - 1))

    def f(self, eta):

        return self.complexity_loss(self.eta_to_a(eta))
    
    def omega(self, w):

        # make sure the optimizer doesn't run off to infinity for small w, since f is so flat
        return min(self.minimize_eta(w, bounds=(0, 1)).fun, self.minimize_eta(w).fun) / 2

    def complexity_loss(self, a):

        return sigmoid(np.log(a) - self.beta * np.log(-self.gamma / self.zeta))
    
    def eta_to_a(self, eta):

        return np.interp(eta, self.etas, self._as)
    

class SeparablePenalty(Penalty):

    def __init__(self, penalty_1d):

        self.penalty_1d = penalty_1d
    
    def omega(self, w):

        return sum(self.penalty_1d.omega(x) for x in w)
    
    def f(self, eta):

        return sum(self.penalty_1d.f(x) for x in eta)
    
    def eta_hat(self, w):

        return np.asarray([self.penalty_1d.eta_hat(x) for x in w])


class NonSeparablePenaltyWrapper1D(Penalty):

    def __init__(self, penalty, dir, origin):

        self.penalty = penalty
        self.dir = dir
        self.origin = origin
    
    def omega(self, w):

        return self.penalty.omega(w * self.dir + self.origin)
    
    def f(self, eta):

        return self.penalty.f(eta * self.dir + self.origin)


class ARDPenalty(Penalty):

    def __init__(self, tau=1e-3, X=None, n=100, p=100):

        self.tau = tau

        if X is not None:
            self.X = X.copy()
        else:
            self.X = np.random.randn(n, p) / np.sqrt(n)
        self.n, self.p = self.X.shape
    
    def f(self, eta):

        _, logdet = np.linalg.slogdet(self.tau * np.eye(self.n) + self.X @ (eta[:, None] * self.X.T))
        return logdet
    
    def omega(self, w):

        eta = self.eta_hat(w)
        return ((w ** 2) @ (1 / eta) + self.f(eta)) / 2

    def m(self, eta):

        def root_fun(c):
            return 1 / c - (self.tau + 1 / self.n * np.sum(eta / (1 + eta * c)))
        
        return brentq(root_fun, self.tau, 1 / self.tau)
    
    def eta_hat(self, w):

        def eta(c):
            return (w ** 2 + np.sqrt(w ** 4 + 4 / c * w ** 2)) / 2

        def root_fun(c):
            return c - self.m(eta(c))
        
        c = brentq(root_fun, self.tau, 1 / self.tau)
        return eta(c)


def sigmoid(x):

    return 1 / (1 + np.exp(-x))