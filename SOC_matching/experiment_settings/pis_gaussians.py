# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory.
import torch
import numpy as np
import torch.nn.functional as F

import os.path as osp
import pathlib
from .cox_utils import Cox

import torch.distributions as D
from torch.distributions.mixture_same_family import MixtureSameFamily

from SOC_matching import method

class PIS_Sampler(method.NeuralSDE):
    def __init__(
        self,
        device="cuda",
        dim=2,
        hdims=[256, 128, 64],
        hdims_M=[128, 128],
        u=None,
        lmbd=1.0,
        kappa=1.0,
        eta=1.0,
        sigma=torch.eye(2),
        gamma=3.0,
        scaling_factor_nabla_V=1.0,
        scaling_factor_M=1.0,
        T=1.0,
        u_warm_start=None,
        use_warm_start=False,
        setting=None,
        output_matrix=False,
    ):
        super().__init__(
            device=device,
            dim=dim,
            hdims=hdims,
            hdims_M=hdims_M,
            u=u,
            lmbd=lmbd,
            sigma=sigma,
            gamma=gamma,
            scaling_factor_nabla_V=scaling_factor_nabla_V,
            scaling_factor_M=scaling_factor_M,
            T=T,
            u_warm_start=u_warm_start,
            use_warm_start=use_warm_start,
            output_matrix=output_matrix,
        )
        self.kappa = kappa
        self.eta = eta

    # Base Drift
    def b(self, t, x):
        # return torch.zeros_like(x).to(self.device)
        return 0 * x

    # Gradient of base drift
    def nabla_b(self, t, x):
        x_shape = x.shape
        if len(x.shape) == 2:
            # return (
            #     torch.zeros_like(x)
            #     .to(self.device)
            #     .unsqueeze(2)
            #     .repeat(1, 1, x_shape[-1])
            # )
            return (
                (0 * x)
                .to(self.device)
                .unsqueeze(2)
                .repeat(1, 1, x_shape[-1])
            )
        elif len(x.shape) == 3:
            # return (
            #     torch.zeros_like(x)
            #     .to(self.device)
            #     .unsqueeze(3)
            #     .repeat(1, 1, 1, x_shape[-1])
            # )
            return (
                (0 * x)
                .to(self.device)
                .unsqueeze(3)
                .repeat(1, 1, 1, x_shape[-1])
            )

    # Final cost
    def g(self, x):
        """
        x: (B, dim)
        output: (B,)
        """
        log_mu_plus = - (x[:,0] - self.kappa) ** 2 / (2 * self.eta) - torch.sum(x[:,1:] ** 2, dim=1) / (2 * self.eta) + torch.log(torch.exp(- (2 * self.kappa * x[:,0]) / self.eta) + 1) - np.log(2) - (self.dim / 2) * np.log(2 * np.pi * self.eta)
        log_mu_minus = - (x[:,0] + self.kappa) ** 2 / (2 * self.eta) - torch.sum(x[:,1:] ** 2, dim=1) / (2 * self.eta) + torch.log(torch.exp((2 * self.kappa * x[:,0]) / self.eta) + 1) - np.log(2) - (self.dim / 2) * np.log(2 * np.pi * self.eta)
        log_mu = (x[:,0] >= 0.05) * log_mu_plus + (x[:,0] < 0.05) * log_mu_minus

        log_mu_0 = - torch.sum(x ** 2, dim=1) / 2 - (self.dim / 2) * np.log(2 * np.pi)
        return log_mu_0 - log_mu
        # log_mu_plus = - (x[:,0] - self.kappa) / self.eta - (2 * self.kappa / self.eta) / (torch.exp((2 * self.kappa * x[:,0]) / self.eta) + 1) 

    # Gradient of Final cost
    def nabla_g(self, x):
        with torch.enable_grad():
            x = x.requires_grad_(True)
            output = torch.autograd.grad(self.g(x).sum(), x)[0]
            return output
        
    # Running cost
    def f(self, t, x):
        if len(x.shape) == 2:
            return torch.zeros(x.shape[0]).to(x.device)
        elif len(x.shape) == 3:
            return torch.zeros(x.shape[0], x.shape[1]).to(x.device)
        elif len(x.shape) == 4:
            return torch.zeros(x.shape[0], x.shape[1], x.shape[2]).to(
                x.device
            )

    # Gradient of running cost
    def nabla_f(self, t, x):
        return torch.zeros_like(x).to(x.device)