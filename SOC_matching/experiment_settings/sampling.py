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

class Sampler(method.NeuralSDE):
    def __init__(
        self,
        device="cuda",
        dim=2,
        hdims=[256, 128, 64],
        hdims_M=[128, 128],
        u=None,
        lmbd=1.0,
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
        self.setting = setting
        if self.setting == "sampling_cox":
            fcsv = osp.join(pathlib.Path(__file__).parent.resolve(), "df_pines.csv")
            self.cox = Cox(fcsv, 40, use_whitened=False)

        elif self.setting == "sampling_funnel":
            self.dist_dominant = D.Normal(torch.tensor([0.0]).to(self.device), torch.tensor([1.0]).to(self.device))
            self.mean_other = torch.zeros(dim - 1).float().to(self.device)
            self.cov_eye = torch.eye(dim - 1).float().to(self.device).view(1, dim - 1, dim - 1)

        elif self.setting == "sampling_MG":
            nmode=3
            xlim=3.0
            scale=0.15
            mix = D.Categorical(torch.ones(nmode).to(self.device))
            angles = np.linspace(0, 2 * 3.14, nmode, endpoint=False)
            poses = xlim * np.stack([np.cos(angles), np.sin(angles)]).T
            poses = torch.from_numpy(poses).to(self.device)
            comp = D.Independent(
                D.Normal(poses, torch.ones(size=(nmode, 2)).to(self.device) * scale * xlim), 1
            )

            self.gmm = MixtureSameFamily(mix, comp)

    # Base Drift
    def b(self, t, x):
        return - x 

    # Gradient of Base Drift
    def nabla_b(self, t, x):
        identity = torch.eye(x.shape[-1]).to(x.device)
        if len(x.shape) == 2:
            return - torch.transpose(identity.unsqueeze(0).repeat(x.shape[0], 1, 1), 1, 2)
        elif len(x.shape) == 3:
            return - torch.transpose(
                identity.unsqueeze(0).unsqueeze(0).repeat(x.shape[0], x.shape[1], 1, 1),
                2,
                3,
            )

    # Final cost
    def g(self, x):
        """
        x: (B, dim)
        output: (B,)
        """
        if self.setting == "sampling_cox":
            return -self.cox.evaluate_log_density(x)

        elif self.setting == "sampling_funnel":

            def _dist_other(dominant_x):
                variance_other = torch.exp(dominant_x)
                cov_other = variance_other.view(-1, 1, 1) * self.cov_eye
                return D.multivariate_normal.MultivariateNormal(self.mean_other, cov_other)

            def funnel_log_pdf(x):
                dominant_x = x[:, 0]
                log_density_dominant = self.dist_dominant.log_prob(dominant_x)  # (B, )

                log_density_other = _dist_other(dominant_x).log_prob(x[:, 1:])  # (B, )
                return log_density_dominant + log_density_other
            
            return -funnel_log_pdf(x)

        elif self.setting == "sampling_MG":
            return -self.gmm.log_prob(x)

    # Gradient of Final cost
    def nabla_g(self, x):
        with torch.enable_grad():
            x = x.requires_grad_(True)
            output = torch.autograd.grad(self.g(x).sum(), x)[0]
            return output
        
    # State cost
    def f(self, t, x):
        return self.dim * torch.ones_like(x[...,0]).to(x.device)

    # Gradient of state cost
    def nabla_f(self, t, x):
        return torch.zeros_like(x).to(x.device)