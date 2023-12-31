# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory.
import torch
import torch.nn.functional as F
from matplotlib.patches import Rectangle

from SOC_matching import method


class MolecularDynamics(method.NeuralSDE):
    def __init__(
        self,
        device="cuda",
        dim=2,
        hdims=[256, 128, 64],
        hdims_M=[128, 128],
        u=None,
        lmbd=1.0,
        kappa=torch.ones(2),
        sigma=torch.eye(2),
        gamma=3.0,
        scaling_factor_nabla_V=1.0,
        scaling_factor_M=1.0,
        T=1.0,
        u_warm_start=None,
        use_warm_start=False,
        use_stopping_time=False,
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
            use_stopping_time=use_stopping_time,
        )
        self.kappa = kappa

    # Base Drift
    def b(self, t, x):
        if len(x.shape) == 2:
            return -2 * self.kappa.unsqueeze(0) * (x**2 - 1) * 2 * x
        elif len(x.shape) == 3:
            return -2 * self.kappa.unsqueeze(0).unsqueeze(0) * (x**2 - 1) * 2 * x

    # Gradient of base drift
    def nabla_b(self, t, x):
        if len(x.shape) == 2:
            return -torch.diag_embed(
                8 * self.kappa.unsqueeze(0) * x**2
                + 4 * self.kappa.unsqueeze(0) * (x**2 - 1)
            )
        elif len(x.shape) == 3:
            return -torch.diag_embed(
                8 * self.kappa.unsqueeze(0).unsqueeze(0) * x**2
                + 4 * self.kappa.unsqueeze(0).unsqueeze(0) * (x**2 - 1)
            )

    # Final cost
    def g(self, x):
        """
        x: (B, D)
        output: (B,)
        """
        return torch.zeros_like(x[...,0])

    def nabla_g(self, x):
        # with torch.enable_grad():
        #     x = x.requires_grad_(True)
        #     output = torch.autograd.grad(self.g(x).sum(), x)[0]
        #     return output
        return torch.zeros_like(x)

    # Running cost
    def f(self, t, x):
        """
        x: (T, B, D) or (B, D)
        output: (T, B) or (B)
        """
        return torch.ones_like(x[...,0])

    def nabla_f(self, t, x):
        # with torch.enable_grad():
        #     x = x.requires_grad_(True)
        #     output = torch.autograd.grad(self.f(t, x).sum(), x)[0]
        #     return output
        return torch.zeros_like(x)

    def stopping_condition(self, x): # 0 if x is stopped, 1 if x is not stopped
        if len(x.shape) == 2:
            return (x[:,0] < 0).to(torch.int)
        elif len(x.shape) == 3:
            return (x[:,:,0] < 0).to(torch.int)
        
    def Phi(self, x):
        if len(x.shape) == 2:
            return - x[:,0]
        elif len(x.shape) == 3:
            return - x[:,:,0]