# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory.
import torch

from SOC_matching import method


class OU_Quadratic(method.NeuralSDE):
    def __init__(
        self,
        device="cuda",
        dim=2,
        hdims=[256, 128, 64],
        hdims_M=[128, 128],
        u=None,
        lmbd=1.0,
        A=torch.eye(2),
        P=torch.eye(2),
        Q=torch.eye(2),
        sigma=torch.eye(2),
        gamma=3.0,
        scaling_factor_nabla_V=1.0,
        scaling_factor_M=1.0,
        T=1.0,
        u_warm_start=None,
        use_warm_start=False,
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
        )
        self.A = A
        self.P = P
        self.Q = Q

    # Base Drift
    def b(self, t, x):
        return torch.einsum("ij,...j->...i", self.A, x)

    # Gradient of base drift
    def nabla_b(self, t, x):
        if len(x.shape) == 2:
            return torch.transpose(self.A.unsqueeze(0).repeat(x.shape[0], 1, 1), 1, 2)
        elif len(x.shape) == 3:
            return torch.transpose(
                self.A.unsqueeze(0).unsqueeze(0).repeat(x.shape[0], x.shape[1], 1, 1),
                2,
                3,
            )

    # Running cost
    def f(self, t, x):
        return torch.sum(
            x * torch.einsum("ij,...j->...i", self.P, x), -1
        )

    # Gradient of running cost
    def nabla_f(self, t, x):
        return 2 * torch.einsum("ij,...j->...i", self.P, x)

    # Final cost
    def g(self, x):
        return torch.sum(
            x * torch.einsum("ij,...j->...i", self.Q, x), -1
        )

    # Gradient of final cost
    def nabla_g(self, x):
        return 2 * torch.einsum("ij,...j->...i", self.Q, x)
