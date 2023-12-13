# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory.
import torch

from SOC_matching import method


class OU_Linear(method.NeuralSDE):
    def __init__(
        self,
        device="cuda",
        dim=2,
        u=None,
        hdims=[256, 128, 64],
        hdims_M=[128, 128],
        lmbd=1.0,
        A=torch.eye(2),
        sigma=torch.eye(2),
        omega=torch.ones(2),
        gamma=3.0,
        scaling_factor_nabla_V=1.0,
        scaling_factor_M=1.0,
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
        )
        self.A = A
        self.omega = omega

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
        elif len(x.shape) == 3:
            return torch.transpose(
                self.A.unsqueeze(0)
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(x.shape[0], x.shape[1], x.shape[2], 1, 1),
                3,
                4,
            )

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

    # Final cost
    def g(self, x):
        return torch.einsum("j,...j->...", self.omega, x)

    # Gradient of final cost
    def nabla_g(self, x):
        if len(x.shape) == 2:
            return self.omega.unsqueeze(0).repeat(x.shape[0], 1)
        elif len(x.shape) == 3:
            return self.omega.unsqueeze(0).unsqueeze(0).repeat(
                x.shape[0], x.shape[1], 1
            )
        elif len(x.shape) == 3:
            return self.omega.unsqueeze(0).unsqueeze(0).unsqueeze(
                0
            ).repeat(x.shape[0], x.shape[1], x.shape[2], 1)
