# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory.
import torch
import torch.nn.functional as F
from matplotlib.patches import Rectangle

from SOC_matching import method


class Multiagent8(method.NeuralSDE):
    def __init__(
        self,
        device="cuda",
        dim=2,
        hdims=[256, 128, 64],
        hdims_M=[128, 128],
        u=None,
        lmbd=1.0,
        g_center=torch.zeros(2),
        g_coeff=1.0,
        f_coeff=1.0,
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
        self.g_center = g_center
        self.g_coeff = g_coeff
        self.f_coeff = f_coeff

    # Base Drift
    def b(self, t, x):
        return torch.zeros_like(x).to(self.device)

    # Gradient of base drift
    def nabla_b(self, t, x):
        x_shape = x.shape
        if len(x.shape) == 2:
            return (
                torch.zeros_like(x)
                .to(self.device)
                .unsqueeze(2)
                .repeat(1, 1, x_shape[-1])
            )
        elif len(x.shape) == 3:
            return (
                torch.zeros_like(x)
                .to(self.device)
                .unsqueeze(3)
                .repeat(1, 1, 1, x_shape[-1])
            )

    # Final cost
    def g(self, x):
        """
        x: (B, 2)
        output: (B,)
        """
        diff = x - self.g_center
        loss = (diff * diff).sum(-1)
        return (loss - 160) * self.g_coeff #- 120

    def nabla_g(self, x):
        with torch.enable_grad():
            x = x.requires_grad_(True)
            output = torch.autograd.grad(self.g(x).sum(), x)[0]
            return output

    # Running cost
    def f(self, t, x):
        """
        x: (T, B, 2) or (B, 2)
        output: (T, B) or (B)
        """
        del t

        if x.shape[-1] == 2:
            x1, x2 = x[..., 0], x[..., 1]
        elif len(x.shape) == 3:
            num_steps_p1, batch_size, dimension = x.shape
            x_reshape = x.reshape(num_steps_p1, batch_size, dimension // 2, 2)
            x1, x2 = x_reshape[..., 0], x_reshape[..., 1]
        elif len(x.shape) == 2:
            batch_size, dimension = x.shape
            x_reshape = x.reshape(batch_size, dimension // 2, 2)
            x1, x2 = x_reshape[..., 0], x_reshape[..., 1]

        def cost_fn(xy, width, height):
            xbound = xy[0], xy[0] + width
            ybound = xy[1], xy[1] + height

            a = -5 * (x1 - xbound[0]) * (x1 - xbound[1])
            b = -5 * (x2 - ybound[0]) * (x2 - ybound[1])

            cost = F.softplus(a, beta=20, threshold=1) * F.softplus(
                b, beta=20, threshold=1
            )
            if x.shape[-1] != 2:
                cost = torch.sum(cost, dim = -1)
            assert cost.shape == x.shape[:-1]
            return cost

        return self.f_coeff * (
            sum(
                cost_fn(xy, width, height)
                for xy, width, height in zip(*obstacle_multiagent_8())
            )
            - 680
        ) - 100

    def nabla_f(self, t, x):
        with torch.enable_grad():
            x = x.requires_grad_(True)
            output = torch.autograd.grad(self.f(t, x).sum(), x)[0]
            return output
        
def obstacle_multiagent_8():
            xys = [[-2, 3], [-2, -23]]
            widths = [4, 4]
            heights = [20, 20]
            return xys, widths, heights
        
def plot_obs(ax):
    xys, widths, heights = obstacle_multiagent_8()

    for xy, width, height in zip(xys, widths, heights):
        rec = Rectangle(xy=xy, width=width, height=height, zorder=0)
        ax.add_artist(rec)
        rec.set_clip_box(ax.bbox)
        rec.set_facecolor("darkgray")
        rec.set_edgecolor(None)
