# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory.
import torch

from . import interp1d

DIRECTIONS = ["fwd", "bwd"]

class EndPointSpline(torch.nn.Module):
    def __init__(self, t, xt, spline_type="linear", fix_init=False, init_knots=1):
        """
        t: (T,)
        xt: (B, T, D)
        """
        super(EndPointSpline, self).__init__()
        B, T, D = xt.shape
        assert t.shape == (T,) and T > 2, "Need at least 3 points"
        assert t.device == xt.device

        t = t.detach().clone()
        xt = xt.permute(1, 0, 2).detach().clone()
        
        self.B = B
        self.T = T
        self.D = D
        self.spline_type = spline_type
        self.fix_init = fix_init
        self.init_knots = init_knots

        self.register_buffer("t", t)
        self.register_buffer("tt", t.reshape(-1, 1).expand(-1, B))

        if fix_init:
            self.register_buffer("x0", xt[:init_knots].reshape(init_knots, B, D))
            self.register_parameter("x1", torch.nn.Parameter(xt[-1].reshape(1, B, D)))
            self.register_parameter("knots", torch.nn.Parameter(xt[init_knots:-1]))
        else:
            self.register_buffer("x0", xt[0].reshape(1, B, D))
            self.register_parameter("x1", torch.nn.Parameter(xt[-1].reshape(1, B, D)))
            self.register_parameter("knots", torch.nn.Parameter(xt[1:-1]))

    @property
    def device(self):
        return self.parameters().__next__().device

    @property
    def xt(self):  # (B, T, D)
        return torch.cat([self.x0, self.knots, self.x1], dim=0).permute(1, 0, 2)

    def forward(self, query_t):
        """
        query_t: (S,) --> yt: (B, S, D)
        """

        (S,) = query_t.shape
        query_t = query_t.reshape(-1, 1).expand(-1, self.B)
        assert query_t.shape == (S, self.B)

        mask = None
        xt = torch.cat([self.x0, self.knots, self.x1], dim=0)  # (T, B, D)
        if self.spline_type == "linear":
            yt = interp1d.linear_interp1d(self.tt, xt, mask, query_t)
        elif self.spline_type == "cubic":
            yt = interp1d.cubic_interp1d(self.tt, xt, mask, query_t)
        yt = yt.permute(1, 0, 2)
        assert yt.shape == (self.B, S, self.D), yt.shape
        return yt


class MeanSpline(EndPointSpline):
    def __init__(self, t, xt):
        super(MeanSpline, self).__init__(t, xt, spline_type="linear")


class GammaSpline(torch.nn.Module):
    def __init__(self, t, xt, sigma, fix_init=False, init_knots=1):
        """
        t: (T,)
        xt: (B, T, 1)
        """
        super(GammaSpline, self).__init__()
        B, T, D = xt.shape
        assert t.shape == (T,) and D == 1

        self.T = T
        self.B = B
        self.sigma = sigma

        self.spline = EndPointSpline(t, xt, fix_init=fix_init, init_knots=init_knots)

        self.softplus = torch.nn.Softplus()
        self.sigmoid = torch.nn.Sigmoid()
        self.scale = 1.0 / self.softplus(torch.tensor([1.0]))

    @property
    def t(self):
        return self.spline.t

    @property
    def xt(self):
        return self.spline.xt

    @property
    def device(self):
        return self.spline.device

    def forward(self, t):
        base_gamma = brownian_motion_std(t, self.sigma) 

        xt = self.spline(t).squeeze(-1)

        gamma = self.scale.to(xt.device) * base_gamma * self.softplus(xt)
        return gamma


class BaseGammaSpline(torch.nn.Module):
    def __init__(self, t, xt, sigma):
        """
        t: (T,)
        xt: (B, T, 1)
        """
        super(BaseGammaSpline, self).__init__()
        B, T, D = xt.shape
        assert t.shape == (T,) and D == 1

        self.T = T
        self.B = B
        self.sigma = sigma

        self.spline = EndPointSpline(t, xt)

        self.softplus = torch.nn.Softplus()
        self.sigmoid = torch.nn.Sigmoid()
        self.scale = 1.0

    @property
    def t(self):
        return self.spline.t

    @property
    def xt(self):
        return self.spline.xt

    @property
    def device(self):
        return self.spline.device

    def forward(self, t):
        base_gamma = brownian_bridge_std(t, self.sigma)
        gamma = self.scale * base_gamma
        return gamma.reshape(1, -1).expand(self.B, -1)


################################################################################################


def brownian_bridge_std(t, sigma):
    return sigma * (t * (1 - t)).sqrt()


def brownian_motion_std(t, sigma):
    return sigma * (t).sqrt()


class EndPointGaussianPath(torch.nn.Module):
    def __init__(self, mean, sigma, gamma, basedrift=None):
        super(EndPointGaussianPath, self).__init__()
        print(f"mean.B: {mean.B}, gamma.B: {gamma.B}")
        assert mean.B == gamma.B

        self.B = mean.B
        self.T = mean.T
        self.S = gamma.T
        self.D = mean.D

        self.mean = mean  # t: (T,) --> (B, T, D)
        self.sigma = sigma
        self.gamma = gamma  # t: (T,) --> (B, T)
        self.basedrift = basedrift  # xt: (*, T, D), t: (T,) --> (*, T, D)

    @property
    def device(self):
        return self.parameters().__next__().device

    def sample_xt(self, t, N):
        """marginal
        t: (T,) --> xt: (B, N, T, D)
        """

        mean_t = self.mean(t)  # (B, T, D)
        B, T, D = mean_t.shape

        assert t.shape == (T,)
        std_t = self.gamma(t).reshape(B, 1, T, 1)  # (B, 1, T, 1)

        noise = torch.randn(B, N, T, D, device=t.device)  # (B, N, T, D)

        xt = mean_t.unsqueeze(1) + std_t * noise
        assert xt.shape == noise.shape
        return xt

    def ft(self, t, xt, direction):
        """
        t: (T,)
        xt: (B, N, T, D)
        ===
        ft: (B, N, T, D)
        """
        B, N, T, D = xt.shape
        assert t.shape == (T,)

        if self.basedrift == None:
            return torch.zeros_like(xt)

        sign = 1.0 if direction == "fwd" else -1

        ft = self.basedrift(
            xt.reshape(B * N, T, D),
            t,
        ).reshape(B, N, T, D)
        return sign * ft

    # @profile
    def ut(self, t, xt, direction, create_graph_jvp=None, verbose=False):
        """
        t: (T,)
        xt: (B, N, T, D)
        ===
        ut: (B, N, T, D)
        """
        assert (t > 0).all() and (t < 1).all()

        B, N, T, D = xt.shape
        if verbose:
            print(f"xt.shape: {xt.shape}")
        assert t.shape == (T,)

        create_graph = create_graph_jvp or self.training

        mean, dmean = torch.autograd.functional.jvp(
            self.mean, t, torch.ones_like(t), create_graph=create_graph
        )
        if verbose:
            print(f"mean.shape: {mean.shape}, dmean.shape: {dmean.shape}")
        assert mean.shape == dmean.shape == (B, T, D)

        dmean = dmean.reshape(B, 1, T, D)
        mean = mean.reshape(B, 1, T, D)

        std, dstd = torch.autograd.functional.jvp(
            self.gamma, t, torch.ones_like(t).to(t.device), create_graph=create_graph
        )
        assert std.shape == dstd.shape == (B, T)

        if direction == "fwd":
            R_inverse = torch.matmul(self.sigma, torch.transpose(self.sigma, 0, 1))
            R_inverse_reshape = R_inverse.unsqueeze(0).unsqueeze(0).repeat(B, T, 1, 1)
            dstd_reshape = torch.diag_embed(dstd.unsqueeze(2).repeat(1, 1, D))
            inverse_std_reshape = torch.diag_embed(
                (1 / std).unsqueeze(2).repeat(1, 1, D)
            )
            a = torch.einsum(
                "...ij,...jk->...ik",
                (
                    dstd_reshape
                    - torch.einsum(
                        "...ij,...jk->...ik",
                        R_inverse_reshape,
                        0.5 * inverse_std_reshape,
                    )
                ),
                inverse_std_reshape,
            )
            drift_t = dmean + torch.einsum(
                "...ij,...j->...i", a.reshape(B, 1, T, D, D), xt - mean
            )

        else:
            R_inverse = torch.matmul(self.sigma, torch.transpose(self.sigma, 0, 1))
            R_inverse_reshape = R_inverse.unsqueeze(0).unsqueeze(0).repeat(B, T, 1, 1)
            dstd_reshape = torch.diag_embed(dstd.unsqueeze(2).repeat(1, 1, D))
            inverse_std_reshape = torch.diag_embed(
                (1 / std).unsqueeze(2).repeat(1, 1, D)
            )
            a = torch.einsum(
                "...ij,...jk->...ik",
                (
                    -dstd_reshape
                    - torch.einsum(
                        "...ij,...jk->...ik",
                        R_inverse_reshape,
                        0.5 * inverse_std_reshape,
                    )
                ),
                inverse_std_reshape,
            )
            drift_t = -dmean + torch.einsum(
                "...ij,...j->...i", a.reshape(B, 1, T, D, D), xt - mean
            )

        ft = self.ft(t, xt, direction)
        assert drift_t.shape == ft.shape == xt.shape
        return drift_t - ft

    def ut_zeros(self, t, xt, direction, create_graph_jvp=None, verbose=False):
        return torch.zeros_like(xt).to(xt.device)

    def drift(self, x, t, N, direction):
        """
        x: (B*N, D)
        t: (B*N,)
        ===
        ut: (B*N, D)
        """
        assert torch.allclose(t, t[0] * torch.ones_like(t))
        BN, D = x.shape
        assert BN % N == 0
        B = BN // N

        _t = t[0].reshape(1)
        _x = x.reshape(B, N, 1, D)
        u = self.ut(_t, _x, direction)

        assert u.shape == _x.shape
        return u.reshape(B * N, D)

    def forward(self, t, N, direction):
        """
        t: (T,)
        ===
        xt: (B, N, T, D)
        ut: (B, N, T, D)
        """
        xt = self.sample_xt(t, N)

        B, N, T, D = xt.shape
        assert t.shape == (T,)

        ut = self.ut(t, xt, direction)
        assert ut.shape == xt.shape

        return xt, ut


################################################################################################


def init_spline(x0, x1, n_knots):

    T, (B, D) = n_knots, x0.shape
    assert x1.shape == (B, D)

    t = torch.linspace(0, 1, T, device=x0.device)
    tt = t.reshape(1, T, 1)
    xt = (1 - tt) * x0.reshape(B, 1, D) + tt * x1.reshape(B, 1, D)

    assert t.shape == (T,)
    assert xt.shape == (B, T, D)
    return MeanSpline(t, xt)
