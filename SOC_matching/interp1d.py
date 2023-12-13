# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory.
import torch
import torch.nn.functional as F


def linear_interp1d(x, y, mask, xs):
    """Linear splines.
    Inputs:
        x: (T, N) ### N = 1
        y: (T, N, D) ### T = 1000
        mask: (T, N) ### None
        xs: (S, N) ### S = 200
    """
    T, N, D = y.shape
    S = xs.shape[0]

    if mask is None:
        mask = torch.ones_like(x).bool()

    m = (y[1:] - y[:-1]) / (x[1:] - x[:-1] + 1e-10).unsqueeze(-1)

    left = torch.searchsorted(x[1:].T.contiguous(), xs.T.contiguous(), side="left").T
    mask_l = F.one_hot(left, T).permute(0, 2, 1).reshape(S, T, N, 1)

    x = x.reshape(1, T, N, 1)
    y = y.reshape(1, T, N, D)
    m = m.reshape(1, T - 1, N, D)
    xs = xs.reshape(S, N, 1)

    x0 = torch.sum(x * mask_l, dim=1)
    p0 = torch.sum(y * mask_l, dim=1)
    m0 = torch.sum(m * mask_l[:, :-1], dim=1)

    t = xs - x0

    return t * m0 + p0


def cubic_interp1d(x, y, mask, xs):
    """
    Inputs:
        x: (T, N)
        y: (T, N, D)
        mask: (T, N)
        xs: (S, N)
    """
    T, N, D = y.shape
    S = xs.shape[0]

    if x.shape == xs.shape:
        if torch.linalg.norm(x - xs) == 0:
            return y

    mask = mask.unsqueeze(-1)

    fd = (y[1:] - y[:-1]) / (x[1:] - x[:-1] + 1e-10).unsqueeze(-1)
    # Set tangents for the interior points.
    m = torch.cat([(fd[1:] + fd[:-1]) / 2, torch.zeros_like(fd[0:1])], dim=0)
    # Set tangent for the right end point.
    m = torch.where(torch.cat([mask[2:], torch.zeros_like(mask[0:1])]), m, fd)
    # Set tangent for the left end point.
    m = torch.cat([fd[[0]], m], dim=0)

    mask = mask.squeeze(-1)

    left = torch.searchsorted(x[1:].T.contiguous(), xs.T.contiguous(), side="left").T
    right = (left + 1) % mask.sum(0).long()
    mask_l = F.one_hot(left, T).permute(0, 2, 1).reshape(S, T, N, 1)
    mask_r = F.one_hot(right, T).permute(0, 2, 1).reshape(S, T, N, 1)

    x = x.reshape(1, T, N, 1)
    y = y.reshape(1, T, N, D)
    m = m.reshape(1, T, N, D)
    xs = xs.reshape(S, N, 1)

    x0 = torch.sum(x * mask_l, dim=1)
    x1 = torch.sum(x * mask_r, dim=1)
    p0 = torch.sum(y * mask_l, dim=1)
    p1 = torch.sum(y * mask_r, dim=1)
    m0 = torch.sum(m * mask_l, dim=1)
    m1 = torch.sum(m * mask_r, dim=1)

    dx = x1 - x0
    t = (xs - x0) / (dx + 1e-10)

    return (
        t**3 * (2 * p0 + m0 - 2 * p1 + m1)
        + t**2 * (-3 * p0 + 3 * p1 - 2 * m0 - m1)
        + t * m0
        + p0
    )