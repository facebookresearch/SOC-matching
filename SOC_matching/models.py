# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory.
import torch
import torch.nn as nn


class LinearControl:
    def __init__(self, u, T):
        self.u = u
        self.T = T

    def evaluate(self, t):
        nsteps = self.u.shape[0]
        idx = torch.floor((nsteps - 1) * t / self.T).to(torch.int64)
        return self.u[idx]

    def evaluate_tensor(self, t):
        nsteps = self.u.shape[0]
        idx = torch.floor((nsteps - 1) * t / self.T).to(torch.int64)
        return self.u[idx, :, :]

    def __call__(self, t, x, t_is_tensor=False):
        if not t_is_tensor:
            if len(self.evaluate(t).shape) == 2:
                evaluate_t = self.evaluate(t)
            else:
                evaluate_t = self.evaluate(t)[0, :, :]
            if len(x.shape) == 2:
                return torch.einsum("ij,bj->bi", evaluate_t, x)
            elif len(x.shape) == 3:
                return torch.einsum("ij,abj->abi", evaluate_t, x)
        else:
            if len(x.shape) == 2:
                return torch.einsum("bij,bj->bi", self.evaluate_tensor(t), x)
            elif len(x.shape) == 3:
                return torch.einsum("aij,abj->abi", self.evaluate_tensor(t), x)


class ConstantControlQuadratic:
    def __init__(self, pt, sigma, T, batchsize):
        self.pt = pt
        self.sigma = sigma
        self.T = T
        self.batchsize = batchsize

    def evaluate_pt(self, t):
        nsteps = self.pt.shape[0]
        idx = torch.floor(nsteps * t / self.T).to(torch.int64)
        return self.pt[idx]

    def __call__(self, t, x):
        control = -torch.matmul(
            torch.transpose(self.sigma, 0, 1), self.evaluate_pt(t)
        ).unsqueeze(0)
        return control


class ConstantControlLinear:
    def __init__(self, ut, T):
        self.ut = ut
        self.T = T

    def evaluate_ut(self, t):
        nsteps = self.ut.shape[0]
        idx = torch.floor(nsteps * t / self.T).to(torch.int64)
        return self.ut[idx]

    def evaluate_ut_tensor(self, t):
        nsteps = self.ut.shape[0]
        idx = torch.floor((nsteps - 1) * t / self.T).to(torch.int64)
        return self.ut[idx, :]

    def __call__(self, t, x, t_is_tensor=False):
        if not t_is_tensor:
            control = self.evaluate_ut(t).unsqueeze(0).repeat(x.shape[0], 1)
        else:
            control = self.evaluate_ut_tensor(t).unsqueeze(1).repeat(1, x.shape[1], 1)
        return control


class LowDimControl:
    def __init__(self, ut, T, xb, dim, delta_t, delta_x):
        self.ut = ut
        self.T = T
        self.xb = xb
        self.dim = dim
        self.delta_t = delta_t
        self.delta_x = delta_x

    def evaluate_ut(self, t, x):
        x_reshape = x.reshape(-1, self.dim)
        t = torch.tensor([t]).to(x.device)
        t = t.reshape(-1, 1).expand(x_reshape.shape[0], 1)
        tx = torch.cat([t, x_reshape], dim=-1)

        idx = torch.zeros_like(tx).to(tx.device).to(torch.int64)

        idx[:, 0] = torch.ceil(tx[:, 0] / self.delta_t).to(torch.int64)

        idx[:, 1:] = torch.floor((tx[:, 1:] + self.xb) / self.delta_x).to(torch.int64)
        idx[:, 1:] = torch.minimum(
            idx[:, 1:],
            torch.tensor(self.ut.shape[1] - 1).to(torch.int64).to(idx.device),
        )
        idx[:, 1:] = torch.maximum(
            idx[:, 1:], torch.tensor(0).to(torch.int64).to(idx.device)
        )
        control = torch.zeros_like(x_reshape)
        for j in range(self.dim):
            idx_j = idx[:, [0, j + 1]]
            ut_j = self.ut[:, :, j]
            control_j = ut_j[idx_j[:, 0], idx_j[:, 1]]
            control[:, j] = control_j
        return control

    def evaluate_ut_tensor(self, t, x):
        t = t.reshape(-1, 1, 1).expand(x.shape[0], x.shape[1], 1)
        tx = torch.cat([t, x], dim=-1)
        tx_shape = tx.shape
        tx = tx.reshape(-1, tx.shape[2])

        idx = torch.zeros_like(tx).to(tx.device).to(torch.int64)

        idx[:, 0] = torch.ceil(tx[:, 0] / self.delta_t).to(torch.int64)

        idx[:, 1:] = torch.floor((tx[:, 1:] + self.xb) / self.delta_x).to(torch.int64)
        idx[:, 1:] = torch.minimum(
            idx[:, 1:],
            torch.tensor(self.ut.shape[1] - 1).to(torch.int64).to(idx.device),
        )
        idx[:, 1:] = torch.maximum(
            idx[:, 1:], torch.tensor(0).to(torch.int64).to(idx.device)
        )
        control = torch.zeros_like(tx).to(tx.device)
        for j in range(self.dim):
            idx_j = idx[:, [0, j + 1]]
            ut_j = self.ut[:, :, j]
            control_j = ut_j[idx_j[:, 0], idx_j[:, 1]]
            control[:, j + 1] = control_j
        control = torch.reshape(control, tx_shape)[:, :, 1:]
        return control

    def __call__(self, t, x, t_is_tensor=False):
        if not t_is_tensor:
            return self.evaluate_ut(t, x)
        else:
            return self.evaluate_ut_tensor(t, x)


class RestrictedControl:
    def __init__(self, gpath, sigma, b, device, T, B):
        self.device = device
        self.gpath = gpath
        self.sigma = sigma
        self.sigma_inverse = torch.inverse(sigma)
        self.b = b
        self.T = T
        self.B = B

    def __call__(self, t, x, verbose=False):
        if verbose:
            print(f"x.shape in control: {x.shape}")
        len_2 = len(x.shape) == 2
        if len(x.shape) == 2:
            x = x[None, :, None, :].repeat((self.B, 1, 1, 1))
            t = torch.tensor([t]).to(self.device)
            t = t + 1e-4 if t < self.T / 2 else t - 1e-4
            control = self.gpath.ut(
                t, x, direction="fwd", create_graph_jvp=False, verbose=False
            )
            control_reshape = control[0, :, 0, :]
            b_eval = self.b(t, x[0, :, 0, :])
            output = torch.einsum(
                "ij,...j->...i", self.sigma_inverse, control_reshape - b_eval
            )
            return output
        if len(x.shape) == 3:
            x_transpose = (
                torch.transpose(x, 0, 1).unsqueeze(0).repeat((self.B, 1, 1, 1))
            )
            t_copy = t.clone().to(t.device)
            t_copy[t < self.T / 2] = t[t < self.T / 2] + 1e-4
            t_copy[t > self.T / 2] = t[t > self.T / 2] - 1e-4
            control = self.gpath.ut(
                t_copy,
                x_transpose,
                direction="fwd",
                create_graph_jvp=False,
                verbose=False,
            ).detach()
            control_reshape = torch.transpose(control.squeeze(0), 0, 1)
            b_eval = self.b(t_copy, x_transpose.squeeze(0))
            output = torch.einsum(
                "ij,...j->...i", self.sigma_inverse, control.squeeze(0) - b_eval
            ).detach()
            return torch.transpose(output, 0, 1)


class FullyConnectedUNet(torch.nn.Module):
    def __init__(self, dim=2, hdims=[256, 128, 64], scaling_factor=1.0):
        super().__init__()

        def initialize_weights(layer, scaling_factor):
            for m in layer:
                if isinstance(m, nn.Linear):
                    m.weight.data *= scaling_factor
                    m.bias.data *= scaling_factor

        self.down_0 = nn.Sequential(nn.Linear(dim + 1, hdims[0]), nn.ReLU())
        self.down_1 = nn.Sequential(nn.Linear(hdims[0], hdims[1]), nn.ReLU())
        self.down_2 = nn.Sequential(nn.Linear(hdims[1], hdims[2]), nn.ReLU())
        initialize_weights(self.down_0, scaling_factor)
        initialize_weights(self.down_1, scaling_factor)
        initialize_weights(self.down_2, scaling_factor)

        self.res_0 = nn.Sequential(nn.Linear(dim + 1, dim))
        self.res_1 = nn.Sequential(nn.Linear(hdims[0], hdims[0]))
        self.res_2 = nn.Sequential(nn.Linear(hdims[1], hdims[1]))
        initialize_weights(self.res_0, scaling_factor)
        initialize_weights(self.res_1, scaling_factor)
        initialize_weights(self.res_2, scaling_factor)

        self.up_2 = nn.Sequential(nn.Linear(hdims[2], hdims[1]), nn.ReLU())
        self.up_1 = nn.Sequential(nn.Linear(hdims[1], hdims[0]), nn.ReLU())
        self.up_0 = nn.Sequential(nn.Linear(hdims[0], dim), nn.ReLU())
        initialize_weights(self.up_0, scaling_factor)
        initialize_weights(self.up_1, scaling_factor)
        initialize_weights(self.up_2, scaling_factor)

    def forward(self, x):
        residual0 = x
        residual1 = self.down_0(x)
        residual2 = self.down_1(residual1)
        residual3 = self.down_2(residual2)

        out2 = self.up_2(residual3) + self.res_2(residual2)
        out1 = self.up_1(out2) + self.res_1(residual1)
        out0 = self.up_0(out1) + self.res_0(residual0)
        return out0


class SigmoidMLP(torch.nn.Module):
    def __init__(self, dim=10, hdims=[128, 128], gamma=3.0, scaling_factor=1.0):
        super().__init__()

        self.dim = dim
        self.gamma = gamma
        self.sigmoid_layers = nn.Sequential(
            nn.Linear(2, hdims[0]),
            nn.ReLU(),
            nn.Linear(hdims[0], hdims[1]),
            nn.ReLU(),
            nn.Linear(hdims[1], dim**2),
        )

        self.scaling_factor = scaling_factor
        for m in self.sigmoid_layers:
            if isinstance(m, nn.Linear):
                m.weight.data *= self.scaling_factor
                m.bias.data *= self.scaling_factor

    def forward(self, t, s):
        ts = torch.cat((t.unsqueeze(1), s.unsqueeze(1)), dim=1)
        sigmoid_layers_output = self.sigmoid_layers(ts).reshape(-1, self.dim, self.dim)
        exp_factor = (
            torch.exp(self.gamma * (ts[:, 1] - ts[:, 0])).unsqueeze(1).unsqueeze(2)
        )
        identity = torch.eye(self.dim).unsqueeze(0).to(ts.device)
        output = (1 / exp_factor) * identity.repeat(ts.shape[0], 1, 1) + (
            1 - 1 / exp_factor
        ) * sigmoid_layers_output
        return output
    
class Identity(torch.nn.Module):
    def __init__(self, dim=10, output_matrix=False):
        super().__init__()

        self.dim = dim
        self.output_matrix = output_matrix

    def forward(self, t, s):
        ts = torch.cat((t.unsqueeze(1), s.unsqueeze(1)), dim=1)
        # sigmoid_layers_output = self.sigmoid_layers(ts).reshape(-1, self.dim, self.dim)
        # exp_factor = (
        #     torch.exp(self.gamma * (ts[:, 1] - ts[:, 0])).unsqueeze(1).unsqueeze(2)
        # )
        if self.output_matrix:
            identity = torch.eye(self.dim).unsqueeze(0).to(ts.device)
            # output = (1 / exp_factor) * identity.repeat(ts.shape[0], 1, 1) + (
            #     1 - 1 / exp_factor
            # ) * sigmoid_layers_output
            return identity.repeat(ts.shape[0], 1, 1)
        else:
            return torch.ones(ts.shape[0]).to(ts.device)
    
class ScalarSigmoidMLP(torch.nn.Module):
    def __init__(self, dim=10, hdims=[128, 128], gamma=3.0, scaling_factor=1.0, output_matrix=False):
        super().__init__()

        self.dim = dim
        self.gamma = gamma
        self.output_matrix = output_matrix
        self.sigmoid_layers = nn.Sequential(
            nn.Linear(2, hdims[0]),
            nn.ReLU(),
            nn.Linear(hdims[0], hdims[1]),
            nn.ReLU(),
            nn.Linear(hdims[1], 1),
        )

        self.scaling_factor = scaling_factor
        for m in self.sigmoid_layers:
            if isinstance(m, nn.Linear):
                m.weight.data *= self.scaling_factor
                m.bias.data *= self.scaling_factor

        # print(f'ScalarSigmoidMLP, self.output_matrix: {self.output_matrix}')

    def forward(self, t, s):
        ts = torch.cat((t.unsqueeze(1), s.unsqueeze(1)), dim=1)
        # print(f'ScalarSigmoidMLP forward')
        # sigmoid_layers_output = self.sigmoid_layers(ts).reshape(-1, self.dim, self.dim)
        if self.output_matrix:
            sigmoid_layers_output = self.sigmoid_layers(ts).unsqueeze(2)
            exp_factor = (
                torch.exp(self.gamma * (ts[:, 1] - ts[:, 0])).unsqueeze(1).unsqueeze(2)
            )
            identity = torch.eye(self.dim).unsqueeze(0).to(ts.device)
            # print(f'exp_factor.shape: {exp_factor.shape}, sigmoid_layers_output.shape: {sigmoid_layers_output.shape}')
            output = ( (1 / exp_factor) + (1 - 1 / exp_factor) * sigmoid_layers_output ) * identity.repeat(ts.shape[0], 1, 1)
            return output
        else:
            sigmoid_layers_output = self.sigmoid_layers(ts).squeeze(1)
            exp_factor = (
                torch.exp(self.gamma * (ts[:, 1] - ts[:, 0]))
            )
            # print(f'sigmoid_layers_output.shape: {sigmoid_layers_output.shape}, exp_factor.shape: {exp_factor.shape}')
            output = (1 / exp_factor) + (1 - 1 / exp_factor) * sigmoid_layers_output
            return output
    
class DiagonalSigmoidMLP(torch.nn.Module):
    def __init__(self, dim=10, hdims=[128, 128], gamma=3.0, scaling_factor=1.0, output_matrix=False):
        super().__init__()

        self.dim = dim
        self.gamma = gamma
        self.output_matrix = output_matrix
        self.sigmoid_layers = nn.Sequential(
            nn.Linear(2, hdims[0]),
            nn.ReLU(),
            nn.Linear(hdims[0], hdims[1]),
            nn.ReLU(),
            nn.Linear(hdims[1], dim),
        )

        self.scaling_factor = scaling_factor
        for m in self.sigmoid_layers:
            if isinstance(m, nn.Linear):
                m.weight.data *= self.scaling_factor
                m.bias.data *= self.scaling_factor

    def forward(self, t, s):
        ts = torch.cat((t.unsqueeze(1), s.unsqueeze(1)), dim=1)
        if self.output_matrix:
            sigmoid_layers_output = torch.diag_embed(self.sigmoid_layers(ts)) #.unsqueeze(2)
            exp_factor = (
                torch.exp(self.gamma * (ts[:, 1] - ts[:, 0])).unsqueeze(1).unsqueeze(2)
            )
            identity = torch.eye(self.dim).unsqueeze(0).to(ts.device)
            output = (1 / exp_factor) * identity.repeat(ts.shape[0], 1, 1) + (
                1 - 1 / exp_factor
            ) * sigmoid_layers_output
            return output
        else:
            sigmoid_layers_output = self.sigmoid_layers(ts) #.unsqueeze(2)
            exp_factor = torch.exp(self.gamma * (ts[:, 1] - ts[:, 0])).unsqueeze(1) #.unsqueeze(2)
            # print(f'sigmoid_layers_output.shape: {sigmoid_layers_output.shape}, exp_factor.shape: {exp_factor.shape}')
            output = (1 / exp_factor) + (1 - 1 / exp_factor) * sigmoid_layers_output
            return output
    
class TwoBoundaryScalarSigmoidMLP(torch.nn.Module):
    def __init__(self, dim=10, hdims=[128, 128], gamma=3.0, gamma2=3.0, gamma3=10.0, scaling_factor=1.0, T=1.0, output_matrix=False):
        super().__init__()

        self.dim = dim
        self.gamma = gamma
        self.gamma2 = gamma2
        self.gamma3 = gamma3
        self.T = T
        self.output_matrix = output_matrix
        self.sigmoid_layers = nn.Sequential(
            nn.Linear(2, hdims[0]),
            nn.ReLU(),
            nn.Linear(hdims[0], hdims[1]),
            nn.ReLU(),
            nn.Linear(hdims[1], 1),
        )

        self.scaling_factor = scaling_factor
        for m in self.sigmoid_layers:
            if isinstance(m, nn.Linear):
                m.weight.data *= self.scaling_factor
                m.bias.data *= self.scaling_factor

    def forward(self, t, s):
        ts = torch.cat((t.unsqueeze(1), s.unsqueeze(1)), dim=1)
        sigmoid_layers_output = self.sigmoid_layers(ts).unsqueeze(2)

        exp_denominator = 1 / (self.T - ts[:, 0])
        # exp_denominator[-1] = 1.0
        factor1 = torch.exp(- self.gamma3 * (ts[:, 1] - ts[:, 0]) / exp_denominator)
        factor1 = torch.nan_to_num(factor1, nan=1.0, posinf=1.0, neginf=1.0)

        factor2 = (1 - torch.exp(-self.gamma * (ts[:, 1] - ts[:, 0]))) * (torch.exp(-self.gamma2 * (ts[:, 1] - ts[:, 0])) - torch.exp(-self.gamma2 * (1 - ts[:, 0])))
        identity = torch.eye(self.dim).unsqueeze(0).to(ts.device)

        output = (factor1[:,None,None] + sigmoid_layers_output * factor2[:,None,None]) * identity.repeat(ts.shape[0], 1, 1)
        return output
    
class TwoBoundaryDiagonalSigmoidMLP(torch.nn.Module):
    def __init__(self, dim=10, hdims=[128, 128], gamma=3.0, gamma2=3.0, gamma3=10.0, scaling_factor=1.0, T=1.0, output_matrix=False):
        super().__init__()

        self.dim = dim
        self.gamma = gamma
        self.gamma2 = gamma2
        self.gamma3 = gamma3
        self.T = T
        self.output_matrix = output_matrix
        self.sigmoid_layers = nn.Sequential(
            nn.Linear(2, hdims[0]),
            nn.ReLU(),
            nn.Linear(hdims[0], hdims[1]),
            nn.ReLU(),
            nn.Linear(hdims[1], 1),
        )

        self.scaling_factor = scaling_factor
        for m in self.sigmoid_layers:
            if isinstance(m, nn.Linear):
                m.weight.data *= self.scaling_factor
                m.bias.data *= self.scaling_factor

    def forward(self, t, s):
        ts = torch.cat((t.unsqueeze(1), s.unsqueeze(1)), dim=1)
        sigmoid_layers_output = torch.diag_embed(self.sigmoid_layers(ts))

        # exp_denominator = 1 / (1 - ts[:, 0])
        # exp_denominator[-1] = 1.0
        # factor1 = torch.exp(- self.gamma3 * (ts[:, 1] - ts[:, 0]) / exp_denominator)
        # print(f'self.T: {self.T}')
        factor1 = 1 - (ts[:, 1] - ts[:, 0]) / (self.T - ts[:, 0])
        factor1 = torch.nan_to_num(factor1, nan=1.0, posinf=1.0, neginf=1.0)
        # print(f'torch.max(ts[:, 1]): {torch.max(ts[:, 1])}, torch.min(ts[:, 1]): {torch.min(ts[:, 1])}')
        # print(f'torch.max(ts[:, 0]): {torch.max(ts[:, 0])}, torch.min(ts[:, 0]): {torch.min(ts[:, 0])}')
        # print(f'torch.max(ts[:, 1] - ts[:, 0]): {torch.max(ts[:, 1] - ts[:, 0])}, torch.min(ts[:, 1] - ts[:, 0]): {torch.min(ts[:, 1] - ts[:, 0])}')
        # print(f'torch.max((ts[:, 1] - ts[:, 0]) / (self.T + 1e-4 - ts[:, 0])): {torch.max((ts[:, 1] - ts[:, 0]) / (self.T + 1e-4 - ts[:, 0]))}, torch.min((ts[:, 1] - ts[:, 0]) / (self.T - ts[:, 0])): {torch.min((ts[:, 1] - ts[:, 0]) / (self.T - ts[:, 0]))}')
        # print(f'torch.max(factor1): {torch.max(factor1)}, torch.min(factor1): {torch.min(factor1)}')
        argmax_factor1 = torch.argmax(factor1)
        argmin_factor1 = torch.argmin(factor1)
        # print(f'torch.argmax(factor1): {torch.argmax(factor1)}, torch.argmin(factor1): {torch.argmin(factor1)}')
        # factor1[-1] = 0.0
        # print(f'ts[argmax_factor1, 1]: {ts[argmax_factor1, 1]}, ts[argmax_factor1, 0]: {ts[argmax_factor1, 0]}')
        # print(f'ts[argmin_factor1, 1]: {ts[argmin_factor1, 1]}, ts[argmin_factor1, 0]: {ts[argmin_factor1, 0]}')
        # Count the number of NaNs
        num_nans = torch.isnan(factor1).sum().item()
        # print(f"Number of NaNs: {num_nans}")
        # Find positions of NaNs
        nan_positions = torch.nonzero(torch.isnan(factor1), as_tuple=False)
        # print(f'nan_positions: {nan_positions}, len(factor1): {len(factor1)}')
        # print(f'factor1[1830]: {factor1[1830]}, ts[1830, 0]: {ts[1830, 0]}, ts[1830, 1]: {ts[1830, 1]}')
        # print(f'factor1[-50:]: {factor1[-50:]}')
        # print(f'ts[-50:, 0]: {ts[-50:, 0]}, ts[-50:, 1]: {ts[-50:, 1]}')

        factor2 = (1 - torch.exp(-self.gamma * (ts[:, 1] - ts[:, 0]))) * (torch.exp(-self.gamma2 * (ts[:, 1] - ts[:, 0])) - torch.exp(-self.gamma2 * (1 - ts[:, 0])))
        identity = torch.eye(self.dim).unsqueeze(0).to(ts.device)

        output = factor1[:,None,None] * identity.repeat(ts.shape[0], 1, 1) + sigmoid_layers_output * factor2[:,None,None]
        # print(f'TwoBoundaryDiagonalSigmoidMLP torch.mean(output): {torch.mean(output)}, torch.mean(factor1): {torch.mean(factor1)}, torch.mean(sigmoid_layers_output): {torch.mean(sigmoid_layers_output)}, torch.mean(factor2): {torch.mean(factor2)}')
        return output


class TwoBoundarySigmoidMLP(torch.nn.Module):
    def __init__(
        self,
        dim=10,
        hdims=[128, 128],
        gamma=3.0,
        gamma2=3.0,
        gamma3=3.0,
        scaling_factor=1.0,
        T=1.0,
    ):
        super().__init__()

        self.dim = dim
        self.gamma = gamma
        self.gamma2 = gamma2
        self.gamma3 = gamma3
        self.T = T
        self.sigmoid_layers = nn.Sequential(
            nn.Linear(3, hdims[0]),
            nn.ReLU(),
            nn.Linear(hdims[0], hdims[1]),
            nn.ReLU(),
            nn.Linear(hdims[1], dim**2),
        )
        # the third input of sigmoid_layers specifies whether the process is stopped or not

        self.scaling_factor = scaling_factor
        for m in self.sigmoid_layers:
            if isinstance(m, nn.Linear):
                m.weight.data *= self.scaling_factor
                m.bias.data *= self.scaling_factor

    def forward(self, t, s, stopping_timestep_values):

        ts_zero = torch.cat(
            (
                t.unsqueeze(1),
                s.unsqueeze(1),
                torch.zeros_like(s).to(s.device).unsqueeze(1),
            ),
            dim=1,
        )
        sigmoid_layers_output_stopped = self.sigmoid_layers(ts_zero).reshape(
            -1, 1, self.dim, self.dim
        )
        # sigmoid_layers_output_stopped contains sigmoid_layers evaluations with third input 0

        ts_one = torch.cat(
            (
                t.unsqueeze(1),
                s.unsqueeze(1),
                torch.ones_like(s).to(s.device).unsqueeze(1),
            ),
            dim=1,
        )
        sigmoid_layers_output_not_stopped = self.sigmoid_layers(ts_one).reshape(
            -1, 1, self.dim, self.dim
        )
        # sigmoid_layers_output_stopped contains sigmoid_layers evaluations with third input 1

        identity = torch.eye(self.dim).unsqueeze(0).unsqueeze(0).to(t.device)

        factor1 = torch.nan_to_num(
            1
            - torch.minimum(
                (1 - torch.exp(-self.gamma * (s - t))).unsqueeze(1)
                / (
                    1
                    - torch.exp(
                        -self.gamma
                        * torch.abs(stopping_timestep_values - t.unsqueeze(1))
                    )
                    + 1e-7
                ),
                torch.tensor([1]).to(t.device),
            ),
            nan=0.0,
        )
        # factor_1 takes values in [0,1], value 1 when s=t and value 0 when s >= stopping time
        factor1_non_zero = (stopping_timestep_values - 1e-3 > s.unsqueeze(1)).to(
            torch.int
        )
        factor1 = factor1 * factor1_non_zero
        # factor_1 is only non-zero when the process is not stopped at time s

        exp_gamma3_fun = lambda x: torch.exp(-self.gamma3 * x)
        not_stopped = (stopping_timestep_values > self.T - 1e-3).to(torch.int)
        output1 = (
            (1 - not_stopped) * factor1
            + not_stopped * exp_gamma3_fun(s - t).unsqueeze(1)
        ).unsqueeze(2).unsqueeze(3) * identity.repeat(
            t.shape[0], not_stopped.shape[1], 1, 1
        )
        # when process is stopped, output1 = factor1 * identity, 
        # when process is not stopped, output1 = exp(- gamma3 * (s - t)) * identity

        fun_gamma2 = lambda x: (1 - torch.exp(-self.gamma2 * x)) * (
            torch.exp(-self.gamma2 * x) - torch.exp(-self.gamma2)
        )
        # fun_gamma2 takes values in [0,1] when input is in [0,1], it maps 0 to 0 and 1 to 0

        output2 = ((1 - not_stopped) * fun_gamma2(factor1)).unsqueeze(2).unsqueeze(
            3
        ) * sigmoid_layers_output_stopped + (
            not_stopped * (1 - exp_gamma3_fun(s - t).unsqueeze(1))
        ).unsqueeze(
            2
        ).unsqueeze(
            3
        ) * sigmoid_layers_output_not_stopped
        # when process is stopped, output2 = fun_gamma2(factor1) * sigmoid_layers_output_stopped, 
        # when process is not stopped, output2 = (1 - exp(- gamma3 * (s - t))) * sigmoid_layers_output_stopped

        output = output1 + output2
        return output