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
            x = x[None, :, None, :].repeat(
                (self.B, 1, 1, 1)
            )  
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
    def __init__(
        self, dim=2, hdims=[256, 128, 64], scaling_factor=1.0
    ):  
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
    
class TwoBoundarySigmoidMLP(torch.nn.Module):
    def __init__(self, dim=10, hdims=[128, 128], gamma=3.0, gamma2=3.0, gamma3=3.0, #stopping_function=None, 
                 scaling_factor=1.0):
        super().__init__()

        self.dim = dim
        self.gamma = gamma
        self.gamma2 = gamma2
        self.gamma3 = gamma3
        #self.stopping_function = stopping_function
        self.sigmoid_layers = nn.Sequential(
            # nn.Linear(2, hdims[0]),
            nn.Linear(3, hdims[0]),
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

    def forward(self, t, s, stopping_function_output, stopping_function_output_int_cumsum, init_stopping_function_output, final_stopping_function_output, stopping_timestep_values):
        # not_stopped = (stopping_timestep_values > 1 - 1e-3).to(torch.int)
        # ts = torch.cat((t.unsqueeze(1), s.unsqueeze(1)), dim=1)
        # sigmoid_layers_output = self.sigmoid_layers(ts).reshape(-1, 1, self.dim, self.dim)
        ts_zero = torch.cat((t.unsqueeze(1), s.unsqueeze(1), torch.zeros_like(s).to(s.device).unsqueeze(1)), dim=1)
        sigmoid_layers_output_stopped = self.sigmoid_layers(ts_zero).reshape(-1, 1, self.dim, self.dim)
        ts_one = torch.cat((t.unsqueeze(1), s.unsqueeze(1), torch.ones_like(s).to(s.device).unsqueeze(1)), dim=1)
        sigmoid_layers_output_not_stopped = self.sigmoid_layers(ts_one).reshape(-1, 1, self.dim, self.dim)
        # print(f'torch.mean(sigmoid_layers_output): {torch.mean(sigmoid_layers_output)}')
        exp_factor_1 = (
            torch.exp(self.gamma * (s - t)).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        )
        # init_stopped = (init_stopping_function_output > 0).to(torch.int).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        identity = torch.eye(self.dim).unsqueeze(0).unsqueeze(0).to(t.device)
        # stopping_function_output = self.stopping_function(x) ## Finish
        exp_factor_2 = (
            torch.exp(self.gamma2 * stopping_function_output).unsqueeze(2).unsqueeze(3)
        )
        init_exp_factor_2 = (
            torch.exp(self.gamma2 * init_stopping_function_output).unsqueeze(2).unsqueeze(3)
        )
        final_exp_factor_2 = (
            torch.exp(self.gamma2 * final_stopping_function_output).unsqueeze(2).unsqueeze(3)
        )
        # print(f'torch.min(stopping_function_output): {torch.min(stopping_function_output)}, torch.max(stopping_function_output): {torch.max(stopping_function_output)}')
        # print(f'sigmoid_layers_output.shape: {sigmoid_layers_output.shape}')
        # print(f'stopping_function_output.shape: {stopping_function_output.shape}, init_stopping_function_output.shape: {init_stopping_function_output.shape}')
        # print(f'exp_factor_1.shape: {exp_factor_1.shape}, exp_factor_2.shape: {exp_factor_2.shape}, init_exp_factor_2.shape: {init_exp_factor_2.shape}')
        # output = (1 / exp_factor_1) * (1 - (1 / exp_factor_2)) / (1 - (1 / init_exp_factor_2)) * identity.repeat(ts.shape[0], 1, 1) + (
        #     1 - 1 / exp_factor_1
        # ) * (1 - (1 / exp_factor_2)) * sigmoid_layers_output
        # print(f'torch.mean(1 - (1 / exp_factor_1)): {torch.mean(1 - (1 / exp_factor_1))}')
        # print(f'torch.mean(1 - (1 / exp_factor_2)): {torch.mean(1 - (1 / exp_factor_2))}')
        # print(f'torch.mean(1 - (1 / init_exp_factor_2)): {torch.mean(1 - (1 / init_exp_factor_2) + 1e-4)}')
        # print(f'(1 - (1 / init_exp_factor_2)): {(1 - (1 / init_exp_factor_2) + 1e-4)}')
        # print(f'torch.mean((1 - (1 / exp_factor_2)) / (1 - (1 / init_exp_factor_2))): {torch.mean((1 - (1 / exp_factor_2) + 1e-4) / (1 - (1 / init_exp_factor_2) + 1e-4))}')
        # print(f'(1 - (1 / exp_factor_2)) / (1 - (1 / init_exp_factor_2)): {(1 - (1 / exp_factor_2) + 1e-4) / (1 - (1 / init_exp_factor_2) + 1e-4)}')
        # output1 = (1 / exp_factor_1) * (1 - (1 / exp_factor_2) + 1e-3) / (1 - (1 / init_exp_factor_2) + 1e-3) * identity.repeat(ts.shape[0], exp_factor_2.shape[1], 1, 1)
        
        # one_tensor = torch.tensor([1]).to(ts.device)
        # print(f'torch.min(1 / exp_factor_2): {torch.min(1 / exp_factor_2)}, torch.max(1 / exp_factor_2): {torch.max(1 / exp_factor_2)}')
        # print(f'torch.min(1 / init_exp_factor_2): {torch.min(1 / init_exp_factor_2)}, torch.max(1 / init_exp_factor_2): {torch.max(1 / init_exp_factor_2)}')
        # print(f'torch.min(1 / final_exp_factor_2): {torch.min(1 / final_exp_factor_2)}, torch.max(1 / final_exp_factor_2): {torch.max(1 / final_exp_factor_2)}')
        # output1 = (1 / exp_factor_1) * (torch.maximum(one_tensor,(1 / final_exp_factor_2)) - (1 / exp_factor_2) + 1e-3) / (torch.maximum(one_tensor,(1 / final_exp_factor_2)) - (1 / init_exp_factor_2) + 1e-3) * identity.repeat(ts.shape[0], exp_factor_2.shape[1], 1, 1)
        # print(f'torch.min((torch.maximum(one_tensor,(1 / final_exp_factor_2)) - (1 / exp_factor_2) + 1e-3) / (torch.maximum(one_tensor,(1 / final_exp_factor_2)) - (1 / init_exp_factor_2) + 1e-3)): {torch.min(((1 / final_exp_factor_2) - (1 / exp_factor_2) + 1e-3) / ((1 / final_exp_factor_2) - (1 / init_exp_factor_2) + 1e-3))}')
        # print(f'torch.max((torch.maximum(one_tensor,(1 / final_exp_factor_2)) - (1 / exp_factor_2) + 1e-3) / (torch.maximum(one_tensor,(1 / final_exp_factor_2)) - (1 / init_exp_factor_2) + 1e-3)): {torch.max(((1 / final_exp_factor_2) - (1 / exp_factor_2) + 1e-3) / ((1 / final_exp_factor_2) - (1 / init_exp_factor_2) + 1e-3))}')
        # print(f'(torch.maximum(one_tensor,(1 / final_exp_factor_2)) - (1 / exp_factor_2) + 1e-3) / (torch.maximum(one_tensor,(1 / final_exp_factor_2)) - (1 / init_exp_factor_2) + 1e-3): {((1 / final_exp_factor_2) - (1 / exp_factor_2) + 1e-3) / ((1 / final_exp_factor_2) - (1 / init_exp_factor_2) + 1e-3)}')
        
        # print(f'stopping_function_output_int_cumsum.shape: {stopping_function_output_int_cumsum.shape}')
        # print(f'torch.min(stopping_function_output_int_cumsum): {torch.min(stopping_function_output_int_cumsum)}')
        # print(f'torch.max(stopping_function_output_int_cumsum): {torch.max(stopping_function_output_int_cumsum)}')
        
        # not_stopped = (stopping_function_output > 0).to(torch.int)
        # not_stopped_sum = torch.sum(not_stopped, dim=0)
        # factor1 = torch.cumsum(not_stopped, )
        # print(f'not_stopped.shape: {not_stopped.shape}')

        # print(f's.shape: {s.shape}, t.shape: {t.shape}, stopping_timestep_values.shape: {stopping_timestep_values.shape}')
        # factor1 = torch.nan_to_num(1 - torch.minimum((s - t).unsqueeze(1) / (torch.abs(stopping_timestep_values - t.unsqueeze(1)) + 1e-7), torch.tensor([1]).to(ts.device)), nan=0.0)
        factor1 = torch.nan_to_num(1 - torch.minimum((1-torch.exp(- self.gamma * (s - t))).unsqueeze(1) / (1 - torch.exp(- self.gamma * torch.abs(stopping_timestep_values - t.unsqueeze(1))) + 1e-7), torch.tensor([1]).to(t.device)), nan=0.0)
        # factor1 = torch.nan_to_num(1 - torch.minimum((1-torch.exp(- self.gamma * (fractional_s - t.unsqueeze(1)))) / (1 - torch.exp(- self.gamma * torch.abs(stopping_timestep_values - t.unsqueeze(1))) + 1e-7), torch.tensor([1]).to(t.device)), nan=0.0)
        factor1_non_zero = (stopping_timestep_values - 1e-3 > s.unsqueeze(1)).to(torch.int)
        # factor1_non_zero = (stopping_timestep_values - 1e-3 > fractional_s.unsqueeze(1)).to(torch.int)
        factor1 = factor1 * factor1_non_zero
        # print(f'torch.max(factor1): {torch.max(factor1)}, torch.min(factor1): {torch.min(factor1)}')
        # print(f'torch.min(factor1): {torch.min(factor1)}, torch.max(factor1): {torch.max(factor1)}')
        # output1 = stopping_function_output_int_cumsum.unsqueeze(2).unsqueeze(3) * identity.repeat(ts.shape[0], exp_factor_2.shape[1], 1, 1)
        # output1 = factor1.unsqueeze(2).unsqueeze(3) * identity.repeat(t.shape[0], exp_factor_2.shape[1], 1, 1)
        exp_gamma3_fun = lambda x: torch.exp(- self.gamma3 * x)
        not_stopped = (stopping_timestep_values > 1 - 1e-3).to(torch.int)
        output1 = ((1 - not_stopped) * factor1 + not_stopped * exp_gamma3_fun(s - t).unsqueeze(1)).unsqueeze(2).unsqueeze(3) * identity.repeat(t.shape[0], not_stopped.shape[1], 1, 1)
        # print(f'torch.mean(output1): {torch.mean(output1)}')

        # print(f'torch.min((1 - (1 / init_exp_factor_2) + 1e-3)): {torch.min((1 - (1 / init_exp_factor_2) + 1e-3))}')
        # print(f'torch.min(output1): {torch.min(output1)}')
        # print(f'torch.mean(output1**2): {torch.mean(output1**2)}')
        # print(f'torch.mean(output1): {torch.mean(output1)}')
        # print(f'output1.shape: {output1.shape}')
        # output2 = (1 - 1 / exp_factor_1) * (1 - (1 / exp_factor_2)) * sigmoid_layers_output
        # factor_fun_1 = lambda x: 2 * torch.sqrt((x + 1e-6) * (1 - x + 1e-6))
        # factor_fun_1 = lambda x: 4 * x * (1 - x)
        factor_fun_1 = lambda x: (1 - torch.exp(- self.gamma2 * x)) * (torch.exp(- self.gamma2 * x) - torch.exp(- self.gamma2))
        # s_not_stopped = (s.unsqueeze(1) < stopping_timestep_values).to(torch.int)
        # factor_fun_2 = lambda x: torch.sqrt(x + 1e-6)
        # exp_gamma3_fun = lambda x: (1 - torch.exp(- self.gamma2 * x))

        # factor_fun_1 = lambda x: 2 * torch.sqrt(x * (1 - x))
        # output2 = (1 - 1 / exp_factor_1) * ((1 / final_exp_factor_2) - (1 / exp_factor_2)) * sigmoid_layers_output
        # output2 = factor_fun_1(stopping_function_output_int_cumsum).unsqueeze(2).unsqueeze(3) * sigmoid_layers_output
        # output2 = factor_fun_1(factor1).unsqueeze(2).unsqueeze(3) * sigmoid_layers_output
        # output2 = ((1 - not_stopped) * factor_fun_1(factor1) + not_stopped * factor_fun_2(factor1)).unsqueeze(2).unsqueeze(3) * sigmoid_layers_output
        output2 = ((1 - not_stopped) * factor_fun_1(factor1)).unsqueeze(2).unsqueeze(3) * sigmoid_layers_output_stopped + (not_stopped * (1 - exp_gamma3_fun(s - t).unsqueeze(1))).unsqueeze(2).unsqueeze(3) * sigmoid_layers_output_not_stopped
        # output2 = (factor_fun_1(factor1)).unsqueeze(2).unsqueeze(3) * sigmoid_layers_output_stopped
        # print(f'torch.min(factor_fun(stopping_function_output_int_cumsum)): {torch.min(factor_fun(stopping_function_output_int_cumsum))}')
        # print(f'torch.max(factor_fun(stopping_function_output_int_cumsum)): {torch.max(factor_fun(stopping_function_output_int_cumsum))}')
        
        # print(f'torch.min((1 - 1 / exp_factor_1) * ((1 / final_exp_factor_2) - (1 / exp_factor_2))): {torch.min((1 - 1 / exp_factor_1) * ((1 / final_exp_factor_2) - (1 / exp_factor_2)))}')
        # print(f'torch.max((1 - 1 / exp_factor_1) * ((1 / final_exp_factor_2) - (1 / exp_factor_2))): {torch.max((1 - 1 / exp_factor_1) * ((1 / final_exp_factor_2) - (1 / exp_factor_2)))}')
        # print(f'torch.mean(output2**2): {torch.mean(output2**2)}')
        # print(f'torch.mean(output2): {torch.mean(output2)}')
        # print(f'output2.shape: {output2.shape}')
        output = output1 + output2
        # output = output1
        return output

        # ts = torch.cat((t.unsqueeze(1), s.unsqueeze(1)), dim=1)
        # sigmoid_layers_output = self.sigmoid_layers(ts).reshape(-1, self.dim, self.dim)
        # exp_factor = (
        #     torch.exp(self.gamma * (ts[:, 1] - ts[:, 0])).unsqueeze(1).unsqueeze(2)
        # )
        # identity = torch.eye(self.dim).unsqueeze(0).to(ts.device)
        # output = (1 / exp_factor) * identity.repeat(ts.shape[0], 1, 1) + (
        #     1 - 1 / exp_factor
        # ) * sigmoid_layers_output
        # return output.unsqueeze(1)
