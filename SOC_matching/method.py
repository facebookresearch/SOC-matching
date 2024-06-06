# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory.
import numpy as np
import torch
import torch.nn as nn
import functorch
import nvidia_smi

from SOC_matching import utils, models


class NeuralSDE(torch.nn.Module):
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(
        self,
        device="cuda",
        dim=2,
        hdims=[256, 128, 64],
        hdims_M=[128, 128],
        u=None,
        lmbd=1.0,
        sigma=torch.eye(2),
        gamma=1.0,
        gamma2=1.0,
        gamma3=1.0,
        scaling_factor_nabla_V=1.0,
        scaling_factor_M=1.0,
        T=1.0,
        u_warm_start=None,
        use_warm_start=False,
        use_stopping_time=False,
        output_matrix=False,
    ):
        super().__init__()
        self.device = device
        self.dim = dim
        self.hdims = hdims
        self.hdims_M = hdims_M
        self.u = u
        self.lmbd = lmbd
        self.sigma = sigma
        self.gamma = gamma
        self.gamma2 = gamma2
        self.gamma3 = gamma3
        self.scaling_factor_nabla_V = scaling_factor_nabla_V
        self.scaling_factor_M = scaling_factor_M
        self.use_learned_control = False
        self.T = T
        self.u_warm_start = u_warm_start
        self.use_warm_start = use_warm_start
        self.use_stopping_time = use_stopping_time
        self.output_matrix = output_matrix

    # Control
    def control(self, t, x, verbose=False):
        if verbose:
            print(
                f"self.use_learned_control: {self.use_learned_control}, self.u: {self.u}"
            )
        if self.use_learned_control:
            if len(x.shape) == 2:
                x = x.reshape(-1, self.dim)
                t_expand = t.reshape(-1, 1).expand(x.shape[0], 1)
                tx = torch.cat([t_expand, x], dim=-1)
                learned_control = -torch.einsum(
                    "ij,bj->bi",
                    torch.transpose(self.sigma, 0, 1),
                    self.nabla_V(tx).reshape(x.shape),
                )
                if verbose:
                    print(
                        f"self.use_warm_start: {self.use_warm_start}, self.u_warm_start: {self.u_warm_start}"
                    )
                if self.use_warm_start and self.u_warm_start:
                    return learned_control + self.u_warm_start(t, x).detach()
                else:
                    return learned_control
            if len(x.shape) == 3:
                ts_repeat = t.unsqueeze(1).unsqueeze(2).repeat(1, x.shape[1], 1)
                tx = torch.cat([ts_repeat, x], dim=-1)
                tx_reshape = torch.reshape(tx, (-1, tx.shape[2]))

                # Evaluate nabla_V
                nabla_V = self.nabla_V(tx_reshape)
                nabla_V = torch.reshape(nabla_V, x.shape)

                learned_control = -torch.einsum(
                    "ij,abj->abi",
                    torch.transpose(self.sigma, 0, 1),
                    nabla_V,
                )
                if verbose:
                    print(
                        f"self.use_warm_start: {self.use_warm_start}, self.u_warm_start: {self.u_warm_start}"
                    )
                if self.use_warm_start and self.u_warm_start:
                    return learned_control + self.u_warm_start(t, x).detach()
                else:
                    return learned_control
        else:
            if self.u is None:
                return None
            else:
                return self.u(t, x)

    def initialize_models(self, algorithm, setting):
        self.nabla_V = models.FullyConnectedUNet(
            dim=self.dim,
            hdims=self.hdims,
            scaling_factor=self.scaling_factor_nabla_V,
        ).to(self.device)

        print(f"initialize_models, self.use_stopping_time: {self.use_stopping_time}")
        if self.use_stopping_time:
            self.gamma = torch.nn.Parameter(torch.tensor([self.gamma]).to(self.device))
            self.gamma2 = torch.nn.Parameter(
                torch.tensor([self.gamma2]).to(self.device)
            )
            self.gamma3 = torch.nn.Parameter(
                torch.tensor([self.gamma3]).to(self.device)
            )
            self.M = models.TwoBoundarySigmoidMLP(
                dim=self.dim,
                hdims=self.hdims_M,
                gamma=self.gamma,
                gamma2=self.gamma2,
                gamma3=self.gamma3,
                scaling_factor=self.scaling_factor_M,
            ).to(self.device)
        else:
            self.gamma = torch.nn.Parameter(torch.tensor([self.gamma]).to(self.device))
            if algorithm in ['SOCM_sc','UW_SOCM_sc','SOCM_cost_sc','SOCM_work_sc']:
                print(f'Using ScalarSigmoidMLP...')
                self.M = models.ScalarSigmoidMLP(
                    dim=self.dim,
                    hdims=self.hdims_M,
                    gamma=self.gamma,
                    scaling_factor=self.scaling_factor_M,
                    output_matrix=self.output_matrix,
                ).to(self.device)
            elif algorithm in ['SOCM_diag','UW_SOCM_diag','SOCM_cost_diag','SOCM_work_diag']:
                # if setting == "sampling_cox":
                #     print(f'Using DiagonalCNN...')
                #     self.M = models.DiagonalCNN(
                #         # dim=self.dim,
                #         gamma=self.gamma,
                #         device=self.device,
                #     ).to(self.device)
                # else:
                print(f'Using DiagonalSigmoidMLP...')
                self.M = models.DiagonalSigmoidMLP(
                    dim=self.dim,
                    hdims=self.hdims_M,
                    gamma=self.gamma,
                    scaling_factor=self.scaling_factor_M,
                    output_matrix=self.output_matrix,
                ).to(self.device)
            elif algorithm in ['SOCM_sc_2B','UW_SOCM_sc_2B','SOCM_cost_sc_2B','SOCM_work_sc_2B']:
                print(f'Using TwoBoundaryScalarSigmoidMLP...')
                self.gamma2 = torch.nn.Parameter(
                    torch.tensor([self.gamma2]).to(self.device)
                )
                self.M = models.TwoBoundaryScalarSigmoidMLP(
                    dim=self.dim,
                    hdims=self.hdims_M,
                    gamma=self.gamma,
                    gamma2=self.gamma2,
                    scaling_factor=self.scaling_factor_M,
                    T = self.T,
                    output_matrix=self.output_matrix,
                ).to(self.device)
            elif algorithm in ['SOCM_diag_2B','UW_SOCM_diag_2B','SOCM_cost_diag_2B','SOCM_work_diag_2B']:
                print(f'Using TwoBoundaryDiagonalSigmoidMLP...')
                self.gamma2 = torch.nn.Parameter(
                    torch.tensor([self.gamma2]).to(self.device)
                )
                self.M = models.TwoBoundaryDiagonalSigmoidMLP(
                    dim=self.dim,
                    hdims=self.hdims_M,
                    gamma=self.gamma,
                    gamma2=self.gamma2,
                    scaling_factor=self.scaling_factor_M,
                    T = self.T,
                    output_matrix=self.output_matrix,
                ).to(self.device)
            elif algorithm in ['SOCM_cost_identity','SOCM_work_identity','SOCM_cost_identity_2B','SOCM_work_identity_2B','UW_SOCM_identity']:
                print(f'Using Identity...')
                self.M = models.Identity(
                    dim=self.dim,
                    output_matrix=self.output_matrix,
                ).to(self.device)
            else:
                print(f'Using SigmoidMLP...')
                self.M = models.SigmoidMLP(
                    dim=self.dim,
                    hdims=self.hdims_M,
                    gamma=self.gamma,
                    scaling_factor=self.scaling_factor_M,
                ).to(self.device)

        # Use learned control in the stochastic_trajectories function
        self.use_learned_control = True


class SOC_Solver(nn.Module):
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(
        self,
        neural_sde,
        x0,
        ut,
        T=1.0,
        num_steps=100,
        lmbd=1.0,
        d=2,
        sigma=torch.eye(2),
        output_matrix=False,
    ):
        super().__init__()
        self.dim = neural_sde.dim
        self.neural_sde = neural_sde
        self.x0 = x0
        self.ut = ut
        self.T = T
        self.ts = torch.linspace(0, T, num_steps + 1).to(x0.device)
        self.num_steps = num_steps
        self.dt = T / num_steps
        self.lmbd = lmbd
        self.d = d
        self.y0 = torch.nn.Parameter(torch.randn(1, device=x0.device))
        self.sigma = sigma
        self.output_matrix = output_matrix

    def control(self, t0, x0):
        x0 = x0.reshape(-1, self.dim)
        t0_expanded = t0.reshape(-1, 1).expand(x0.shape[0], 1)
        tx = torch.cat([t0_expanded, x0], dim=-1)
        nabla_V = self.neural_sde.nabla_V(tx)
        learned_control = -torch.einsum(
            "ij,bj->bi", torch.transpose(self.sigma, 0, 1), nabla_V
        )
        return learned_control

    def control_objective(self, batch_size, total_n_samples=65536):
        n_batches = int(total_n_samples // batch_size)
        effective_n_samples = n_batches * batch_size
        for k in range(n_batches):
            state0 = self.x0.repeat(batch_size, 1)
            (
                states,
                _,
                _,
                _,
                log_path_weight_deterministic,
                _,
                log_terminal_weight,
                _,
                _,
                _,
                _,
            ) = utils.stochastic_trajectories(
                self.neural_sde,
                state0,
                self.ts.to(state0),
                self.lmbd,
            )
            if k == 0:
                ctrl_losses = -self.lmbd * (
                    log_path_weight_deterministic + log_terminal_weight
                )
                trajectory = states
            else:
                ctrl_loss = -self.lmbd * (
                    log_path_weight_deterministic + log_terminal_weight
                )
                ctrl_losses = torch.cat((ctrl_losses, ctrl_loss), 0)
            if k % 32 == 31:
                print(f"Batch {k+1}/{n_batches} done")
        return (
            torch.mean(ctrl_losses),
            torch.std(ctrl_losses) / np.sqrt(effective_n_samples - 1),
            trajectory,
        )

    def loss(
        self,
        batch_size,
        compute_L2_error=False,
        optimal_control=None,
        compute_control_objective=False,
        algorithm="SOCM_const_M",
        add_weights=False,
        total_n_samples=65536,
        verbose=False,
        u_warm_start=None,
        use_warm_start=True,
        use_stopping_time=False,
        efficient_memory=False,
    ):
        if len(self.x0.shape) == 1:
            state0 = self.x0.repeat(batch_size, 1)
        elif len(self.x0.shape) == 2:
            state0 = self.x0
        d = state0.shape[1]
        detach = algorithm != "discrete_adjoint"
        (
            states,
            noises,
            stop_indicators,
            fractional_timesteps,
            log_path_weight_deterministic,
            log_path_weight_stochastic,
            log_terminal_weight,
            controls,
            log_path_weight_deterministic_tensor,
            log_path_weight_f_tensor,
            log_path_weight_stochastic_tensor,
        ) = utils.stochastic_trajectories(
            self.neural_sde,
            state0,
            self.ts.to(state0),
            self.lmbd,
            detach=detach,
        )
        unsqueezed_stop_indicators = stop_indicators.unsqueeze(2)
        weight = torch.exp(
            log_path_weight_deterministic
            + log_path_weight_stochastic
            + log_terminal_weight
        )

        if algorithm == "discrete_adjoint":
            ctrl_losses = -self.lmbd * (
                log_path_weight_deterministic + log_terminal_weight
            )
            objective = torch.mean(ctrl_losses)
            weight = weight.detach()
            learned_control = controls.detach()
        else:
            ts_repeat = self.ts.unsqueeze(1).unsqueeze(2).repeat(1, states.shape[1], 1)
            tx = torch.cat([ts_repeat, states], dim=-1)
            tx_reshape = torch.reshape(tx, (-1, tx.shape[2]))

            # Evaluate nabla_V
            nabla_V = self.neural_sde.nabla_V(tx_reshape)
            nabla_V = torch.reshape(nabla_V, states.shape)

            if u_warm_start and use_warm_start:
                sigma_inverse_transpose = torch.transpose(
                    torch.inverse(self.sigma), 0, 1
                )
                u_warm_start_eval = u_warm_start(self.ts, states).detach()
                nabla_V = nabla_V - torch.einsum(
                    "ij,abj->abi", sigma_inverse_transpose, u_warm_start_eval
                )

        if algorithm == "SOCM_const_M":
            sigma_inverse_transpose = torch.transpose(torch.inverse(self.sigma), 0, 1)
            least_squares_target_integrand_term_1 = (
                self.neural_sde.nabla_f(self.ts[0], states)
            )[:-1, :, :]
            least_squares_target_integrand_term_2 = -np.sqrt(self.lmbd) * torch.einsum(
                "abij,abj->abi",
                self.neural_sde.nabla_b(self.ts[0], states)[:-1, :, :, :],
                torch.einsum("ij,abj->abi", sigma_inverse_transpose, noises),
            )
            least_squares_target_integrand_term_3 = -torch.einsum(
                "abij,abj->abi",
                self.neural_sde.nabla_b(self.ts[0], states)[:-1, :, :, :],
                torch.einsum("ij,abj->abi", sigma_inverse_transpose, controls),
            )
            least_squares_target_terminal = self.neural_sde.nabla_g(states[-1, :, :])

            dts = self.ts[1:] - self.ts[:-1]
            least_squares_target_integrand_term_1_times_dt = torch.cat(
                (
                    torch.zeros_like(
                        least_squares_target_integrand_term_1[0, :, :]
                    ).unsqueeze(0),
                    least_squares_target_integrand_term_1
                    * dts.unsqueeze(1).unsqueeze(2),
                ),
                0,
            )
            least_squares_target_integrand_term_2_times_sqrt_dt = torch.cat(
                (
                    torch.zeros_like(
                        least_squares_target_integrand_term_2[0, :, :]
                    ).unsqueeze(0),
                    least_squares_target_integrand_term_2
                    * torch.sqrt(dts).unsqueeze(1).unsqueeze(2),
                ),
                0,
            )
            least_squares_target_integrand_term_3_times_dt = torch.cat(
                (
                    torch.zeros_like(
                        least_squares_target_integrand_term_3[0, :, :]
                    ).unsqueeze(0),
                    least_squares_target_integrand_term_3
                    * dts.unsqueeze(1).unsqueeze(2),
                ),
                0,
            )

            cumulative_sum_least_squares_term_1 = torch.sum(
                least_squares_target_integrand_term_1_times_dt, dim=0
            ).unsqueeze(0) - torch.cumsum(
                least_squares_target_integrand_term_1_times_dt, dim=0
            )
            cumulative_sum_least_squares_term_2 = torch.sum(
                least_squares_target_integrand_term_2_times_sqrt_dt, dim=0
            ).unsqueeze(0) - torch.cumsum(
                least_squares_target_integrand_term_2_times_sqrt_dt, dim=0
            )
            cumulative_sum_least_squares_term_3 = torch.sum(
                least_squares_target_integrand_term_3_times_dt, dim=0
            ).unsqueeze(0) - torch.cumsum(
                least_squares_target_integrand_term_3_times_dt, dim=0
            )
            least_squares_target = (
                cumulative_sum_least_squares_term_1
                + cumulative_sum_least_squares_term_2
                + cumulative_sum_least_squares_term_3
                + least_squares_target_terminal.unsqueeze(0)
            )
            control_learned = -torch.einsum(
                "ij,...j->...i", torch.transpose(self.sigma, 0, 1), nabla_V
            )
            control_target = -torch.einsum(
                "ij,...j->...i", torch.transpose(self.sigma, 0, 1), least_squares_target
            )

            objective = torch.sum(
                (control_learned - control_target) ** 2
                * weight.unsqueeze(0).unsqueeze(2)
            ) / (states.shape[0] * states.shape[1])

        if algorithm == "SOCM_exp":
            sigma_inverse_transpose = torch.transpose(torch.inverse(self.sigma), 0, 1)
            exp_factor = torch.exp(-self.gamma * self.ts)
            identity = torch.eye(d).to(self.x0.device)
            least_squares_target_integrand_term_1 = (
                exp_factor.unsqueeze(1).unsqueeze(2)
                * self.neural_sde.nabla_f(self.ts[0], states)
            )[:-1, :, :]
            least_squares_target_integrand_term_2 = exp_factor[:-1].unsqueeze(
                1
            ).unsqueeze(2) * (
                -np.sqrt(self.lmbd)
                * torch.einsum(
                    "abij,abj->abi",
                    self.neural_sde.nabla_b(self.ts[0], states)[:-1, :, :, :]
                    + self.gamma * identity,
                    torch.einsum("ij,abj->abi", sigma_inverse_transpose, noises),
                )
            )
            least_squares_target_integrand_term_3 = exp_factor[:-1].unsqueeze(
                1
            ).unsqueeze(2) * (
                -torch.einsum(
                    "abij,abj->abi",
                    self.neural_sde.nabla_b(self.ts[0], states)[:-1, :, :, :]
                    + self.gamma * identity,
                    torch.einsum("ij,abj->abi", sigma_inverse_transpose, controls),
                )
            )
            least_squares_target_terminal = torch.exp(
                -self.gamma * (self.T - self.ts)
            ).unsqueeze(1).unsqueeze(2) * self.neural_sde.nabla_g(
                states[-1, :, :]
            ).unsqueeze(
                0
            )

            dts = self.ts[1:] - self.ts[:-1]
            least_squares_target_integrand_term_1_times_dt = torch.cat(
                (
                    torch.zeros_like(
                        least_squares_target_integrand_term_1[0, :, :]
                    ).unsqueeze(0),
                    least_squares_target_integrand_term_1
                    * dts.unsqueeze(1).unsqueeze(2),
                ),
                0,
            )
            least_squares_target_integrand_term_2_times_sqrt_dt = torch.cat(
                (
                    torch.zeros_like(
                        least_squares_target_integrand_term_2[0, :, :]
                    ).unsqueeze(0),
                    least_squares_target_integrand_term_2
                    * torch.sqrt(dts).unsqueeze(1).unsqueeze(2),
                ),
                0,
            )
            least_squares_target_integrand_term_3_times_dt = torch.cat(
                (
                    torch.zeros_like(
                        least_squares_target_integrand_term_3[0, :, :]
                    ).unsqueeze(0),
                    least_squares_target_integrand_term_3
                    * dts.unsqueeze(1).unsqueeze(2),
                ),
                0,
            )

            inv_exp_factor = 1 / exp_factor
            cumsum_least_squares_term_1 = inv_exp_factor.unsqueeze(1).unsqueeze(2) * (
                torch.sum(
                    least_squares_target_integrand_term_1_times_dt, dim=0
                ).unsqueeze(0)
                - torch.cumsum(least_squares_target_integrand_term_1_times_dt, dim=0)
            )
            cumsum_least_squares_term_2 = inv_exp_factor.unsqueeze(1).unsqueeze(2) * (
                torch.sum(
                    least_squares_target_integrand_term_2_times_sqrt_dt, dim=0
                ).unsqueeze(0)
                - torch.cumsum(
                    least_squares_target_integrand_term_2_times_sqrt_dt, dim=0
                )
            )
            cumsum_least_squares_term_3 = inv_exp_factor.unsqueeze(1).unsqueeze(2) * (
                torch.sum(
                    least_squares_target_integrand_term_3_times_dt, dim=0
                ).unsqueeze(0)
                - torch.cumsum(least_squares_target_integrand_term_3_times_dt, dim=0)
            )

            least_squares_target = (
                cumsum_least_squares_term_1
                + cumsum_least_squares_term_2
                + cumsum_least_squares_term_3
                + least_squares_target_terminal
            )
            control_learned = -torch.einsum(
                "ij,...j->...i", torch.transpose(self.sigma, 0, 1), nabla_V
            )
            control_target = -torch.einsum(
                "ij,...j->...i", torch.transpose(self.sigma, 0, 1), least_squares_target
            )

            objective = torch.sum(
                (control_learned - control_target) ** 2
                * weight.unsqueeze(0).unsqueeze(2)
            ) / (states.shape[0] * states.shape[1])

        if efficient_memory and algorithm in ["SOCM", "UW_SOCM", 
                                              "SOCM_sc", "UW_SOCM_sc", "SOCM_sc_2B", "UW_SOCM_sc_2B",
                                              "SOCM_diag", "UW_SOCM_diag", "SOCM_diag_2B", "UW_SOCM_diag_2B", "UW_SOCM_identity"]:
            if self.output_matrix:
                diagonal_M = False
                scalar_M = False
            else:
                diagonal_M = algorithm in ["SOCM_diag", "UW_SOCM_diag", "SOCM_diag_2B", "UW_SOCM_diag_2B"]
                scalar_M = algorithm in ["SOCM_sc", "UW_SOCM_sc", "SOCM_sc_2B", "UW_SOCM_sc_2B", "UW_SOCM_identity"]
            sigma_inverse_transpose = torch.transpose(torch.inverse(self.sigma), 0, 1)
            dts = self.ts[1:] - self.ts[:-1]
            identity = torch.eye(d).to(self.x0.device)

            sum_M = lambda t, s: self.neural_sde.M(t, s).sum(dim=0)

            if diagonal_M:
                derivative_M_0 = functorch.jacrev(sum_M, argnums=1)
                derivative_M = lambda t, s: torch.transpose(derivative_M_0(t, s), 0, 1)

                M_evals = torch.zeros(len(self.ts), len(self.ts), d).to(
                    self.ts.device
                )
                derivative_M_evals = torch.zeros(len(self.ts), len(self.ts), d).to(
                    self.ts.device
                )

            elif scalar_M:
                derivative_M_0 = functorch.jacrev(sum_M, argnums=1)
                derivative_M = lambda t, s: derivative_M_0(t, s)

                M_evals = torch.zeros(len(self.ts), len(self.ts)).to(
                    self.ts.device
                )
                derivative_M_evals = torch.zeros(len(self.ts), len(self.ts)).to(
                    self.ts.device
                )

            else:
                derivative_M_0 = functorch.jacrev(sum_M, argnums=1)
                derivative_M = lambda t, s: torch.transpose(
                    torch.transpose(derivative_M_0(t, s), 1, 2), 0, 1
                )

                M_evals = torch.zeros(len(self.ts), len(self.ts), d, d).to(
                    self.ts.device
                )
                derivative_M_evals = torch.zeros(len(self.ts), len(self.ts), d, d).to(
                    self.ts.device
                )

            s_vector = []
            t_vector = []
            for k, t in enumerate(self.ts):
                s_vector.append(
                    torch.linspace(t, self.T, self.num_steps + 1 - k).to(self.ts.device)
                )
                t_vector.append(
                    t * torch.ones(self.num_steps + 1 - k).to(self.ts.device)
                )
                if use_stopping_time:
                    stopping_timestep_vector.append(
                        stopping_timestep.unsqueeze(0).repeat(self.num_steps + 1 - k, 1)
                    )
            s_vector = torch.cat(s_vector)
            t_vector = torch.cat(t_vector)

            M_evals_all = self.neural_sde.M(
                t_vector,
                s_vector,
            )
            # derivative_M_evals_all = derivative_M(
            #     t_vector,
            #     s_vector,
            # )
            # print(f'M_evals_all.shape: {M_evals_all.shape}, derivative_M_evals_all.shape: {derivative_M_evals_all.shape}')
            # print(f't_vector.shape: {t_vector.shape}, s_vector.shape: {s_vector.shape}')
            counter = 0
            for k, t in enumerate(self.ts):
                if diagonal_M:
                    M_evals[k, k:, :] = M_evals_all[
                        counter : (counter + self.num_steps + 1 - k), :
                    ]
                    # derivative_M_evals[k, k:, :] = derivative_M_evals_all[
                    #     counter : (counter + self.num_steps + 1 - k), :
                    # ]
                    derivative_M_evals[k, k:-1, :] = (M_evals_all[
                        counter + 1 : (counter + self.num_steps + 1 - k), :
                    ] - M_evals_all[
                        counter : (counter + self.num_steps - k), :
                    ]) / (s_vector[counter + 1 : (counter + self.num_steps + 1 - k)] - s_vector[counter : (counter + self.num_steps - k)])[:,None]
                elif scalar_M:
                    M_evals[k, k:] = M_evals_all[
                        counter : (counter + self.num_steps + 1 - k)
                    ]
                    # derivative_M_evals[k, k:] = derivative_M_evals_all[
                    #     counter : (counter + self.num_steps + 1 - k)
                    # ]
                    derivative_M_evals[k, k:-1] = (M_evals_all[
                        counter + 1 : (counter + self.num_steps + 1 - k)
                    ] - M_evals_all[
                        counter : (counter + self.num_steps - k)
                    ]) / (s_vector[counter + 1 : (counter + self.num_steps + 1 - k)] - s_vector[counter : (counter + self.num_steps - k)])
                else:
                    M_evals[k, k:, :, :] = M_evals_all[
                        counter : (counter + self.num_steps + 1 - k), :, :
                    ]
                    # derivative_M_evals[k, k:, :, :] = derivative_M_evals_all[
                    #     counter : (counter + self.num_steps + 1 - k), :, :
                    # ]
                    # original_quantity = derivative_M_evals_all[
                    #     counter : (counter + self.num_steps - k), :, :
                    # ]
                    derivative_M_evals[k, k:-1, :, :] = (M_evals_all[
                        counter + 1 : (counter + self.num_steps + 1 - k), :, :
                    ] - M_evals_all[
                        counter : (counter + self.num_steps - k), :, :
                    ]) / (s_vector[counter + 1 : (counter + self.num_steps + 1 - k)] - s_vector[counter : (counter + self.num_steps - k)])[:,None,None]
                    # print(f'original_quantity.shape: {original_quantity.shape}, derivative_M_evals[k, k:-1, :, :].shape: {derivative_M_evals[k, k:-1, :, :].shape}')
                    # print(f'torch.norm(original_quantity - derivative_M_evals[k, k:-1, :, :]): {torch.norm(original_quantity - derivative_M_evals[k, k:-1, :, :])}')
                    # print(f'torch.norm(derivative_M_evals[k, k:-1, :, :]): {torch.norm(derivative_M_evals[k, k:-1, :, :])}')
                counter += self.num_steps + 1 - k

            # Compute terms corresponding to state and terminal costs
            if diagonal_M:
                least_squares_target_integrand_term_1 = (M_evals[:, :, None, :]
                                                         * self.neural_sde.nabla_f(self.ts, states)[None, :, :, :])[:, :-1, :, :]
                M_evals_final = M_evals[:, -1, :]
                least_squares_target_terminal = (M_evals_final[:, None, :] * self.neural_sde.nabla_g(states[-1, :, :])[None, :, :])
            elif scalar_M:
                least_squares_target_integrand_term_1 = (M_evals[:, :, None, None]
                                                         * self.neural_sde.nabla_f(self.ts, states)[None, :, :, :])[:, :-1, :, :]
                M_evals_final = M_evals[:, -1]
                least_squares_target_terminal = (M_evals_final[:, None, None] * self.neural_sde.nabla_g(states[-1, :, :])[None, :, :])
            else:
                least_squares_target_integrand_term_1 = torch.einsum(
                    "ijkl,jml->ijmk",
                    M_evals,
                    self.neural_sde.nabla_f(self.ts, states),
                )[:, :-1, :, :]
                M_evals_final = M_evals[:, -1, :, :]
                least_squares_target_terminal = torch.einsum(
                    "ikl,ml->imk",
                    M_evals_final,
                    self.neural_sde.nabla_g(states[-1, :, :]),
                )
            least_squares_target_integrand_term_1_times_dt = (
                least_squares_target_integrand_term_1
                * dts.unsqueeze(1).unsqueeze(2).unsqueeze(0)
            )
            cumsum_least_squares_term_1 = torch.sum(
                least_squares_target_integrand_term_1_times_dt, dim=1
            )

            # Compute the remaining term
            def control_autograd_arg(ts, states, direction_vector):
                output = torch.sum((self.neural_sde.b(self.ts, states) 
                                    )[:-1,:,:] * torch.einsum("ij,abj->abi", sigma_inverse_transpose, direction_vector), dim=2)
                return output

            direction_vector = noises * torch.sqrt(self.lmbd * dts).unsqueeze(1).unsqueeze(2) + controls * dts.unsqueeze(1).unsqueeze(2)
            # Check if states requires grad
            if not states.requires_grad:
                states.requires_grad = True
                # print(f'states did not require grad but now does')
            nabla_control_noise = torch.autograd.grad(control_autograd_arg(self.ts, states, direction_vector).sum(), states)[0]
            states.requires_grad = False

            if diagonal_M:
                least_squares_target_integrand_term_2 = -(
                    M_evals[:,:-1,None,:] * nabla_control_noise[None,:-1,:,:]
                )
                least_squares_target_integrand_term_3 = (
                    derivative_M_evals[:,:-1,None,:]
                    * torch.einsum("ij,abj->abi", sigma_inverse_transpose, direction_vector)[None,:,:,:]
                )
            elif scalar_M:
                least_squares_target_integrand_term_2 = -(
                    M_evals[:,:-1,None,None] * nabla_control_noise[None,:-1,:,:]
                )
                least_squares_target_integrand_term_3 = (
                    derivative_M_evals[:,:-1,None,None]
                    * torch.einsum("ij,abj->abi", sigma_inverse_transpose, direction_vector)[None,:,:,:]
                )
            else:
                least_squares_target_integrand_term_2 = -torch.einsum(
                    "ijkl,jml->ijmk",
                    M_evals[:,:-1,:,:],
                    nabla_control_noise[:-1,:,:],
                )
                least_squares_target_integrand_term_3 = torch.einsum(
                    "ijkl,jml->ijmk",
                    derivative_M_evals[:,:-1,:,:],
                    torch.einsum("ij,abj->abi", sigma_inverse_transpose, direction_vector)
                )

            least_squares_target_integrand_term_2_3_times_sqrt_dt = (
                least_squares_target_integrand_term_2
                + least_squares_target_integrand_term_3
            )

            cumsum_least_squares_term_2_3 = torch.sum(
                least_squares_target_integrand_term_2_3_times_sqrt_dt, dim=1
            )

            # Compute matching vector field
            least_squares_target = (
                cumsum_least_squares_term_1
                + cumsum_least_squares_term_2_3
                + least_squares_target_terminal
            )

            control_learned = -torch.einsum(
                "ij,...j->...i", torch.transpose(self.sigma, 0, 1), nabla_V
            )
            control_target = -torch.einsum(
                "ij,...j->...i",
                torch.transpose(self.sigma, 0, 1),
                least_squares_target,
            )

            if algorithm in ["UW_SOCM", "UW_SOCM_sc", "UW_SOCM_sc_2B", "UW_SOCM_diag", "UW_SOCM_diag_2B", "UW_SOCM_identity"]:
                objective = torch.sum(
                    (control_learned - control_target) ** 2
                ) / (states.shape[0] * states.shape[1])
            else:
                objective = torch.sum(
                    (control_learned - control_target) ** 2
                    * weight.unsqueeze(0).unsqueeze(2)
                ) / (states.shape[0] * states.shape[1])

        if not efficient_memory and algorithm in ["SOCM", "UW_SOCM", 
                                                  "SOCM_sc", "UW_SOCM_sc", "SOCM_sc_2B", "UW_SOCM_sc_2B",
                                                  "SOCM_diag", "UW_SOCM_diag", "SOCM_diag_2B", "UW_SOCM_diag_2B", "UW_SOCM_identity", "UW_SOCM_no_v", "UW_SOCM_no_nabla_b_term"]:
            if self.output_matrix:
                diagonal_M = False
                scalar_M = False
            else:
                diagonal_M = algorithm in ["SOCM_diag", "UW_SOCM_diag", "SOCM_diag_2B", "UW_SOCM_diag_2B"]
                scalar_M = algorithm in ["SOCM_sc", "UW_SOCM_sc", "SOCM_sc_2B", "UW_SOCM_sc_2B", "UW_SOCM_identity"]
            sigma_inverse_transpose = torch.transpose(torch.inverse(self.sigma), 0, 1)
            identity = torch.eye(d).to(self.x0.device)

            if use_stopping_time:
                sum_M = lambda t, s, stopping_timestep_values: self.neural_sde.M(
                    t, s, stopping_timestep_values
                ).sum(dim=0)

                derivative_M_0 = functorch.jacrev(sum_M, argnums=1)
                derivative_M = lambda t, s, stopping_timestep_values: torch.transpose(
                    torch.transpose(
                        torch.transpose(
                            derivative_M_0(t, s, stopping_timestep_values), 2, 3
                        ),
                        1,
                        2,
                    ),
                    0,
                    1,
                )

                M_evals = torch.zeros(len(self.ts), len(self.ts), batch_size, d, d).to(
                    self.ts.device
                )
                derivative_M_evals = torch.zeros(
                    len(self.ts), len(self.ts), batch_size, d, d
                ).to(self.ts.device)

            else:
                sum_M = lambda t, s: self.neural_sde.M(t, s).sum(dim=0)

                if diagonal_M:
                    derivative_M_0 = functorch.jacrev(sum_M, argnums=1)
                    derivative_M = lambda t, s: torch.transpose(derivative_M_0(t, s), 0, 1)

                    M_evals = torch.zeros(len(self.ts), len(self.ts), d).to(
                        self.ts.device
                    )
                    derivative_M_evals = torch.zeros(len(self.ts), len(self.ts), d).to(
                        self.ts.device
                    )

                elif scalar_M:
                    derivative_M_0 = functorch.jacrev(sum_M, argnums=1)
                    derivative_M = lambda t, s: derivative_M_0(t, s)

                    M_evals = torch.zeros(len(self.ts), len(self.ts)).to(
                        self.ts.device
                    )
                    derivative_M_evals = torch.zeros(len(self.ts), len(self.ts)).to(
                        self.ts.device
                    )

                else:
                    derivative_M_0 = functorch.jacrev(sum_M, argnums=1)
                    derivative_M = lambda t, s: torch.transpose(
                        torch.transpose(derivative_M_0(t, s), 1, 2), 0, 1
                    )

                    M_evals = torch.zeros(len(self.ts), len(self.ts), d, d).to(
                        self.ts.device
                    )
                    derivative_M_evals = torch.zeros(len(self.ts), len(self.ts), d, d).to(
                        self.ts.device
                    )

            if use_stopping_time:
                stopping_function_output_int = (self.neural_sde.Phi(states) > 0).to(
                    torch.int
                )
                stopping_timestep = (
                    torch.sum(stopping_function_output_int, dim=0) - 1
                ) / (len(self.ts) - 1)
                stopping_timestep_vector = []

            s_vector = []
            t_vector = []
            for k, t in enumerate(self.ts):
                s_vector.append(
                    torch.linspace(t, self.T, self.num_steps + 1 - k).to(self.ts.device)
                )
                t_vector.append(
                    t * torch.ones(self.num_steps + 1 - k).to(self.ts.device)
                )
                if use_stopping_time:
                    stopping_timestep_vector.append(
                        stopping_timestep.unsqueeze(0).repeat(self.num_steps + 1 - k, 1)
                    )
            s_vector = torch.cat(s_vector)
            t_vector = torch.cat(t_vector)
            if use_stopping_time:
                stopping_timestep_vector = torch.cat(stopping_timestep_vector, dim=0)
                M_evals_all = self.neural_sde.M(
                    t_vector, s_vector, stopping_timestep_vector
                )
                derivative_M_evals_all = torch.nan_to_num(
                    derivative_M(t_vector, s_vector, stopping_timestep_vector)
                )
                counter = 0
                for k, t in enumerate(self.ts):
                    M_evals[k, k:, :, :, :] = M_evals_all[
                        counter : (counter + self.num_steps + 1 - k), :, :, :
                    ]
                    derivative_M_evals[k, k:, :, :, :] = derivative_M_evals_all[
                        counter : (counter + self.num_steps + 1 - k), :, :, :
                    ]
                    counter += self.num_steps + 1 - k
            else:
                M_evals_all = self.neural_sde.M(
                    t_vector,
                    s_vector,
                )
                derivative_M_evals_all = derivative_M(
                    t_vector,
                    s_vector,
                )
                counter = 0
                for k, t in enumerate(self.ts):
                    if diagonal_M:
                        M_evals[k, k:, :] = M_evals_all[
                            counter : (counter + self.num_steps + 1 - k), :
                        ]
                        derivative_M_evals[k, k:, :] = derivative_M_evals_all[
                            counter : (counter + self.num_steps + 1 - k), :
                        ]
                    elif scalar_M:
                        M_evals[k, k:] = M_evals_all[
                            counter : (counter + self.num_steps + 1 - k)
                        ]
                        derivative_M_evals[k, k:] = derivative_M_evals_all[
                            counter : (counter + self.num_steps + 1 - k)
                        ]
                    else:
                        M_evals[k, k:, :, :] = M_evals_all[
                            counter : (counter + self.num_steps + 1 - k), :, :
                        ]
                        derivative_M_evals[k, k:, :, :] = derivative_M_evals_all[
                            counter : (counter + self.num_steps + 1 - k), :, :
                        ]
                    counter += self.num_steps + 1 - k

            if use_stopping_time:
                least_squares_target_integrand_term_1 = torch.einsum(
                    "ijmkl,jml->ijmk",
                    M_evals,
                    self.neural_sde.nabla_f(self.ts, states),
                )[:, :-1, :, :]
            else:
                if diagonal_M:
                    least_squares_target_integrand_term_1 = (M_evals[:, :, None, :]
                                                             * self.neural_sde.nabla_f(self.ts, states)[None, :, :, :])[:, :-1, :, :]
                elif scalar_M:
                    least_squares_target_integrand_term_1 = (M_evals[:, :, None, None]
                                                             * self.neural_sde.nabla_f(self.ts, states)[None, :, :, :])[:, :-1, :, :]
                else:
                    least_squares_target_integrand_term_1 = torch.einsum(
                        "ijkl,jml->ijmk",
                        M_evals,
                        self.neural_sde.nabla_f(self.ts, states),
                    )[:, :-1, :, :]

            if use_stopping_time:
                M_nabla_b_term = (
                    torch.einsum(
                        "ijmkl,jmln->ijmkn",
                        M_evals,
                        self.neural_sde.nabla_b(self.ts, states),
                    )
                    - derivative_M_evals
                )
                least_squares_target_integrand_term_2 = -np.sqrt(
                    self.lmbd
                ) * torch.einsum(
                    "ijmkn,jmn->ijmk",
                    M_nabla_b_term[:, :-1, :, :, :],
                    torch.einsum("ij,abj->abi", sigma_inverse_transpose, noises),
                )
            else:
                if diagonal_M:
                    M_nabla_b_term = (M_evals[:, :, None, :, None] * self.neural_sde.nabla_b(self.ts, states)[None, :, :, :, :] 
                                      - derivative_M_evals[:, :, None, :, None])
                elif scalar_M:
                    M_nabla_b_term = (M_evals[:, :, None, None, None] * self.neural_sde.nabla_b(self.ts, states)[None, :, :, :, :] 
                                      - derivative_M_evals[:, :, None, None, None])
                else:
                    M_nabla_b_term = torch.einsum(
                        "ijkl,jmln->ijmkn",
                        M_evals,
                        self.neural_sde.nabla_b(self.ts, states),
                    ) - derivative_M_evals.unsqueeze(2)
                least_squares_target_integrand_term_2 = -np.sqrt(
                    self.lmbd
                ) * torch.einsum(
                    "ijmkn,jmn->ijmk",
                    M_nabla_b_term[:, :-1, :, :, :],
                    torch.einsum("ij,abj->abi", sigma_inverse_transpose, noises),
                )
            if algorithm == "UW_SOCM_no_nabla_b_term":
                least_squares_target_integrand_term_2 = torch.zeros_like(least_squares_target_integrand_term_2)

            least_squares_target_integrand_term_3 = -torch.einsum(
                "ijmkn,jmn->ijmk",
                M_nabla_b_term[:, :-1, :, :, :],
                torch.einsum("ij,abj->abi", sigma_inverse_transpose, controls),
            )
            if algorithm == "UW_SOCM_no_v" or algorithm == "UW_SOCM_no_nabla_b_term":
                least_squares_target_integrand_term_3 = torch.zeros_like(least_squares_target_integrand_term_3)

            if use_stopping_time:
                M_evals_final = M_evals[:, -1, :, :, :]
                least_squares_target_terminal = torch.einsum(
                    "imkl,ml->imk",
                    M_evals_final,
                    self.neural_sde.nabla_g(states[-1, :, :]),
                )
            else:
                if diagonal_M:
                    M_evals_final = M_evals[:, -1, :]
                    least_squares_target_terminal = (M_evals_final[:, None, :] * self.neural_sde.nabla_g(states[-1, :, :])[None, :, :])
                elif scalar_M:
                    M_evals_final = M_evals[:, -1]
                    least_squares_target_terminal = (M_evals_final[:, None, None] * self.neural_sde.nabla_g(states[-1, :, :])[None, :, :])
                else:
                    M_evals_final = M_evals[:, -1, :, :]
                    least_squares_target_terminal = torch.einsum(
                        "ikl,ml->imk",
                        M_evals_final,
                        self.neural_sde.nabla_g(states[-1, :, :]),
                    )

            if use_stopping_time:
                least_squares_target_integrand_term_1_times_dt = (
                    least_squares_target_integrand_term_1
                    * fractional_timesteps.unsqueeze(0).unsqueeze(3)
                )
                least_squares_target_integrand_term_2_times_sqrt_dt = (
                    least_squares_target_integrand_term_2
                    * torch.sqrt(fractional_timesteps).unsqueeze(0).unsqueeze(3)
                )
                least_squares_target_integrand_term_3_times_dt = (
                    least_squares_target_integrand_term_3
                    * fractional_timesteps.unsqueeze(0).unsqueeze(3)
                )
            else:
                dts = self.ts[1:] - self.ts[:-1]
                least_squares_target_integrand_term_1_times_dt = (
                    least_squares_target_integrand_term_1
                    * dts.unsqueeze(1).unsqueeze(2).unsqueeze(0)
                )
                least_squares_target_integrand_term_2_times_sqrt_dt = (
                    least_squares_target_integrand_term_2
                    * torch.sqrt(dts).unsqueeze(1).unsqueeze(2)
                )
                least_squares_target_integrand_term_3_times_dt = (
                    least_squares_target_integrand_term_3 * dts.unsqueeze(1).unsqueeze(2)
                )

            cumsum_least_squares_term_1 = torch.sum(
                least_squares_target_integrand_term_1_times_dt, dim=1
            )
            cumsum_least_squares_term_2 = torch.sum(
                least_squares_target_integrand_term_2_times_sqrt_dt, dim=1
            )
            cumsum_least_squares_term_3 = torch.sum(
                least_squares_target_integrand_term_3_times_dt, dim=1
            )

            least_squares_target = (
                cumsum_least_squares_term_1
                + cumsum_least_squares_term_2
                + cumsum_least_squares_term_3
                + least_squares_target_terminal
            )

            if use_stopping_time:
                control_learned = -unsqueezed_stop_indicators * torch.einsum(
                    "ij,...j->...i", torch.transpose(self.sigma, 0, 1), nabla_V
                )
                control_target = -unsqueezed_stop_indicators * torch.einsum(
                    "ij,...j->...i",
                    torch.transpose(self.sigma, 0, 1),
                    least_squares_target,
                )
            else:
                control_learned = -torch.einsum(
                    "ij,...j->...i", torch.transpose(self.sigma, 0, 1), nabla_V
                )
                control_target = -torch.einsum(
                    "ij,...j->...i",
                    torch.transpose(self.sigma, 0, 1),
                    least_squares_target,
                )

            if use_stopping_time:
                objective = torch.sum(
                    (control_learned - control_target) ** 2
                    * weight.unsqueeze(0).unsqueeze(2)
                ) / (torch.sum(stop_indicators))
            else:
                if algorithm in ["UW_SOCM", "UW_SOCM_sc", "UW_SOCM_sc_2B", "UW_SOCM_diag", "UW_SOCM_diag_2B", "UW_SOCM_identity", "UW_SOCM_no_v", "UW_SOCM_no_nabla_b_term"]:
                    objective = torch.sum(
                        (control_learned - control_target) ** 2
                    ) / (states.shape[0] * states.shape[1])
                else:
                    objective = torch.sum(
                        (control_learned - control_target) ** 2
                        * weight.unsqueeze(0).unsqueeze(2)
                    ) / (states.shape[0] * states.shape[1])

        if algorithm == "SOCM_adjoint" or algorithm == "continuous_adjoint":
            nabla_f_evals = self.neural_sde.nabla_f(self.ts, states)
            nabla_b_evals = self.neural_sde.nabla_b(self.ts, states)
            nabla_g_evals = self.neural_sde.nabla_g(states[-1, :, :])

            a_vectors = torch.zeros_like(states)
            a = nabla_g_evals
            a_vectors[-1, :, :] = a

            for k in range(1,len(self.ts)):
                a += self.dt * ((nabla_f_evals[-1-k, :, :] + nabla_f_evals[-k, :, :]) / 2 + torch.einsum("mkl,ml->mk", (nabla_b_evals[-1-k, :, :, :] + nabla_b_evals[-k, :, :, :]) / 2, a))
                a_vectors[-1-k, :, :] = a

            control_learned = -torch.einsum(
                    "ij,...j->...i", torch.transpose(self.sigma, 0, 1), nabla_V
                )
            control_target = -torch.einsum(
                "ij,...j->...i",
                torch.transpose(self.sigma, 0, 1),
                a_vectors,
            )
            if algorithm == "SOCM_adjoint":
                objective = torch.sum(
                    (control_learned - control_target) ** 2
                    * weight.unsqueeze(0).unsqueeze(2)
                ) / (states.shape[0] * states.shape[1])
            else:
                objective = torch.sum(
                    (control_learned - control_target) ** 2
                ) / (states.shape[0] * states.shape[1])

        elif algorithm == "cross_entropy":
            learned_controls = -torch.einsum(
                "ij,abj->abi", torch.transpose(self.sigma, 0, 1), nabla_V
            )
            integrand_term_1 = -(1 / self.lmbd) * torch.sum(
                learned_controls[:-1, :, :] * controls, dim=2
            )
            integrand_term_2 = (1 / (2 * self.lmbd)) * torch.sum(
                learned_controls**2, dim=2
            )[:-1, :]
            deterministic_integrand = integrand_term_1 + integrand_term_2
            stochastic_integrand = -np.sqrt(1 / self.lmbd) * torch.sum(
                learned_controls[:-1, :, :] * noises, dim=2
            )

            if use_stopping_time:
                deterministic_integrand_times_dt = (
                    deterministic_integrand * fractional_timesteps
                )
                stochastic_integrand_times_sqrt_dt = stochastic_integrand * torch.sqrt(
                    fractional_timesteps
                )
            else:
                dts = self.ts[1:] - self.ts[:-1]
                deterministic_integrand_times_dt = (
                    deterministic_integrand * dts.unsqueeze(1)
                )
                stochastic_integrand_times_sqrt_dt = stochastic_integrand * torch.sqrt(
                    dts
                ).unsqueeze(1)

            deterministic_term = torch.sum(deterministic_integrand_times_dt, dim=0)
            stochastic_term = torch.sum(stochastic_integrand_times_sqrt_dt, dim=0)

            objective = torch.mean((deterministic_term + stochastic_term) * weight)

        elif algorithm == "reinf_unadj":
            reward = -self.lmbd * (
                log_path_weight_deterministic + log_terminal_weight
            ).detach()
            control_learned = -torch.einsum(
                "ij,...j->...i", torch.transpose(self.sigma, 0, 1), nabla_V
            )
            dts = self.ts[1:] - self.ts[:-1]
            stochastic_term = torch.sum(control_learned[:-1,:,:] * noises * torch.sqrt(dts)[:,None,None], (0, 2))
            objective = torch.mean(reward * stochastic_term)
            weight = weight.detach()

        elif algorithm == "reinf":
            reward = -np.sqrt(self.lmbd) * (
                log_path_weight_deterministic + log_terminal_weight
            ).detach()
            control_learned = -torch.einsum(
                "ij,...j->...i", torch.transpose(self.sigma, 0, 1), nabla_V
            )
            dts = self.ts[1:] - self.ts[:-1]
            stochastic_term = torch.sum(control_learned[:-1,:,:] * noises * torch.sqrt(dts)[:,None,None], (0, 2))
            control_term = 0.5 * torch.sum(control_learned[:-1,:,:] ** 2 * dts[:,None,None], (0, 2))
            objective = torch.mean(reward * stochastic_term + control_term)
            weight = weight.detach()

        elif algorithm == "reinf_fr":
            reward = -np.sqrt(self.lmbd) * (
                log_path_weight_deterministic_tensor[-1,:].unsqueeze(0) - log_path_weight_deterministic_tensor[:-1,:]
                + log_terminal_weight.unsqueeze(0)
            ).detach()
            control_learned = -torch.einsum(
                "ij,...j->...i", torch.transpose(self.sigma, 0, 1), nabla_V
            )
            dts = self.ts[1:] - self.ts[:-1]
            term_1 = torch.sum(reward.unsqueeze(2) * control_learned[:-1,:,:] * noises * torch.sqrt(dts)[:,None,None], (0, 2))
            control_term = 0.5 * torch.sum(control_learned[:-1,:,:] ** 2 * dts[:,None,None], (0, 2))
            objective = torch.mean(term_1 + control_term)
            weight = weight.detach()

        elif algorithm in ["SOCM_cost","SOCM_cost_sc","SOCM_cost_sc_2B","SOCM_cost_diag","SOCM_cost_diag_2B","SOCM_cost_identity","SOCM_cost_identity_2B"]: # or algorithm == "reinf_PWRT":
            sigma_inverse_transpose = torch.transpose(torch.inverse(self.sigma), 0, 1)
            identity = torch.eye(d).to(self.x0.device)
            dts = self.ts[1:] - self.ts[:-1]

            sum_M = lambda t, s: self.neural_sde.M(t, s).sum(dim=0)

            derivative_M_0 = functorch.jacrev(sum_M, argnums=1)
            derivative_M = lambda t, s: torch.transpose(
                torch.transpose(derivative_M_0(t, s), 1, 2), 0, 1
            )

            M_evals = torch.zeros(len(self.ts), len(self.ts), d, d).to(
                self.ts.device
            )
            derivative_M_evals = torch.zeros(len(self.ts), len(self.ts), d, d).to(
                self.ts.device
            )

            s_vector = []
            t_vector = []
            for k, t in enumerate(self.ts):
                s_vector.append(
                    torch.linspace(t, self.T, self.num_steps + 1 - k).to(self.ts.device)
                )
                t_vector.append(
                    t * torch.ones(self.num_steps + 1 - k).to(self.ts.device)
                )
                if use_stopping_time:
                    stopping_timestep_vector.append(
                        stopping_timestep.unsqueeze(0).repeat(self.num_steps + 1 - k, 1)
                    )
            s_vector = torch.cat(s_vector)
            t_vector = torch.cat(t_vector)

            M_evals_all = self.neural_sde.M(
                t_vector,
                s_vector,
            )
            derivative_M_evals_all = derivative_M(
                t_vector,
                s_vector,
            )
            counter = 0
            for k, t in enumerate(self.ts):
                M_evals[k, k:, :, :] = M_evals_all[
                    counter : (counter + self.num_steps + 1 - k), :, :
                ]
                derivative_M_evals[k, k:, :, :] = derivative_M_evals_all[
                    counter : (counter + self.num_steps + 1 - k), :, :
                ]
                counter += self.num_steps + 1 - k

            if algorithm == 'SOCM_cost_identity_2B':
                M_evals[:, -1, :, :] = 0

            nabla_norm_control = utils.grad_norm_control(self.ts, states, self.neural_sde, self.sigma)

            least_squares_target_integrand_term_1 = torch.einsum(
                "ijkl,jml->ijmk",
                M_evals,
                self.neural_sde.nabla_f(self.ts, states) + nabla_norm_control,
            )[:, :-1, :, :]

            least_squares_target_integrand_term_1_times_dt = (
                least_squares_target_integrand_term_1
                * dts.unsqueeze(1).unsqueeze(2).unsqueeze(0)
            )

            cumsum_least_squares_term_1 = torch.sum(
                least_squares_target_integrand_term_1_times_dt, dim=1
            )

            M_evals_final = M_evals[:, -1, :, :]
            least_squares_target_terminal = torch.einsum(
                "ikl,ml->imk",
                M_evals_final,
                self.neural_sde.nabla_g(states[-1, :, :]),
            )

            def control_autograd_arg(ts, states):
                ts_repeat = ts.unsqueeze(1).unsqueeze(2).repeat(1, states.shape[1], 1)
                tx = torch.cat([ts_repeat, states], dim=-1)
                tx_reshape = torch.reshape(tx, (-1, tx.shape[2]))

                # Evaluate nabla_V
                nabla_V_autograd = self.neural_sde.nabla_V(tx_reshape)
                nabla_V_autograd = torch.reshape(nabla_V_autograd, states.shape)

                learned_control = -torch.einsum(
                    "ij,...j->...i", torch.transpose(self.sigma, 0, 1), nabla_V_autograd
                )
                sigma_learned_control = torch.einsum(
                    "ij,...j->...i", self.sigma, learned_control
                )
                
                output = torch.sum((self.neural_sde.b(self.ts, states) + sigma_learned_control)[:-1,:,:] * torch.einsum("ij,abj->abi", sigma_inverse_transpose, noises), dim=2)
                return output

            nabla_control_noise = torch.autograd.grad(control_autograd_arg(self.ts, states).sum(), states)[0]

            least_squares_target_integrand_term_2 = -torch.einsum(
                "ijkl,jml->ijmk",
                M_evals[:,:-1,:,:],
                nabla_control_noise[:-1,:,:],
            )

            least_squares_target_integrand_term_3 = torch.einsum(
                "ijkl,jml->ijmk",
                derivative_M_evals[:,:-1,:,:],
                torch.einsum("ij,abj->abi", sigma_inverse_transpose, noises)
            )

            least_squares_target_integrand_term_2_3_times_sqrt_dt = (
                (least_squares_target_integrand_term_2
                 + least_squares_target_integrand_term_3)
                * torch.sqrt(dts).unsqueeze(1).unsqueeze(2)
            )

            cumsum_least_squares_term_2_3 = torch.sum(
                least_squares_target_integrand_term_2_3_times_sqrt_dt, dim=1
            )

            reward = - np.sqrt(self.lmbd) * (
                log_path_weight_deterministic_tensor[-1,:].unsqueeze(0) - log_path_weight_deterministic_tensor
                + log_terminal_weight.unsqueeze(0)
            ).detach()

            reward_times_M_term = reward.unsqueeze(2) * cumsum_least_squares_term_2_3 

            cum_terms = cumsum_least_squares_term_1 + least_squares_target_terminal - reward_times_M_term

            control_learned = -torch.einsum(
                "ij,...j->...i", torch.transpose(self.sigma, 0, 1), nabla_V
            )
            control_target = -torch.einsum(
                "ij,...j->...i",
                torch.transpose(self.sigma, 0, 1),
                cum_terms.to(self.sigma.dtype),
            )

            objective = torch.sum(
                (control_learned - control_target) ** 2
            ) / (states.shape[0] * states.shape[1])

            #### TO DEBUG ######
            objective_per_time = torch.sum(
                (control_learned - control_target) ** 2, dim=[1,2]
            ) / (states.shape[0] * states.shape[1])
            print(f'objective_per_time: {objective_per_time}')
            #### TO DEBUG ######

        elif algorithm in ["SOCM_work","SOCM_work_sc","SOCM_work_sc_2B","SOCM_work_diag","SOCM_work_diag_2B","SOCM_work_identity","SOCM_work_identity_2B"]:
            sigma_inverse_transpose = torch.transpose(torch.inverse(self.sigma), 0, 1)
            identity = torch.eye(d).to(self.x0.device)
            dts = self.ts[1:] - self.ts[:-1]

            sum_M = lambda t, s: self.neural_sde.M(t, s).sum(dim=0)

            derivative_M_0 = functorch.jacrev(sum_M, argnums=1)
            derivative_M = lambda t, s: torch.transpose(
                torch.transpose(derivative_M_0(t, s), 1, 2), 0, 1
            )

            M_evals = torch.zeros(len(self.ts), len(self.ts), d, d).to(
                self.ts.device
            )
            derivative_M_evals = torch.zeros(len(self.ts), len(self.ts), d, d).to(
                self.ts.device
            )

            s_vector = []
            t_vector = []
            for k, t in enumerate(self.ts):
                s_vector.append(
                    torch.linspace(t, self.T, self.num_steps + 1 - k).to(self.ts.device)
                )
                t_vector.append(
                    t * torch.ones(self.num_steps + 1 - k).to(self.ts.device)
                )
                if use_stopping_time:
                    stopping_timestep_vector.append(
                        stopping_timestep.unsqueeze(0).repeat(self.num_steps + 1 - k, 1)
                    )
            s_vector = torch.cat(s_vector)
            t_vector = torch.cat(t_vector)

            M_evals_all = self.neural_sde.M(
                t_vector,
                s_vector,
            )
            derivative_M_evals_all = derivative_M(
                t_vector,
                s_vector,
            )
            counter = 0
            for k, t in enumerate(self.ts):
                M_evals[k, k:, :, :] = M_evals_all[
                    counter : (counter + self.num_steps + 1 - k), :, :
                ]
                derivative_M_evals[k, k:, :, :] = derivative_M_evals_all[
                    counter : (counter + self.num_steps + 1 - k), :, :
                ]
                counter += self.num_steps + 1 - k

            if algorithm == 'SOCM_work_identity_2B':
                M_evals[:, -1, :, :] = 0

            # nabla_norm_control = utils.grad_norm_control(self.ts, states, self.neural_sde, self.sigma)

            least_squares_target_integrand_term_1 = torch.einsum(
                "ijkl,jml->ijmk",
                M_evals,
                self.neural_sde.nabla_f(self.ts, states), #+ nabla_norm_control,
            )[:, :-1, :, :]

            least_squares_target_integrand_term_1_times_dt = (
                least_squares_target_integrand_term_1
                * dts.unsqueeze(1).unsqueeze(2).unsqueeze(0)
            )

            cumsum_least_squares_term_1 = torch.sum(
                least_squares_target_integrand_term_1_times_dt, dim=1
            )

            M_evals_final = M_evals[:, -1, :, :]
            least_squares_target_terminal = torch.einsum(
                "ikl,ml->imk",
                M_evals_final,
                self.neural_sde.nabla_g(states[-1, :, :]),
            )

            def control_autograd_arg(ts, states):
                output = torch.sum((self.neural_sde.b(self.ts, states) 
                                    )[:-1,:,:] * torch.einsum("ij,abj->abi", sigma_inverse_transpose, noises), dim=2)
                return output

            nabla_control_noise = torch.autograd.grad(control_autograd_arg(self.ts, states).sum(), states)[0]

            least_squares_target_integrand_term_2 = -torch.einsum(
                "ijkl,jml->ijmk",
                M_evals[:,:-1,:,:],
                nabla_control_noise[:-1,:,:],
            )

            least_squares_target_integrand_term_3 = torch.einsum(
                "ijkl,jml->ijmk",
                derivative_M_evals[:,:-1,:,:],
                torch.einsum("ij,abj->abi", sigma_inverse_transpose, noises)
            )

            least_squares_target_integrand_term_2_3_times_sqrt_dt = (
                (least_squares_target_integrand_term_2
                 + least_squares_target_integrand_term_3)
                * torch.sqrt(dts).unsqueeze(1).unsqueeze(2)
            )

            cumsum_least_squares_term_2_3 = torch.sum(
                least_squares_target_integrand_term_2_3_times_sqrt_dt, dim=1
            )

            reward = - np.sqrt(self.lmbd) * (
                log_path_weight_f_tensor[-1,:].unsqueeze(0) - log_path_weight_f_tensor
                + log_terminal_weight.unsqueeze(0)
            ).detach()

            reward_times_M_term = reward.unsqueeze(2) * cumsum_least_squares_term_2_3 

            cum_terms = cumsum_least_squares_term_1 + least_squares_target_terminal - reward_times_M_term

            control_learned = -torch.einsum(
                "ij,...j->...i", torch.transpose(self.sigma, 0, 1), nabla_V
            )
            control_target = -torch.einsum(
                "ij,...j->...i",
                torch.transpose(self.sigma, 0, 1),
                cum_terms.to(self.sigma.dtype),
            )

            objective = torch.sum(
                (control_learned - control_target) ** 2
            ) / (states.shape[0] * states.shape[1])

        elif (
            algorithm == "variance"
            or algorithm == "log-variance"
            or algorithm == "moment"
        ):
            learned_controls = -torch.einsum(
                "ij,abj->abi", torch.transpose(self.sigma, 0, 1), nabla_V
            )
            integrand_term_1 = -(1 / self.lmbd) * torch.sum(
                learned_controls[:-1, :, :] * controls, dim=2
            )
            integrand_term_2 = (1 / (2 * self.lmbd)) * torch.sum(
                learned_controls**2, dim=2
            )[:-1, :]
            integrand_term_3 = (
                -(1 / self.lmbd) * self.neural_sde.f(self.ts[0], states)[:-1, :]
            )
            deterministic_integrand = (
                integrand_term_1 + integrand_term_2 + integrand_term_3
            )
            stochastic_integrand = -np.sqrt(1 / self.lmbd) * torch.sum(
                learned_controls[:-1, :, :] * noises, dim=2
            )
            if use_stopping_time:
                deterministic_integrand = (
                    deterministic_integrand * stop_indicators[:-1, :]
                )
                stochastic_integrand = stochastic_integrand * stop_indicators[:-1, :]

            if use_stopping_time:
                deterministic_integrand_times_dt = (
                    deterministic_integrand * fractional_timesteps
                )
                stochastic_integrand_times_sqrt_dt = stochastic_integrand * torch.sqrt(
                    fractional_timesteps
                )
            else:
                dts = self.ts[1:] - self.ts[:-1]
                deterministic_integrand_times_dt = (
                    deterministic_integrand * dts.unsqueeze(1)
                )
                stochastic_integrand_times_sqrt_dt = stochastic_integrand * torch.sqrt(
                    dts
                ).unsqueeze(1)

            deterministic_term = torch.sum(deterministic_integrand_times_dt, dim=0)
            stochastic_term = torch.sum(stochastic_integrand_times_sqrt_dt, dim=0)
            g_term = -(1 / self.lmbd) * self.neural_sde.g(states[-1, :, :])
            if algorithm == "log-variance":
                sum_terms = deterministic_term + stochastic_term + g_term
            elif algorithm == "variance":
                sum_terms = torch.exp(deterministic_term + stochastic_term + g_term + 120)
            elif algorithm == "moment":
                sum_terms = deterministic_term + stochastic_term + g_term + self.y0

            if add_weights:
                weight_2 = weight
            else:
                weight_2 = torch.ones_like(weight)
            if algorithm == "log-variance" or algorithm == "variance":
                objective = (
                    len(sum_terms)
                    / (len(sum_terms) - 1)
                    * (
                        torch.mean(sum_terms**2 * weight_2)
                        - torch.mean(sum_terms * weight_2) ** 2
                    )
                )
            elif algorithm == "moment":
                objective = torch.mean(sum_terms**2 * weight_2)

        if compute_L2_error:
            if algorithm == "discrete_adjoint":
                target_control = optimal_control(self.ts, states, t_is_tensor=True)[
                    :-1, :, :
                ].detach()
            else:
                target_control = optimal_control(self.ts, states, t_is_tensor=True)
            if algorithm != "discrete_adjoint":
                learned_control = -torch.einsum(
                    "ij,abj->abi", torch.transpose(self.sigma, 0, 1), nabla_V
                )
            norm_sqd_diff = torch.sum(
                (target_control - learned_control) ** 2
                * weight.unsqueeze(0).unsqueeze(2)
                / (target_control.shape[0] * target_control.shape[1])
            )
        else:
            norm_sqd_diff = None

        if compute_control_objective:
            ctrl_loss_mean, ctrl_loss_std_err, trajectory = self.control_objective(
                batch_size, total_n_samples=total_n_samples
            )
        else:
            ctrl_loss_mean = None
            ctrl_loss_std_err = None
            trajectory = None

        if verbose:
            # To print amount of memory used in GPU
            nvidia_smi.nvmlInit()
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
            # card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            print("Total memory:", info.total / 1048576, "MiB")
            print("Free memory:", info.free / 1048576, "MiB")
            print("Used memory:", info.used / 1048576, "MiB")
            nvidia_smi.nvmlShutdown()

        return (
            objective,
            norm_sqd_diff,
            ctrl_loss_mean,
            ctrl_loss_std_err,
            trajectory,
            torch.mean(weight),
            torch.std(weight),
            stop_indicators,
        )
