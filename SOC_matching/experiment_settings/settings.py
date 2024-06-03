# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory.
import torch
import math

from SOC_matching.utils import (
    optimal_control_LQ,
    exponential_t_A,
    restricted_SOC,
)
from SOC_matching.models import (
    LinearControl,
    ConstantControlLinear,
    LowDimControl,
    RestrictedControl,
)
from SOC_matching.experiment_settings.OU_quadratic import OU_Quadratic
from SOC_matching.experiment_settings.OU_linear import OU_Linear
from SOC_matching.experiment_settings.double_well import DoubleWell
from SOC_matching.experiment_settings.molecular_dynamics import MolecularDynamics
from SOC_matching.experiment_settings.sampling import Sampler


def ground_truth_control(cfg, ts, x0, **kwargs):
    if (
        cfg.method.setting == "OU_quadratic_easy"
        or cfg.method.setting == "OU_quadratic_hard"
        or cfg.method.setting == "OU_quadratic_no_state_cost"
    ):
        R_inverse = torch.matmul(
            kwargs["sigma"], torch.transpose(kwargs["sigma"], 0, 1)
        )
        R = torch.inverse(R_inverse)

        ut = optimal_control_LQ(
            kwargs["sigma"], kwargs["A"], kwargs["P"], kwargs["Q"], ts
        )
        ut = LinearControl(ut, cfg.method.T)

        optimal_sde = OU_Quadratic(
            u=ut,
            lmbd=cfg.method.lmbd,
            A=kwargs["A"],
            P=kwargs["P"],
            Q=kwargs["Q"],
            sigma=kwargs["sigma"],
            T=cfg.method.T,
        )

        return optimal_sde

    elif cfg.method.setting == "OU_linear":
        exp_matrix = exponential_t_A(
            cfg.method.T - ts, torch.transpose(kwargs["A"], 0, 1)
        )
        ut = -torch.einsum(
            "aij,j->ai",
            torch.einsum(
                "ij,ajk->aik", torch.transpose(kwargs["sigma"], 0, 1), exp_matrix
            ),
            kwargs["omega"],
        )
        ut = ConstantControlLinear(ut, cfg.method.T)

        optimal_sde = OU_Linear(
            u=ut,
            lmbd=cfg.method.lmbd,
            A=kwargs["A"],
            omega=kwargs["omega"],
            sigma=kwargs["sigma"],
            T=cfg.method.T,
        )

        return optimal_sde

    elif cfg.method.setting == "double_well":
        optimal_sde = DoubleWell(
            lmbd=cfg.method.lmbd,
            kappa=kwargs["kappa"],
            nu=kwargs["nu"],
            sigma=kwargs["sigma"],
            T=cfg.method.T,
        )
        xb = 2.75
        delta_t = cfg.method.delta_t_optimal
        delta_x = cfg.method.delta_x_optimal
        ut_list = []
        for j in range(cfg.method.d):
            ut_discrete = optimal_sde.compute_reference_solution(
                T=cfg.method.T,
                delta_t=delta_t,
                xb=xb,
                delta_x=delta_x,
                lmbd=cfg.method.lmbd,
                idx=j,
            )
            print(f"ut_discrete.shape: {ut_discrete.shape}")
            ut_list.append(torch.from_numpy(ut_discrete).to(cfg.method.device))
        ut_discrete = torch.stack(ut_list, dim=2)
        print(f"ut_discrete.shape: {ut_discrete.shape}")
        print(f"torch.mean(ut_discrete): {torch.mean(ut_discrete)}")

        ut = LowDimControl(
            ut_discrete, cfg.method.T, xb, cfg.method.d, delta_t, delta_x
        )
        optimal_sde.u = ut

        return optimal_sde

    elif cfg.method.setting == "multiagent_8":
        optimal_sde = None
        return optimal_sde

    elif cfg.method.setting == "molecular_dynamics":
        optimal_sde = None
        return optimal_sde
    
    elif cfg.method.setting in ["sampling_cox", "sampling_funnel", "sampling_MG"]:
        optimal_sde = None
        return optimal_sde


def set_warm_start(cfg, sde, x0, sigma):
    if cfg.method.use_warm_start:
        # Use Gaussian path
        print(f"Solving Restricted Gaussian Stochastic Optimal Control problem...")
        if cfg.method.setting == "multiagent_8":
            endpoints = sde.g_center
        else:
            endpoints = torch.zeros_like(x0.unsqueeze(0)).to(cfg.method.device)
        result_spline = restricted_SOC(
            sde, x0.unsqueeze(0), endpoints, cfg.method.device, cfg
        )

        gpath = result_spline["gpath"]
        u_warm_start = RestrictedControl(
            gpath,
            sigma,
            sde.b,
            cfg.method.device,
            cfg.method.T,
            cfg.method.num_splines,
        )
        return u_warm_start
    else:
        return None


def define_neural_sde(cfg, ts, x0, u_warm_start, **kwargs):
    if (
        cfg.method.setting == "OU_quadratic_easy"
        or cfg.method.setting == "OU_quadratic_hard"
        or cfg.method.setting == "OU_quadratic_no_state_cost"
    ):
        neural_sde = OU_Quadratic(
            device=cfg.method.device,
            dim=cfg.method.d,
            hdims=cfg.arch.hdims,
            hdims_M=cfg.arch.hdims_M,
            lmbd=cfg.method.lmbd,
            A=kwargs["A"],
            P=kwargs["P"],
            Q=kwargs["Q"],
            sigma=kwargs["sigma"],
            gamma=cfg.method.gamma,
            T=cfg.method.T,
            scaling_factor_nabla_V=cfg.method.scaling_factor_nabla_V,
            scaling_factor_M=cfg.method.scaling_factor_M,
            u_warm_start=u_warm_start,
            use_warm_start=cfg.method.use_warm_start,
            output_matrix=cfg.method.output_matrix,
        )
    elif cfg.method.setting == "OU_linear":
        neural_sde = OU_Linear(
            device=cfg.method.device,
            dim=cfg.method.d,
            hdims=cfg.arch.hdims,
            hdims_M=cfg.arch.hdims_M,
            lmbd=cfg.method.lmbd,
            A=kwargs["A"],
            omega=kwargs["omega"],
            sigma=kwargs["sigma"],
            gamma=cfg.method.gamma,
            T=cfg.method.T,
            scaling_factor_nabla_V=cfg.method.scaling_factor_nabla_V,
            scaling_factor_M=cfg.method.scaling_factor_M,
            output_matrix=cfg.method.output_matrix,
        )
    elif cfg.method.setting == "double_well":
        neural_sde = DoubleWell(
            device=cfg.method.device,
            dim=cfg.method.d,
            hdims=cfg.arch.hdims,
            hdims_M=cfg.arch.hdims_M,
            lmbd=cfg.method.lmbd,
            kappa=kwargs["kappa"],
            nu=kwargs["nu"],
            sigma=kwargs["sigma"],
            gamma=cfg.method.gamma,
            T=cfg.method.T,
            scaling_factor_nabla_V=cfg.method.scaling_factor_nabla_V,
            scaling_factor_M=cfg.method.scaling_factor_M,
            output_matrix=cfg.method.output_matrix,
        )
    elif cfg.method.setting == "molecular_dynamics":
        neural_sde = MolecularDynamics(
            device=cfg.method.device,
            dim=cfg.method.d,
            hdims=cfg.arch.hdims,
            hdims_M=cfg.arch.hdims_M,
            lmbd=cfg.method.lmbd,
            kappa=kwargs["kappa"],
            sigma=kwargs["sigma"],
            gamma=cfg.method.gamma,
            T=cfg.method.T,
            scaling_factor_nabla_V=cfg.method.scaling_factor_nabla_V,
            scaling_factor_M=cfg.method.scaling_factor_M,
            use_stopping_time=cfg.method.use_stopping_time,
            output_matrix=cfg.method.output_matrix,
        )
    elif cfg.method.setting in ["sampling_cox", "sampling_funnel", "sampling_MG"]:
        neural_sde = Sampler(
            device=cfg.method.device,
            dim=cfg.method.d,
            hdims=cfg.arch.hdims,
            hdims_M=cfg.arch.hdims_M,
            setting=cfg.method.setting,
            lmbd=cfg.method.lmbd,
            sigma=kwargs["sigma"],
            gamma=cfg.method.gamma,
            T=cfg.method.T,
            scaling_factor_nabla_V=cfg.method.scaling_factor_nabla_V,
            scaling_factor_M=cfg.method.scaling_factor_M,
            output_matrix=cfg.method.output_matrix,
        )
    neural_sde.initialize_models(cfg.method.algorithm)
    return neural_sde


def define_variables(cfg, ts):
    if (
        cfg.method.setting == "OU_quadratic_easy"
        or cfg.method.setting == "OU_quadratic_hard"
        or cfg.method.setting == "OU_quadratic_no_state_cost"
    ):
        if cfg.method.d == 2:
            x0 = torch.tensor([0.4, 0.6]).to(cfg.method.device)
        else:
            x0 = 0.5 * torch.randn(cfg.method.d).to(cfg.method.device)
        print(f"x0: {x0}")
        sigma = torch.eye(cfg.method.d).to(cfg.method.device)
        if cfg.method.setting == "OU_quadratic_hard":
            A = 1.0 * torch.eye(cfg.method.d).to(cfg.method.device)
            P = 1.0 * torch.eye(cfg.method.d).to(cfg.method.device)
            Q = 0.5 * torch.eye(cfg.method.d).to(cfg.method.device)
        elif cfg.method.setting == "OU_quadratic_easy":
            A = 0.2 * torch.eye(cfg.method.d).to(cfg.method.device)
            P = 0.2 * torch.eye(cfg.method.d).to(cfg.method.device)
            Q = 0.1 * torch.eye(cfg.method.d).to(cfg.method.device)
        elif cfg.method.setting == "OU_quadratic_no_state_cost":
            A = 0.5 * torch.eye(cfg.method.d).to(cfg.method.device)
            P = 0.0 * torch.eye(cfg.method.d).to(cfg.method.device)
            Q = 0.25 * torch.eye(cfg.method.d).to(cfg.method.device)

        optimal_sde = ground_truth_control(cfg, ts, x0, sigma=sigma, A=A, P=P, Q=Q)
        u_warm_start = set_warm_start(cfg, optimal_sde, x0, sigma)
        neural_sde = define_neural_sde(
            cfg, ts, x0, u_warm_start, sigma=sigma, A=A, P=P, Q=Q
        )
        return x0, sigma, optimal_sde, neural_sde, u_warm_start

    elif cfg.method.setting == "OU_linear":
        x0 = torch.zeros(cfg.method.d).to(cfg.method.device)
        nu = 0.1
        xi = nu * torch.randn(cfg.method.d, cfg.method.d).to(cfg.method.device)
        omega = torch.ones(cfg.method.d).to(cfg.method.device)
        A = -torch.eye(cfg.method.d).to(cfg.method.device) + xi
        sigma = torch.eye(cfg.method.d).to(cfg.method.device) + xi

        optimal_sde = ground_truth_control(cfg, ts, x0, sigma=sigma, omega=omega, A=A)
        u_warm_start = set_warm_start(cfg, optimal_sde, x0, sigma)
        neural_sde = define_neural_sde(
            cfg, ts, x0, u_warm_start, sigma=sigma, omega=omega, A=A
        )
        return x0, sigma, optimal_sde, neural_sde, u_warm_start

    elif cfg.method.setting == "double_well":
        print(f"double_well")
        x0 = torch.zeros(cfg.method.d).to(cfg.method.device)

        kappa_i = 5
        nu_i = 3
        kappa = torch.ones(cfg.method.d).to(cfg.method.device)
        nu = torch.ones(cfg.method.d).to(cfg.method.device)
        kappa[0] = kappa_i
        kappa[1] = kappa_i
        kappa[2] = kappa_i
        nu[0] = nu_i
        nu[1] = nu_i
        nu[2] = nu_i

        sigma = torch.eye(cfg.method.d).to(cfg.method.device)

        optimal_sde = ground_truth_control(cfg, ts, x0, sigma=sigma, kappa=kappa, nu=nu)
        u_warm_start = set_warm_start(cfg, optimal_sde, x0, sigma)
        neural_sde = define_neural_sde(
            cfg, ts, x0, u_warm_start, sigma=sigma, kappa=kappa, nu=nu
        )

        return x0, sigma, optimal_sde, neural_sde, u_warm_start

    elif cfg.method.setting == "molecular_dynamics":
        print(f"molecular_dynamics")
        x0 = -torch.ones(cfg.method.d).to(cfg.method.device)

        kappa = torch.ones(cfg.method.d).to(cfg.method.device)
        sigma = torch.eye(cfg.method.d).to(cfg.method.device)

        optimal_sde = ground_truth_control(
            cfg,
            ts,
            x0,
            sigma=sigma,
            kappa=kappa,
        )
        u_warm_start = set_warm_start(cfg, optimal_sde, x0, sigma)
        neural_sde = define_neural_sde(
            cfg,
            ts,
            x0,
            u_warm_start,
            sigma=sigma,
            kappa=kappa,
        )

        return x0, sigma, optimal_sde, neural_sde, u_warm_start

    elif cfg.method.setting == "multiagent_8":
        print(f"multiagent_8")
        x0 = torch.tensor(
            [
                -4.0,
                4.5,
                -7.0,
                4.5,
                -4.0,
                1.5,
                -7.0,
                1.5,
                -4.0,
                -1.5,
                -7.0,
                -1.5,
                -4.0,
                -4.5,
                -7.0,
                -4.5,
            ]
        ).to(cfg.method.device)

        g_center = torch.tensor(
            [
                4.0,
                4.5,
                7.0,
                4.5,
                4.0,
                1.5,
                7.0,
                1.5,
                4.0,
                -1.5,
                7.0,
                -1.5,
                4.0,
                -4.5,
                7.0,
                -4.5,
            ]
        ).to(cfg.method.device)
        g_coeff = 2.00
        f_coeff = 0.05

        sigma = torch.eye(cfg.method.d).to(cfg.method.device)

        optimal_sde = ground_truth_control(
            cfg,
            ts,
            x0,
            sigma=sigma,
            g_center=g_center,
            g_coeff=g_coeff,
            f_coeff=f_coeff,
        )
        u_warm_start = set_warm_start(cfg, optimal_sde, x0, sigma)
        neural_sde = define_neural_sde(
            cfg,
            ts,
            x0,
            u_warm_start,
            sigma=sigma,
            g_center=g_center,
            g_coeff=g_coeff,
            f_coeff=f_coeff,
        )

        return x0, sigma, optimal_sde, neural_sde, u_warm_start
    
    elif cfg.method.setting in ["sampling_cox", "sampling_funnel", "sampling_MG"]:
        print(cfg.method.setting)
        ### Add batch_size as an argument
        x0 = torch.randn(cfg.optim.batch_size, cfg.method.d).to(cfg.method.device)

        sigma = math.sqrt(2) * torch.eye(cfg.method.d).to(cfg.method.device)

        optimal_sde = ground_truth_control(
            cfg,
            ts,
            x0,
            sigma=sigma,
        )
        u_warm_start = set_warm_start(cfg, optimal_sde, x0, sigma)
        neural_sde = define_neural_sde(
            cfg,
            ts,
            x0,
            u_warm_start,
            sigma=sigma,
        )
        return x0, sigma, optimal_sde, neural_sde, u_warm_start
