# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory.
import numpy as np
import torch
from tqdm.notebook import trange
import pickle
import os
from omegaconf import OmegaConf
import copy

from SOC_matching.gsbm_lib import EndPointGaussianPath, GammaSpline, init_spline


def stochastic_trajectories(
    sde,
    x0,
    t,
    lmbd,
    detach=True,
    verbose=False,
):
    xt = [x0]
    noises = []
    controls = []
    stop_indicators = [torch.ones(x0.shape[0]).to(x0.device)]
    fractional_timesteps = []
    log_path_weight_deterministic = torch.zeros(x0.shape[0]).to(x0.device)
    log_path_weight_stochastic = torch.zeros(x0.shape[0]).to(x0.device)
    log_path_weight_deterministic_list = [log_path_weight_deterministic]
    log_path_weight_stochastic_list = [log_path_weight_stochastic]
    log_terminal_weight = torch.zeros(x0.shape[0]).to(x0.device)
    stopping_condition = hasattr(sde, "Phi")  # If True process stops when Phi(X_t) < 0
    stop_inds = torch.ones(x0.shape[0]).to(
        x0.device
    )  # ones if not stopped, zero if stopped
    for t0, t1 in zip(t[:-1], t[1:]):
        dt = t1 - t0
        noise = torch.randn_like(x0).to(x0.device)
        noises.append(noise)
        u0 = sde.control(t0, x0, verbose=verbose)
        if stopping_condition:
            Phi_values_before_update = sde.Phi(x0)
            x0_before = x0
        update = (
            sde.b(t0, x0) + torch.einsum("ij,bj->bi", sde.sigma, u0)
        ) * dt + torch.sqrt(lmbd * dt) * torch.einsum("ij,bj->bi", sde.sigma, noise)
        x0 = x0 + stop_inds.unsqueeze(1) * update
        if stopping_condition:
            Phi_values_after_update = sde.Phi(x0)
            not_stopped = torch.logical_and(
                Phi_values_before_update > 0, Phi_values_after_update > 0
            ).to(torch.float)
            just_stopped = torch.logical_and(
                Phi_values_before_update > 0, Phi_values_after_update < 0
            ).to(torch.float)
            step_fraction = just_stopped * (
                (Phi_values_before_update)
                / (Phi_values_before_update - Phi_values_after_update + 1e-6)
                + 1e-6
            )
            x0 = (
                just_stopped.unsqueeze(1)
                * (
                    x0_before
                    + step_fraction.unsqueeze(1) * stop_inds.unsqueeze(1) * update
                )
                + (1 - just_stopped.unsqueeze(1)) * x0
            )
            fractional_timestep = (
                just_stopped * step_fraction**2 * dt + not_stopped * dt
            )
            fractional_timesteps.append(fractional_timestep)
            stop_inds = sde.Phi(x0) > 0
            stop_indicators.append(stop_inds)
        else:
            fractional_timesteps.append(dt * torch.ones(x0.shape[0]).to(x0.device))
            stop_indicators.append(torch.ones(x0.shape[0]).to(x0.device))
        xt.append(x0)
        controls.append(u0)

        if stopping_condition:
            log_path_weight_deterministic = (
                log_path_weight_deterministic
                + fractional_timestep
                / lmbd
                * (-sde.f(t0, x0) - 0.5 * torch.sum(u0**2, dim=1))
            )
            log_path_weight_stochastic = log_path_weight_stochastic + torch.sqrt(
                fractional_timestep / lmbd
            ) * (-torch.sum(u0 * noise, dim=1))
        else:
            log_path_weight_deterministic = (
                log_path_weight_deterministic
                + dt / lmbd * (-sde.f(t0, x0) - 0.5 * torch.sum(u0**2, dim=1))
            )
            log_path_weight_stochastic = log_path_weight_stochastic + torch.sqrt(
                dt / lmbd
            ) * (-torch.sum(u0 * noise, dim=1))

        log_path_weight_deterministic_list.append(log_path_weight_deterministic)
        log_path_weight_stochastic_list.append(log_path_weight_stochastic)

    log_terminal_weight = -sde.g(x0) / lmbd

    if detach:
        return (
            torch.stack(xt).detach(),
            torch.stack(noises).detach(),
            torch.stack(stop_indicators).detach(),
            torch.stack(fractional_timesteps).detach()
            if len(fractional_timesteps) > 0
            else None,
            log_path_weight_deterministic.detach(),
            log_path_weight_stochastic.detach(),
            log_terminal_weight.detach(),
            torch.stack(controls).detach(),
            torch.stack(log_path_weight_deterministic_list).detach(),
            torch.stack(log_path_weight_stochastic_list).detach(),
        )
    else:
        return (
            torch.stack(xt),
            torch.stack(noises),
            torch.stack(stop_indicators),
            torch.stack(fractional_timesteps).detach()
            if len(fractional_timesteps) > 0
            else None,
            log_path_weight_deterministic,
            log_path_weight_stochastic,
            log_terminal_weight,
            torch.stack(controls),
            torch.stack(log_path_weight_deterministic_list),
            torch.stack(log_path_weight_stochastic_list),
        )

def grad_control(ts, states, sde, sigma):
    with torch.enable_grad():

        def control_eval(ts, states):
            ts_repeat = ts.unsqueeze(1).unsqueeze(2).repeat(1, states.shape[1], 1)
            tx = torch.cat([ts_repeat, states], dim=-1)
            tx_reshape = torch.reshape(tx, (-1, tx.shape[2]))

            # Evaluate nabla_V
            nabla_V = sde.nabla_V(tx_reshape)
            nabla_V = torch.reshape(nabla_V, states.shape)

            control_learned = -torch.einsum(
                "ij,...j->...i", torch.transpose(sigma, 0, 1), nabla_V
            )

            return control_learned

        states = states.requires_grad_(True)
        # output = torch.autograd.grad(self.f(t, x).sum(), states)[0]
        output = torch.autograd.grad(torch.sum(control_eval(ts, states), dim=(0,1)), states)[0]
        return output

def grad_norm_control(ts, states, sde, sigma):
    with torch.enable_grad():

        def norm_control(ts, states):
            ts_repeat = ts.unsqueeze(1).unsqueeze(2).repeat(1, states.shape[1], 1)
            tx = torch.cat([ts_repeat, states], dim=-1)
            tx_reshape = torch.reshape(tx, (-1, tx.shape[2]))

            # Evaluate nabla_V
            nabla_V = sde.nabla_V(tx_reshape)
            nabla_V = torch.reshape(nabla_V, states.shape)

            control_learned = -torch.einsum(
                "ij,...j->...i", torch.transpose(sigma, 0, 1), nabla_V
            )

            return 0.5 * torch.sum(control_learned ** 2, dim=2)

        states = states.requires_grad_(True)
        # output = torch.autograd.grad(self.f(t, x).sum(), states)[0]
        output = torch.autograd.grad(norm_control(ts, states).sum(), states)[0]
        return output

def control_objective(
    sde, x0, ts, lmbd, batch_size, total_n_samples=65536, verbose=False
):
    n_batches = int(total_n_samples // batch_size)
    effective_n_samples = n_batches * batch_size
    for k in range(n_batches):
        state0 = x0.repeat(batch_size, 1)
        (
            _,
            _,
            _,
            _,
            log_path_weight_deterministic,
            _,
            log_terminal_weight,
            _,
            _,
            _,
        ) = stochastic_trajectories(
            sde,
            state0,
            ts.to(state0),
            lmbd,
            verbose=verbose,
        )
        if k == 0:
            ctrl_losses = -lmbd * (log_path_weight_deterministic + log_terminal_weight)
        else:
            ctrl_loss = -lmbd * (log_path_weight_deterministic + log_terminal_weight)
            ctrl_losses = torch.cat((ctrl_losses, ctrl_loss), 0)
        if k % 32 == 31:
            print(f"Batch {k+1}/{n_batches} done")
    return torch.mean(ctrl_losses), torch.std(ctrl_losses) / np.sqrt(
        effective_n_samples - 1
    )


def normalization_constant(
    sde, x0, ts, cfg, n_batches_normalization=512, ground_truth_control=None
):
    log_weights_list = []
    weights_list = []

    if ground_truth_control is not None:
        norm_sqd_diff_mean = 0
    for k in range(n_batches_normalization):
        (
            states,
            _,
            _,
            _,
            log_path_weight_deterministic,
            log_path_weight_stochastic,
            log_terminal_weight,
            controls,
            _,
            _,
        ) = stochastic_trajectories(
            sde,
            x0,
            ts.to(x0),
            cfg.method.lmbd,
        )
        log_weights = (
            log_path_weight_deterministic
            + log_path_weight_stochastic
            + log_terminal_weight
        )
        log_weights_list.append(log_weights)
        weights = torch.exp(
            log_path_weight_deterministic
            + log_path_weight_stochastic
            + log_terminal_weight
        )
        weights_list.append(weights)

        if ground_truth_control is not None:
            gt_controls = ground_truth_control(ts, states, t_is_tensor=True)[
                :-1, :, :
            ].detach()
            norm_sqd_diff = torch.sum(
                (gt_controls - controls) ** 2
                * weights.unsqueeze(0).unsqueeze(2)
                / (gt_controls.shape[0] * gt_controls.shape[1])
            )
            norm_sqd_diff_mean += norm_sqd_diff
        if k % 32 == 31:
            print(f"Batch {k+1}/{n_batches_normalization} done")
    if ground_truth_control is not None:
        norm_sqd_diff_mean = norm_sqd_diff_mean / n_batches_normalization
    else:
        norm_sqd_diff_mean = None

    log_weights = torch.stack(log_weights_list, dim=1)
    weights = torch.stack(weights_list, dim=1)

    print(
        f"Average and std. dev. of log_weights for all batches: {torch.mean(log_weights)} {torch.std(log_weights)}"
    )

    normalization_const = torch.mean(weights)
    normalization_const_std_error = torch.std(weights) / np.sqrt(
        weights.shape[0] * weights.shape[1] - 1
    )
    return normalization_const, normalization_const_std_error, norm_sqd_diff_mean


def solution_Ricatti(R_inverse, A, P, Q, t):
    FT = Q
    Ft = [FT]
    for t0, t1 in zip(t[:-1], t[1:]):
        dt = t1 - t0
        FT = FT - dt * (
            -torch.matmul(torch.transpose(A, 0, 1), FT)
            - torch.matmul(FT, A)
            + 2 * torch.matmul(torch.matmul(FT, R_inverse), FT)
            - P
        )
        Ft.append(FT)
    Ft.reverse()
    return torch.stack(Ft)


def optimal_control_LQ(sigma, A, P, Q, t):
    R_inverse = torch.matmul(sigma, torch.transpose(sigma, 0, 1))
    Ft = solution_Ricatti(R_inverse, A, P, Q, t)
    ut = -2 * torch.einsum("ij,bjk->bik", torch.transpose(sigma, 0, 1), Ft)
    return ut


def exponential_t_A(t, A):
    return torch.matrix_exp(t.unsqueeze(1).unsqueeze(2) * A.unsqueeze(0))


def get_folder_name(cfg):
    folder_name = (
        cfg.method.algorithm
        + "_"
        + cfg.method.setting
        + "_"
        + str(cfg.method.lmbd)
        + "_"
        + str(cfg.method.T)
        + "_"
        + str(cfg.method.num_steps)
        + "_"
        + str(cfg.method.use_warm_start)
        + "_"
        + str(cfg.method.seed)
        + "_"
        + str(cfg.optim.batch_size)
        + "_"
        + str(cfg.optim.M_lr)
        + "_"
        + str(cfg.optim.nabla_V_lr)
    )
    return folder_name


def get_folder_names_plots(cfg):
    folder_names = []
    if cfg.method.setting == "molecular_dynamics":
        algorithms = [
            "SOCM",
            "rel_entropy",
            "cross_entropy",
            "log-variance",
            "moment",
            "variance",
            "UW_SOCM",
            "reinforce",
        ]
    else:
        algorithms = [
            "SOCM",
            "SOCM_const_M",
            "SOCM_adjoint",
            "rel_entropy",
            "cross_entropy",
            "log-variance",
            "moment",
            "variance",
            "UW_SOCM",
            "reinforce",
        ]
    for k, algorithm in enumerate(algorithms):
        folder_name = (
            "../../outputs/multiruns/"
            + str(k)
            + "/"
            + algorithm
            + "_"
            + cfg.method.setting
            + "_"
            + str(cfg.method.lmbd)
            + "_"
            + str(cfg.method.T)
            + "_"
            + str(cfg.method.num_steps)
            + "_"
            + str(cfg.method.use_warm_start)
            + "_"
            + str(cfg.method.seed)
            + "_"
            + str(cfg.optim.batch_size)
            + "_"
            + str(cfg.optim.M_lr)
            + "_"
            + str(cfg.optim.nabla_V_lr)
        )
        folder_names.append(folder_name)
    plots_folder_name = (
        "../../outputs/multiruns/plots/"
        + cfg.method.setting
        + "_"
        + str(cfg.method.lmbd)
        + "_"
        + str(cfg.method.T)
        + "_"
        + str(cfg.method.num_steps)
        + "_"
        + str(cfg.method.use_warm_start)
        + "_"
        + str(cfg.method.seed)
        + "_"
        + str(cfg.optim.batch_size)
        + "_"
        + str(cfg.optim.M_lr)
        + "_"
        + str(cfg.optim.nabla_V_lr)
    )
    return folder_names, plots_folder_name


def get_file_name(folder_name, num_iterations=0, last=False):
    if last:
        return folder_name + "/last.pkl"
    file_name = str(num_iterations)
    print(f"folder_name: {folder_name}")
    return folder_name + "/" + file_name + ".pkl"


def get_file_names_plots(folder_names, num_iterations=0, last=False):
    file_names = []
    for folder_name in folder_names:
        if last:
            file_name = folder_name + "/last.pkl"
        else:
            file_name = folder_name + "/" + str(num_iterations) + ".pkl"
        file_names.append(file_name)
    return file_names


def save_results(results, folder_name, file_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    with open(file_name, "wb") as f:
        pickle.dump(results, f)


def retrieve_results(file_name):
    with open(file_name, "rb") as f:
        results = pickle.load(f)
    return results


def compute_EMA(value, EMA_value, EMA_coeff=0.01, itr=0):
    itr_avg = int(np.floor(1 / EMA_coeff))
    if itr == 0:
        return value
    elif itr <= itr_avg:
        return (value + itr * EMA_value) / (itr + 1)
    else:
        return EMA_coeff * value + (1 - EMA_coeff) * EMA_value


def fit_gpath(problem, gpath, optim_cfg, eps=0.001, verbose=False):
    """
    V: xt: (*, T, D), t: (T,), gpath --> (*, T)
    """

    results = {}
    results["init_mean"] = copy.deepcopy(gpath.mean)  # .cpu()
    results["init_gamma"] = copy.deepcopy(gpath.gamma)  # .cpu()

    sigma_inverse = torch.inverse(problem.sigma)

    ### optimize spline
    B, D, N, T, device = gpath.B, gpath.D, optim_cfg.N, optim_cfg.num_step, gpath.device
    optim = torch.optim.Adam(
        [
            {"params": gpath.mean.parameters(), "lr": optim_cfg["lr_mean"]},
            {"params": gpath.gamma.parameters(), "lr": optim_cfg["lr_mean"]},
        ],
        eps=1e-4,
    )
    gpath.train()
    losses = np.zeros(optim_cfg.nitr)
    _range = trange if verbose else range
    EMA_coeff = 0.01
    EMA_loss = 0
    EMA_cost_s = 0
    EMA_cost_c = 0
    EMA_cost_T = 0
    for itr in _range(optim_cfg.nitr):
        if itr % 500 == 0:
            print(f"{itr}/{optim_cfg.nitr}")
        optim.zero_grad()

        t = torch.linspace(eps, 1 - eps, T, device=device)
        xt, ut = gpath(t, N, direction="fwd")
        assert xt.shape == ut.shape == (B, N, T, D)
        xt = xt.reshape(-1, T, D).permute(1, 0, 2)
        ut = ut.reshape(-1, T, D).permute(1, 0, 2)

        b_eval = problem.b(t, xt)
        cost_s = problem.f(t, xt).mean() * problem.T
        ctrl = torch.einsum("ij,...j->...i", sigma_inverse, ut - b_eval)
        if itr == 0:
            print(f"xt.shape: {xt.shape}")
        cost_c = 0.5 * (ctrl[:, :, :] ** 2).sum(dim=-1).mean() * problem.T
        cost_T = problem.g(xt[-1, :, :]).mean()

        loss = (cost_s + cost_c + cost_T).mean()

        if itr == 0:
            EMA_loss = loss
            EMA_cost_s = cost_s
            EMA_cost_c = cost_c
            EMA_cost_T = cost_T
        else:
            EMA_loss = EMA_coeff * loss + (1 - EMA_coeff) * EMA_loss
            EMA_cost_s = EMA_coeff * cost_s + (1 - EMA_coeff) * EMA_cost_s
            EMA_cost_c = EMA_coeff * cost_c + (1 - EMA_coeff) * EMA_cost_c
            EMA_cost_T = EMA_coeff * cost_T + (1 - EMA_coeff) * EMA_cost_T
        if itr % 500 == 0:
            print(
                f"loss: {EMA_loss}, cost_s: {EMA_cost_s}, cost_c: {EMA_cost_c}, cost_T: {EMA_cost_T}"
            )

        loss.backward()
        optim.step()
        losses[itr] = loss.cpu().item()

    gpath.eval()

    results["final_mean"] = copy.deepcopy(gpath.mean)
    results["final_gamma"] = copy.deepcopy(gpath.gamma)
    results["gpath"] = copy.deepcopy(gpath)
    results["losses"] = losses

    return results


def restricted_SOC(problem, x0, x1, device, cfg):

    optim_cfg = OmegaConf.create(
        {
            "N": 512,  # 512,
            "num_step": 200,
            "lr_mean": cfg.optim.splines_lr,  # 0.02,
            "lr_gamma": cfg.optim.splines_lr,  # 0.002,
            "momentum": 0.0,
            "nitr": cfg.method.num_iterations_splines,
        }
    )

    B, S = 1, 21  # 1, 11  # number of splines and number of knots
    x0 = x0.repeat((B, 1))
    x1 = x1.repeat((B, 1))

    gpath = EndPointGaussianPath(
        mean=init_spline(x0, x1, S),
        sigma=problem.sigma,
        gamma=GammaSpline(
            torch.linspace(0, 1, S),
            torch.ones(B, S, 1),
            sigma=1.0,
            fix_init=True,
            init_knots=1,
        ),
    ).to(device)

    result = fit_gpath(problem, gpath, optim_cfg, verbose=True)
    return result
