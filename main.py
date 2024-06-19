# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory.
import torch
from tqdm.notebook import tqdm
import sys
import logging
import os
import time
import json

import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)

from SOC_matching.utils import (
    get_folder_name,
    get_file_name,
    control_objective,
    save_results,
    compute_EMA,
    normalization_constant,
    stochastic_trajectories,
)
from SOC_matching.method import (
    SOC_Solver,
)
from SOC_matching.experiment_settings.settings import define_variables


@hydra.main(version_base=None, config_path="configs", config_name="soc")
def main(cfg: DictConfig):

    logging.getLogger("lightning.pytorch").setLevel(logging.getLevelName("INFO"))

    print(cfg)

    print("Found {} CUDA devices.".format(torch.cuda.device_count()))
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(
            "{} \t Memory: {:.2f}GB".format(
                props.name, props.total_memory / (1024**3)
            )
        )

    keys = [
        "SLURM_NODELIST",
        "SLURM_JOB_ID",
        "SLURM_NTASKS",
        "SLURM_JOB_NAME",
        "SLURM_PROCID",
        "SLURM_LOCALID",
        "SLURM_NODEID",
    ]
    log.info(json.dumps({k: os.environ.get(k, None) for k in keys}, indent=4))

    cmd_str = " \\\n".join([f"python {sys.argv[0]}"] + ["\t" + x for x in sys.argv[1:]])
    with open("cmd.sh", "w") as fout:
        print("#!/bin/bash\n", file=fout)
        print(cmd_str, file=fout)

    log.info(f"CWD: {os.getcwd()}")

    if cfg.method.use_gpu:
        cfg.method.device = "cuda:" + str(cfg.method.device_number)
    else:
        cfg.method.device = "cpu"
    torch.manual_seed(cfg.method.seed)
    algorithm = cfg.method.algorithm
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

    ts = torch.linspace(0, cfg.method.T, cfg.method.num_steps + 1).to(cfg.method.device)

    folder_name = get_folder_name(cfg)
    file_name = get_file_name(folder_name, num_iterations=cfg.method.num_iterations)

    EMA_loss = 0
    EMA_norm_sqd_diff = 0
    EMA_coeff = 0.01
    EMA_weight_mean_coeff = 0.1
    EMA_coeff_optimal = 0.02

    x0, sigma, optimal_sde, neural_sde, u_warm_start = define_variables(cfg, ts)
    if optimal_sde is not None:
        ground_truth_control = optimal_sde.u
    else:
        ground_truth_control = None

    state0 = x0.repeat(cfg.optim.batch_size, 1)

    ########### Compute normalization constant and control L2 error for initial control ############
    print(
        f"Estimating normalization constant and control L2 error for initial control..."
    )
    (
        normalization_const,
        normalization_const_std_error,
        norm_sqd_diff_mean,
    ) = normalization_constant(
        neural_sde,
        state0,
        ts,
        cfg,
        n_batches_normalization=512,
        ground_truth_control=ground_truth_control,
    )
    print(
        f"Normalization_constant (mean and std. error): {normalization_const:5.8E} {normalization_const_std_error:5.8E}"
    )
    if ground_truth_control is not None:
        print(
            f"Control L2 error for initial control: {norm_sqd_diff_mean / normalization_const}"
        )

    ########### Compute control loss for optimal control ############
    if optimal_sde is not None:
        (
            optimal_control_objective_mean,
            optimal_control_objective_std_error,
        ) = control_objective(
            optimal_sde,
            x0,
            ts,
            cfg.method.lmbd,
            cfg.optim.batch_size,
            total_n_samples=cfg.method.n_samples_control,
            verbose=False,
        )
        print(
            f"Optimal control loss mean: {optimal_control_objective_mean:5.10f}, Optimal control loss std. error: {optimal_control_objective_std_error:5.10f}"
        )

    soc_solver = SOC_Solver(
        neural_sde,
        x0,
        ground_truth_control,
        T=cfg.method.T,
        num_steps=cfg.method.num_steps,
        lmbd=cfg.method.lmbd,
        d=cfg.method.d,
        sigma=sigma,
        output_matrix=cfg.method.output_matrix,
    )

    if algorithm == "SOCM_exp":
        soc_solver.gamma = torch.nn.Parameter(
            torch.tensor([cfg.method.gamma]).to(cfg.method.device)
        )
    else:
        soc_solver.gamma = cfg.method.gamma

    ####### Set optimizer ########
    if algorithm == "moment":
        optimizer = torch.optim.Adam(
            [{"params": soc_solver.neural_sde.parameters()}]
            + [{"params": soc_solver.y0, "lr": cfg.optim.y0_lr}],
            lr=cfg.optim.nabla_V_lr,
            eps=cfg.optim.adam_eps,
        )
    elif algorithm == "SOCM_exp":
        optimizer = torch.optim.Adam(
            [{"params": soc_solver.neural_sde.parameters()}]
            + [{"params": soc_solver.gamma, "lr": cfg.optim.M_lr}],
            lr=cfg.optim.nabla_V_lr,
            eps=cfg.optim.adam_eps,
        )
    elif algorithm == "SOCM":
        if cfg.method.use_stopping_time or algorithm == 'SOCM_scalar_2B' or algorithm == 'UW_SOCM_sc_2B' or algorithm == 'SOCM_cost_sc_2B':
            optimizer = torch.optim.Adam(
                [{"params": soc_solver.neural_sde.nabla_V.parameters()}]
                + [
                    {
                        "params": soc_solver.neural_sde.M.sigmoid_layers.parameters(),
                        "lr": cfg.optim.M_lr,
                    }
                ]
                + [
                    {
                        "params": soc_solver.neural_sde.gamma,
                        "lr": cfg.optim.M_lr,
                    }
                ]
                + [
                    {
                        "params": soc_solver.neural_sde.gamma2,
                        "lr": cfg.optim.M_lr,
                    }
                ],
                lr=cfg.optim.nabla_V_lr,
                eps=cfg.optim.adam_eps,
            )
        else:
            optimizer = torch.optim.Adam(
                [{"params": soc_solver.neural_sde.nabla_V.parameters()}]
                + [
                    {
                        "params": soc_solver.neural_sde.M.sigmoid_layers.parameters(),
                        "lr": cfg.optim.M_lr,
                    }
                ]
                + [
                    {
                        "params": soc_solver.neural_sde.gamma,
                        "lr": cfg.optim.M_lr,
                    }
                ],
                lr=cfg.optim.nabla_V_lr,
                eps=cfg.optim.adam_eps,
            )
    elif algorithm == "discrete_adjoint" or algorithm == "discrete_adjoint_STL":
        optimizer = torch.optim.Adam(
            soc_solver.parameters(), lr=cfg.optim.nabla_V_lr, eps=cfg.optim.adam_eps
        )
    else:
        print(f'Optimizer over soc_solver.parameters()')
        optimizer = torch.optim.Adam(
            soc_solver.parameters(), lr=cfg.optim.nabla_V_lr, eps=cfg.optim.adam_eps
        )
    optimizer.zero_grad()

    soc_solver.algorithm = cfg.method.algorithm
    soc_solver.training_info = dict()
    training_info = soc_solver.training_info
    training_variables = [
        "time_per_iteration",
        "EMA_time_per_iteration",
        "loss",
        "EMA_loss",
        "norm_sqd_diff",
        "EMA_norm_sqd_diff",
        "weight_mean",
        "EMA_weight_mean",
        "weight_std",
        "EMA_weight_std",
        "grad_norm_sqd",
        "EMA_grad_norm_sqd",
        "sqd_norm_EMA_grad",
    ]
    control_objective_variables = [
        "control_objective_mean",
        "control_objective_std_err",
        "control_objective_itr",
        "trajectories",
    ]
    optimal_variables = [
        "norm_sqd_diff_optimal",
        "EMA_norm_sqd_diff_optimal",
        "optimal_itr",
    ]
    for var in training_variables:
        training_info[var] = []
    for var in control_objective_variables:
        training_info[var] = []
    for var in optimal_variables:
        training_info[var] = []
    if cfg.method.use_warm_start:
        training_info["restricted_control"] = u_warm_start
        training_info["trajectories"] = []
    training_info["cfg"] = cfg

    compute_L2_error = ground_truth_control is not None

    ###### Train control ######
    with torch.inference_mode(False):
        with torch.enable_grad():
            for itr in range(cfg.method.num_iterations):
                start = time.time()

                compute_control_objective = (
                    itr == 0
                    or itr % cfg.method.compute_control_objective_every
                    == cfg.method.compute_control_objective_every - 1
                    or itr == cfg.method.num_iterations - 1
                )
                verbose = itr == 0
                (
                    loss,
                    norm_sqd_diff,
                    control_objective_mean,
                    control_objective_std_err,
                    trajectory,
                    weight_mean,
                    weight_std,
                    stop_indicators,
                ) = soc_solver.loss(
                    cfg.optim.batch_size,
                    compute_L2_error=compute_L2_error,
                    algorithm=algorithm,
                    optimal_control=ground_truth_control,
                    compute_control_objective=compute_control_objective,
                    total_n_samples=cfg.method.n_samples_control,
                    verbose=verbose,
                    u_warm_start=u_warm_start,
                    use_warm_start=cfg.method.use_warm_start,
                    use_stopping_time=cfg.method.use_stopping_time,
                    efficient_memory=cfg.method.efficient_memory,
                )

                if compute_L2_error:
                    norm_sqd_diff = norm_sqd_diff / normalization_const
                if (
                    algorithm == "SOCM_const_M"
                    or algorithm == "SOCM_exp"
                    or algorithm == "SOCM"
                    or algorithm == "SOCM_adjoint"
                    or algorithm == "cross_entropy"
                ):
                    loss = loss / normalization_const
                elif algorithm == "variance":
                    loss = loss / normalization_const**2
                loss.backward()

                grad = []
                grad_norm_sqd = 0
                for param in soc_solver.neural_sde.nabla_V.parameters():
                    grad.append(param.grad.data.detach())
                    grad_norm_sqd += torch.norm(param.grad.data.detach()) ** 2

                if itr == 0:
                    EMA_grad = grad
                    EMA_grad_norm_sqd = grad_norm_sqd
                else:
                    for k in range(len(EMA_grad)):
                        EMA_grad[k] = compute_EMA(
                            grad[k], EMA_grad[k], EMA_coeff=EMA_coeff, itr=itr
                        )
                    EMA_grad_norm_sqd = compute_EMA(
                        grad_norm_sqd, EMA_grad_norm_sqd, EMA_coeff=EMA_coeff, itr=itr
                    )

                sqd_norm_EMA_grad = 0
                for k in range(len(EMA_grad)):
                    sqd_norm_EMA_grad += torch.norm(EMA_grad[k]) ** 2

                with torch.no_grad():
                    optimizer.step()
                    optimizer.zero_grad()

                    end = time.time()
                    time_per_iteration = end - start

                    normalization_const = compute_EMA(
                        weight_mean.detach(),
                        normalization_const,
                        EMA_coeff=EMA_weight_mean_coeff,
                        itr=itr,
                    )

                    if itr == 0:
                        EMA_time_per_iteration = time_per_iteration
                        EMA_loss = loss.detach()
                        EMA_weight_mean = weight_mean.detach()
                        EMA_weight_std = weight_std.detach()
                        if compute_L2_error:
                            EMA_norm_sqd_diff = norm_sqd_diff.detach()
                    else:
                        EMA_time_per_iteration = compute_EMA(
                            time_per_iteration,
                            EMA_time_per_iteration,
                            EMA_coeff=EMA_coeff,
                            itr=itr,
                        )
                        EMA_loss = compute_EMA(
                            loss.detach(), EMA_loss, EMA_coeff=EMA_coeff, itr=itr
                        )
                        EMA_weight_mean = compute_EMA(
                            weight_mean.detach(),
                            EMA_weight_mean,
                            EMA_coeff=EMA_coeff,
                            itr=itr,
                        )
                        EMA_weight_std = compute_EMA(
                            weight_std.detach(),
                            EMA_weight_std,
                            EMA_coeff=EMA_coeff,
                            itr=itr,
                        )
                        if compute_L2_error:
                            EMA_norm_sqd_diff = compute_EMA(
                                norm_sqd_diff.detach(),
                                EMA_norm_sqd_diff,
                                EMA_coeff=EMA_coeff,
                                itr=itr,
                            )

                    if itr % cfg.method.save_every == 0:
                        training_info["time_per_iteration"].append(time_per_iteration)
                        training_info["EMA_time_per_iteration"].append(
                            EMA_time_per_iteration
                        )
                        training_info["loss"].append(loss.detach())
                        training_info["EMA_loss"].append(EMA_loss)
                        training_info["weight_mean"].append(weight_mean.detach())
                        training_info["EMA_weight_mean"].append(EMA_weight_mean)
                        training_info["weight_std"].append(weight_std.detach())
                        training_info["EMA_weight_std"].append(EMA_weight_std)
                        training_info["grad_norm_sqd"].append(grad_norm_sqd.detach())
                        training_info["EMA_grad_norm_sqd"].append(EMA_grad_norm_sqd)
                        training_info["sqd_norm_EMA_grad"].append(sqd_norm_EMA_grad)
                        if compute_L2_error:
                            training_info["norm_sqd_diff"].append(norm_sqd_diff.detach())
                            training_info["EMA_norm_sqd_diff"].append(EMA_norm_sqd_diff)

                    if itr % 10 == 0 or itr == cfg.method.num_iterations - 1:
                        if compute_L2_error:
                            start_optimal = time.time()
                            norm_sqd_diff_optimal = 0
                            norm_sqd_diff_optimal_per_time = 0
                            for _ in range(10):
                                (
                                    optimal_states,
                                    _,
                                    _,
                                    _,
                                    _,
                                    _,
                                    _,
                                    _,
                                    _,
                                    _,
                                    _,
                                ) = stochastic_trajectories(
                                    optimal_sde,
                                    state0,
                                    ts.to(state0),
                                    cfg.method.lmbd,
                                )
                                # if algorithm == "discrete_adjoint":
                                #     target_control = ground_truth_control(ts.to(state0), optimal_states, t_is_tensor=True)[
                                #         :-1, :, :
                                #     ].detach()
                                # else:
                                target_control = ground_truth_control(ts.to(state0), optimal_states, t_is_tensor=True)
                                ts_repeat = ts.unsqueeze(1).unsqueeze(2).repeat(1, optimal_states.shape[1], 1)
                                tx_optimal = torch.cat([ts_repeat, optimal_states], dim=-1)
                                tx_optimal_reshape = torch.reshape(tx_optimal, (-1, tx_optimal.shape[2]))

                                # Evaluate nabla_V
                                nabla_V = neural_sde.nabla_V(tx_optimal_reshape)
                                nabla_V = torch.reshape(nabla_V, optimal_states.shape)

                                learned_control = -torch.einsum(
                                    "ij,abj->abi", torch.transpose(sigma, 0, 1), nabla_V
                                )

                                # print(f'target_control.shape: {target_control.shape}, learned_control.shape: {learned_control.shape}')
                                norm_sqd_diff_optimal += torch.sum(
                                    (target_control - learned_control) ** 2
                                    / (target_control.shape[0] * target_control.shape[1])
                                ).detach()

                                #### TO DEBUG ######
                                norm_sqd_diff_optimal_per_time += torch.sum(
                                    (target_control - learned_control) ** 2
                                    / (target_control.shape[0] * target_control.shape[1]), dim=[1,2]
                                ).detach()
                                #### TO DEBUG ######

                            norm_sqd_diff_optimal /= 10
                            #### TO DEBUG ######
                            norm_sqd_diff_optimal_per_time /= 10
                            print(f'norm_sqd_diff_optimal_per_time: {norm_sqd_diff_optimal_per_time}')
                            #### TO DEBUG ######
                            end_optimal = time.time()
                            time_optimal = end_optimal - start_optimal

                            if itr == 0:
                                EMA_norm_sqd_diff_optimal = norm_sqd_diff_optimal
                            else:
                                EMA_norm_sqd_diff_optimal = compute_EMA(
                                                                norm_sqd_diff_optimal.detach(),
                                                                EMA_norm_sqd_diff_optimal,
                                                                EMA_coeff=EMA_coeff_optimal,
                                                                itr=itr,
                                                            )
                                
                            training_info["norm_sqd_diff_optimal"].append(norm_sqd_diff_optimal)
                            training_info["EMA_norm_sqd_diff_optimal"].append(EMA_norm_sqd_diff_optimal)
                            training_info["optimal_itr"].append(itr + 1)

                            print(
                                f"{itr} - {time_per_iteration:5.3f}s/it (EMA {EMA_time_per_iteration:5.3f}s/it): {loss.item():5.5f} {EMA_loss.item():5.5f} {norm_sqd_diff.item():5.5f} {EMA_norm_sqd_diff.item():5.6f} {EMA_weight_mean.item():5.6E} {EMA_weight_std.item():5.6E}"
                            )
                            print(
                                f"{time_optimal:5.3f}s: {norm_sqd_diff_optimal.item():5.5f} {EMA_norm_sqd_diff_optimal.item():5.5f}"
                            )
                        else:
                            print(
                                f"{itr} - {time_per_iteration:5.3f}s/it (EMA {EMA_time_per_iteration:5.3f}s/it): {loss.item():5.5f} {EMA_loss.item():5.5f} {EMA_weight_mean.item():5.6E} {EMA_weight_std.item():5.6E}"
                            )
                        if algorithm == "moment":
                            print(f"soc_solver.y0: {soc_solver.y0.item()}")
                        elif algorithm == "SOCM_exp":
                            print(f"soc_solver.gamma: {soc_solver.gamma.item()}")
                        elif algorithm == "SOCM":
                            print(
                                f"soc_solver.neural_sde.M.gamma: {soc_solver.neural_sde.M.gamma.item()}"
                            )
                        elif algorithm == "reinf_unadj" or algorithm == "reinf":
                            if compute_L2_error:
                                print(f'normalization_const: {normalization_const}, norms_sqd_diff: {norm_sqd_diff.item()}, weight_mean: {weight_mean}')
                            else:
                                print(f'normalization_const: {normalization_const}, weight_mean: {weight_mean}')
                        if cfg.method.use_stopping_time:
                            print(
                                f"torch.mean(stop_indicators): {torch.mean(stop_indicators)}"
                            )

                        end = time.time()

                    if algorithm == "moment" and itr == 5000:
                        current_lr = optimizer.param_groups[-1]["lr"]
                        optimizer.param_groups[-1]["lr"] = 1e-4
                        new_lr = optimizer.param_groups[-1]["lr"]
                        print(f"current_lr: {current_lr}, new_lr: {new_lr}")

                    if (
                        itr == 0
                        or itr % cfg.method.compute_control_objective_every
                        == cfg.method.compute_control_objective_every - 1
                        or itr == cfg.method.num_iterations - 1
                    ):
                        print(
                            f"Control loss mean: {control_objective_mean:5.5f}, Control loss std. error: {control_objective_std_err:5.5f}"
                        )
                        training_info["control_objective_mean"].append(
                            control_objective_mean.detach()
                        )
                        training_info["control_objective_std_err"].append(
                            control_objective_std_err.detach()
                        )
                        training_info["control_objective_itr"].append(itr + 1)
                        training_info["trajectories"].append(trajectory)

                        soc_solver.num_iterations = itr + 1

                        file_name = get_file_name(folder_name, num_iterations=itr + 1)
                        save_results(soc_solver, folder_name, file_name)

                        file_name = get_file_name(
                            folder_name, num_iterations=itr + 1, last=True
                        )
                        save_results(soc_solver, folder_name, file_name)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        print(traceback.format_exc())
        sys.exit(1)
