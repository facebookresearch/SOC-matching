# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory.
import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
import logging
import os

import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)

from SOC_matching.utils import get_folder_names_plots, get_file_names_plots


def pplot(ax=None):
    if ax is None:
        plt.grid(True, alpha=0.5)
        axoff(plt.gca())
    else:
        ax.grid(True, alpha=0.5)
        axoff(ax)
    return


def axoff(ax, keys=["top", "right"]):
    for k in keys:
        ax.spines[k].set_visible(False)
    return


def compute_SNR(training_info):
    EMA_grad_norm_sqd = torch.stack(training_info["EMA_grad_norm_sqd"]).cpu().numpy()
    sqd_norm_EMA_grad = torch.stack(training_info["sqd_norm_EMA_grad"]).cpu().numpy()
    variance = EMA_grad_norm_sqd - sqd_norm_EMA_grad
    SNR = sqd_norm_EMA_grad / (variance + 1e-7)
    training_info["EMA_SNR"] = SNR


def compute_normalized_importance_weight_std_dev(training_info):
    EMA_weight_std = torch.stack(training_info["EMA_weight_std"]).cpu().numpy()
    EMA_weight_mean = torch.stack(training_info["EMA_weight_mean"]).cpu().numpy()
    training_info["normalized_IW_std_dev"] = EMA_weight_std / EMA_weight_mean


def plot_loss(
    soc_solver_list,
    alg_numbers,
    cfg,
    variable="norm_sqd_diff",
    save_figure=False,
    file_name=None,
    plots_folder_name=None,
    set_ylims=False,
    ylim_inf=None,
    ylim_sup=None,
    title=None,
    use_fixed_colors=False,
):
    # linestyles and colors
    lss = ["-", "-.", ":", "--", "--", "-.", ":", "-", "--", "-.", ":", "-"] * 5
    cmap = mpl.cm.get_cmap("Set1") if cfg.method.plot_number in [5, 8] else mpl.cm.get_cmap("tab20")
    # cmap = mpl.cm.get_cmap("Set1")
    # cmap = mpl.cm.get_cmap("tab20")
    if use_fixed_colors:
        if cfg.method.plot_number == 5:
            colors_cmap = cmap([0, 1, 2, 3, 4, 6, 7, 8])
        elif cfg.method.plot_number == 0:
            colors_cmap = cmap([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
        elif cfg.method.plot_number == 1:
            colors_cmap = cmap([0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
        elif cfg.method.plot_number == 2:
            colors_cmap = cmap([1, 2, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17])
        elif cfg.method.plot_number == 3:
            # colors_cmap = cmap([1, 2, 3, 4])
            colors_cmap = cmap([1, 2])
        elif cfg.method.plot_number == 4:
            colors_cmap = cmap([5, 6, 11, 12, 13, 14, 15, 16, 17, 18])
        elif cfg.method.plot_number == 6:
            colors_cmap = cmap([3, 4])
        elif cfg.method.plot_number == 7:
            colors_cmap = cmap([8])
        elif cfg.method.plot_number == 8:
            colors_cmap = cmap([0, 1, 2, 3, 4, 6, 7])
        elif cfg.method.plot_number == 9:
            colors_cmap = cmap([0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16])
        else:
            colors_cmap = cmap([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])

    plt.figure()

    # alg_list = ['SOCM','UW_SOCM','UW_SOCM_sc','UW_SOCM_sc_2B','SOCM_const_M','SOCM_adjoint','continuous_adjoint',
    #             'discrete_adjoint','cross_entropy','log-variance','moment','variance','reinf','reinf_fr',
    #             'SOCM_cost','SOCM_cost_sc','SOCM_cost_sc_2B','reinf']

    algorithms = ""

    if variable == "norm_sqd_diff":
        ylabel = "Control error"
    elif variable == "EMA_norm_sqd_diff":
        ylabel = "Control " + r"$L^2$" + " error (EMA 0.01)"
    elif variable == "loss":
        ylabel = "Training loss"
    elif variable == "EMA_loss":
        ylabel = "Training loss (EMA 0.01)"
    elif variable == "grad_norm_sqd":
        ylabel = "Gradient norm"
    elif variable == "EMA_grad_norm_sqd":
        ylabel = "Gradient norm sqd. (EMA 0.01)"
    elif variable == "EMA_SNR":
        ylabel = "Gradient SNR (EMA 0.01)"
        first_point_SNR = 500
    elif variable == "control_objective_mean":
        ylabel = "Control objective"
        iterations_array = None
        first_point_control_loss = 1
    elif variable == "normalized_IW_std_dev":
        ylabel = "Importance weight std. dev. (normalized)"
    elif variable == "EMA_norm_sqd_diff_optimal":
        ylabel = "Control " + r"$L^2$" + " error (EMA)"

    num_plots = len(soc_solver_list)
    cmap_values = np.linspace(0, 1, num=num_plots)
    alg_numbers = torch.arange(len(soc_solver_list)) if cfg.method.plot_number in [5,8] else alg_numbers
    for k, soc_solver in enumerate(soc_solver_list):
        algorithm = soc_solver.algorithm
        training_info = soc_solver.training_info
        if use_fixed_colors:
            if cfg.method.plot_number in [5,8]:
                color = colors_cmap[alg_numbers[k]]
            else:
                color = colors_cmap[k]
        else:
            color = cmap(cmap_values[alg_numbers[k]])

        print(
            f"variable: {variable}, algorithm: {algorithm}, plots_folder_name: {plots_folder_name}"
        )
        if (
            variable == "control_objective_mean"
            and algorithm == "variance"
            and cfg.method.setting != "molecular_dynamics"
        ):
            continue
        if variable == "control_objective_mean":
            variable_array = (
                torch.stack(training_info[variable])
                .cpu()
                .numpy()[first_point_control_loss:]
            )
            iterations_array = np.array(training_info["control_objective_itr"])[
                first_point_control_loss:
            ]
            print(f"variable_array: {variable_array}")
            print(f"iterations_array: {iterations_array}")
        elif variable == "EMA_SNR":
            variable_array = training_info[variable][first_point_SNR:]
            iterations_array = first_point_SNR + np.linspace(
                0,
                len(variable_array),
                num=len(variable_array),
                endpoint=False,
                dtype=int,
            )
            print(f"iterations_array: {iterations_array}")
        elif variable == "normalized_IW_std_dev":
            variable_array = training_info[variable]
            iterations_array = np.linspace(
                0,
                len(variable_array),
                num=len(variable_array),
                endpoint=False,
                dtype=int,
            )
        elif variable == "EMA_norm_sqd_diff_optimal":
            variable_array = torch.stack(training_info[variable]).detach().cpu().numpy()
            iterations_array = np.array(training_info["optimal_itr"])
        else:
            variable_array = torch.stack(training_info[variable]).detach().cpu().numpy()
            iterations_array = np.linspace(
                0,
                len(variable_array),
                num=len(variable_array),
                endpoint=False,
                dtype=int,
            )
        print(
            f"variable_array.shape: {variable_array.shape}, iterations_array.shape: {iterations_array.shape}"
        )
        # print(f"np.mean(variable_array): {np.mean(variable_array)}")
        # print(f'variable_array[:20]: {variable_array[:20]}')
        # print(f'variable_array[-20:]: {variable_array[-20:]}')
        if variable == "control_objective_mean":
            plt.plot(
                iterations_array,
                variable_array,
                label=soc_solver.legend_name,
                color=color,
                linestyle=lss[alg_numbers[k]],
            )
        else:
            plt.semilogy(
                iterations_array,
                variable_array,
                label=soc_solver.legend_name,
                color=color,
                linestyle=lss[alg_numbers[k]],
            )
        if variable == "control_objective_mean":
            std_err_array = (
                torch.stack(training_info["control_objective_std_err"])
                .cpu()
                .numpy()[first_point_control_loss:]
            )
            bar_lower = variable_array - std_err_array
            bar_upper = variable_array + std_err_array
            plt.fill_between(
                iterations_array, bar_lower, bar_upper, alpha=0.3, color=color
            )
        algorithms = algorithms + "_" + soc_solver.algorithm
        # algorithms = algorithms + "_" + alg_list[k]

    plt.xlabel("Num. iterations")
    plt.ylabel(ylabel)
    if set_ylims:
        plt.ylim(ylim_inf, ylim_sup)

    if title is not None:
        plt.title(title)

    plt.legend(handletextpad=0.0)

    pplot()
    plt.tight_layout()

    if save_figure:
        figure_name = plots_folder_name + "/" + variable + algorithms + ".png"
        print(f"Figure saved at {figure_name}")
        plt.savefig(figure_name, bbox_inches="tight", pad_inches=0)


@hydra.main(version_base=None, config_path="configs", config_name="soc")
def main(cfg: DictConfig):
    print(cfg)

    folder_names, plots_folder_name, alg_names = get_folder_names_plots(cfg)
    file_names = get_file_names_plots(folder_names, last=True)
    print(f"file_names: {file_names}")
    # file_names = file_names

    if cfg.method.setting == "molecular_dynamics":
        legend_names = [
            "SOCM (ours)",
            "Adjoint",
            "Cross Entropy",
            "Log-Variance",
            "Moment",
            "Variance",
            "UW-SOCM",
        ]
    else:
        # legend_names = [
        #     "SOCM",
        #     "Unweighted SOCM",
        #     "Unweighted SOCM Diagonal",
        #     "Unweighted SOCM Diagonal 2B",
        #     "SOCM " + r"$M_t=I$",
        #     "SOCM-Adjoint",
        #     "Unweighted SOCM-Adjoint",
        #     "Adjoint",
        #     "Cross Entropy",
        #     "Log-Variance",
        #     "Moment",
        #     "Variance",
        #     "REINFORCE",
        #     "REINFORCE future rewards",
        #     "SOCM-Cost",
        #     "SOCM-Cost Diag.",
        #     "SOCM-Cost Diag. Non-diff.",
        #     "REINFORCE (unadjusted)",
        #     "SOCM-Work",
        #     "SOCM-Work Diag.",
        #     "SOCM-Work Diag. Non-diff.",
        # ]
        legend_names = [
            "SOCM",
            "UW-SOCM",
            "UW-SOCM Diag.",
            "UW-SOCM Diag. " + r"$M_t(T)=0$",
            "SOCM " + r"$M_t=I$",
            "SOCM-Adj",
            "Cont. Adjoint",
            "Disc. Adjoint",
            "Cross Entropy",
            "Log-Variance",
            "Moment",
            "Variance",
            "REINFORCE",
            "REINFORCE-FR",
            "SOCM-Cost",
            "SOCM-Cost Diag.",
            "SOCM-Cost Diag. " + r"$M_t(T)=0$",
            "REINFORCE (unadjusted)",
            "SOCM-Work",
            "SOCM-Work Diag.",
            "SOCM-Work Diag. " + r"$M_t(T)=0$",
        ]

    # plot_number = 3
    if cfg.method.plot_number == 0:
        # To show all algorithms, but UW_SOCM_diag_2B, REINFORCE (unadjusted)
        alg_list = [
            "SOCM",
            "UW_SOCM",
            "UW_SOCM_diag",
            "SOCM_const_M",
            "SOCM_adjoint",
            "continuous_adjoint",
            "discrete_adjoint",
            "cross_entropy",
            "log-variance",
            "moment",
            "variance",
            "reinf",
            "reinf_fr",
            "SOCM_cost",
            "SOCM_cost_diag",
            "SOCM_cost_diag_2B",
            "SOCM_work",
            "SOCM_work_diag",
            "SOCM_work_diag_2B",
        ]
    elif cfg.method.plot_number == 1:
        # To show all algorithms but UW_SOCM_diag_2B, SOCM_const_M, REINFORCE (unadjusted)
        alg_list = [
            "SOCM",
            "UW_SOCM",
            "UW_SOCM_diag",
            "SOCM_adjoint",
            "continuous_adjoint",
            "discrete_adjoint",
            "cross_entropy",
            "log-variance",
            "moment",
            "variance",
            "reinf",
            "reinf_fr",
            "SOCM_cost",
            "SOCM_cost_diag",
            "SOCM_cost_diag_2B",
            "SOCM_work",
            "SOCM_work_diag",
            "SOCM_work_diag_2B",
        ]
    elif cfg.method.plot_number == 2:
        # To show all scalable algorithms, no REINFORCE (unadjusted)
        # alg_list = [
        #     "UW_SOCM",
        #     "UW_SOCM_diag",
        #     "continuous_adjoint",
        #     "discrete_adjoint",
        #     "log-variance",
        #     "moment",
        #     "reinf",
        #     "reinf_fr",
        #     "SOCM_cost",
        #     "SOCM_cost_diag",
        #     "SOCM_cost_diag_2B",
        #     "SOCM_work",
        #     "SOCM_work_diag",
        #     "SOCM_work_diag_2B",
        # ]
        alg_list = [
            "UW_SOCM",
            "continuous_adjoint",
            "discrete_adjoint",
            "cross_entropy",
            "log-variance",
            "moment",
            "reinf",
            "reinf_fr",
            "SOCM_cost",
            "SOCM_work",
        ]
    elif cfg.method.plot_number == 3:
        #To show all UW_SOCM algorithms
        alg_list = [
            "UW_SOCM",
            "UW_SOCM_diag",
            # "UW_SOCM_diag_2B",
            # "continuous_adjoint",
        ]
    elif cfg.method.plot_number == 4:
        #To show all REINFORCE-like algorithms
        alg_list = [
            "continuous_adjoint",
            "discrete_adjoint",
            "reinf",
            "reinf_fr",
            "SOCM_cost",
            "SOCM_cost_diag",
            "SOCM_cost_diag_2B",
            "SOCM_work",
            "SOCM_work_diag",
            "SOCM_work_diag_2B",
        ]
    elif cfg.method.plot_number == 5:
        #To show all algorithms in the SOCM paper
        alg_list = [
            "SOCM",
            "SOCM_const_M",
            "SOCM_adjoint",
            "discrete_adjoint",
            "cross_entropy",
            "log-variance",
            "moment",
            "variance",
        ]
    elif cfg.method.plot_number == 6:
        #To show algorithms that don't work: UW_SOCM_diag_2B, REINFORCE (unadjusted)
        alg_list = [
            "UW_SOCM_diag_2B",
            "reinf_unadj",
        ]
    elif cfg.method.plot_number == 7:
        alg_list = [
            "variance",
        ]
    elif cfg.method.plot_number == 8:
        #To show all algorithms in the SOCM paper
        alg_list = [
            "SOCM",
            "SOCM_const_M",
            "SOCM_adjoint",
            "discrete_adjoint",
            "cross_entropy",
            "log-variance",
            "moment",
        ]
    elif cfg.method.plot_number == 9:
        # To show all algorithms but UW_SOCM_diag_2B, SOCM_const_M, REINFORCE (unadjusted)
        alg_list = [
            "SOCM",
            "UW_SOCM",
            "SOCM_adjoint",
            "continuous_adjoint",
            "discrete_adjoint",
            "cross_entropy",
            "log-variance",
            "moment",
            "variance",
            "reinf",
            "reinf_fr",
            "SOCM_cost",
            "SOCM_work",
        ]


    file_name = "last"
    set_ylims = False
    ylim_inf = None
    ylim_sup = None
    set_ylims_grad = False
    ylim_inf_grad = None
    ylim_sup_grad = None
    plot_norm_sqd_diff = True
    title = None
    use_fixed_colors = True

    plt.rcParams["figure.dpi"] = 300
    print("figsize", plt.rcParams["figure.figsize"])  # Prints the default figure size
    print("dpi", plt.rcParams["figure.dpi"])
    print(f"os.getcwd(): {os.getcwd()}")

    soc_solver_list = []
    alg_numbers = []
    for k, file_name in enumerate(file_names):
        print(f"file_name: {file_name}")
        print(f"os.path.exists(file_name): {os.path.exists(file_name)}")

        if os.path.exists(file_name) and alg_names[k] in alg_list:
            print(f"file_name exists")
            with open(file_name, "rb") as f:
                soc_solver = pickle.load(f)
                soc_solver.legend_name = legend_names[k]
                compute_SNR(soc_solver.training_info)
                compute_normalized_importance_weight_std_dev(soc_solver.training_info)
                soc_solver_list.append(soc_solver)
                alg_numbers.append(k)

    last_algorithm = {}
    if cfg.method.setting == "OU_quadratic_easy":
        last_algorithm["EMA_norm_sqd_diff"] = 9
        last_algorithm["EMA_grad_norm_sqd"] = 9
        last_algorithm["control_objective_mean"] = 7
        last_algorithm["EMA_norm_sqd_diff_optimal"] = 18
        title = r"Quadratic Ornstein Uhlenbeck, easy ($d=20$)"
    elif cfg.method.setting == "OU_quadratic_hard" and cfg.method.use_warm_start:
        last_algorithm["EMA_norm_sqd_diff"] = 8
        last_algorithm["EMA_grad_norm_sqd"] = 8
        last_algorithm["control_objective_mean"] = 7
        last_algorithm["EMA_norm_sqd_diff_optimal"] = 18
        title = r"Quadratic Ornstein Uhlenbeck, hard, warm start ($d=20$)"
    elif cfg.method.setting == "OU_quadratic_hard" and not cfg.method.use_warm_start:
        last_algorithm["EMA_norm_sqd_diff"] = 9
        last_algorithm["EMA_grad_norm_sqd"] = 7
        last_algorithm["control_objective_mean"] = 7
        last_algorithm["EMA_norm_sqd_diff_optimal"] = 18
        title = r"Quadratic Ornstein Uhlenbeck, hard, no warm start ($d=20$)"
    elif cfg.method.setting == "OU_linear":
        last_algorithm["EMA_norm_sqd_diff"] = 9
        last_algorithm["EMA_grad_norm_sqd"] = 9
        last_algorithm["control_objective_mean"] = 7
        last_algorithm["EMA_norm_sqd_diff_optimal"] = 18
        title = r"Linear Ornstein Uhlenbeck ($d=10$)"
    elif cfg.method.setting == "double_well":
        last_algorithm["EMA_norm_sqd_diff"] = 9
        last_algorithm["EMA_grad_norm_sqd"] = 9
        last_algorithm["control_objective_mean"] = 7
        last_algorithm["EMA_norm_sqd_diff_optimal"] = 17
        title = r"Double Well ($d=10$)"
    elif cfg.method.setting == "molecular_dynamics":
        last_algorithm["EMA_grad_norm_sqd"] = 8
        last_algorithm["control_objective_mean"] = 8
        plot_norm_sqd_diff = False
        title = r"Molecular dynamics ($d=1$)"

    if cfg.method.setting == "OU_quadratic_hard" and not cfg.method.use_warm_start:
        set_ylims = True
        ylim_inf = 0.01
        ylim_sup = 1000
        set_ylims_grad = True
        ylim_inf_grad = 1
        ylim_sup_grad = 10000000

    if len(soc_solver_list) > 0:
        os.makedirs(plots_folder_name, exist_ok=True)

        if plot_norm_sqd_diff:
            plot_loss(
                soc_solver_list[: last_algorithm["EMA_norm_sqd_diff"]],
                alg_numbers[: last_algorithm["EMA_norm_sqd_diff"]],
                cfg,
                variable="EMA_norm_sqd_diff",
                save_figure=True,
                plots_folder_name=plots_folder_name,
                file_name=file_name,
                set_ylims=set_ylims,
                ylim_inf=ylim_inf,
                ylim_sup=ylim_sup,
                title=title,
                use_fixed_colors=use_fixed_colors,
            )
        plot_loss(
            soc_solver_list[: last_algorithm["EMA_grad_norm_sqd"]],
            alg_numbers[: last_algorithm["EMA_grad_norm_sqd"]],
            cfg,
            variable="EMA_grad_norm_sqd",
            save_figure=True,
            plots_folder_name=plots_folder_name,
            file_name=file_name,
            set_ylims=set_ylims_grad,
            ylim_inf=ylim_inf_grad,
            ylim_sup=ylim_sup_grad,
            title=title,
            use_fixed_colors=use_fixed_colors,
        )
        plot_loss(
            soc_solver_list[:3],
            alg_numbers[:3],
            cfg,
            variable="EMA_loss",
            save_figure=True,
            plots_folder_name=plots_folder_name,
            file_name=file_name,
            title=title,
            use_fixed_colors=use_fixed_colors,
        )
        plot_loss(
            soc_solver_list[: last_algorithm["control_objective_mean"]],
            alg_numbers[: last_algorithm["control_objective_mean"]],
            cfg,
            variable="control_objective_mean",
            save_figure=True,
            plots_folder_name=plots_folder_name,
            file_name=file_name,
            title=title,
            use_fixed_colors=use_fixed_colors,
        )
        plot_loss(
            soc_solver_list,
            alg_numbers,
            cfg,
            variable="EMA_SNR",
            save_figure=True,
            plots_folder_name=plots_folder_name,
            file_name=file_name,
            title=title,
            use_fixed_colors=use_fixed_colors,
        )
        plot_loss(
            soc_solver_list,
            alg_numbers,
            cfg,
            variable="normalized_IW_std_dev",
            save_figure=True,
            plots_folder_name=plots_folder_name,
            file_name=file_name,
            title=title,
            use_fixed_colors=use_fixed_colors,
        )
        if plot_norm_sqd_diff:
            plot_loss(
                soc_solver_list[: last_algorithm["EMA_norm_sqd_diff_optimal"]],
                alg_numbers[: last_algorithm["EMA_norm_sqd_diff_optimal"]],
                cfg,
                variable="EMA_norm_sqd_diff_optimal",
                save_figure=True,
                plots_folder_name=plots_folder_name,
                file_name=file_name,
                title=title,
                use_fixed_colors=use_fixed_colors,
            )

    else:
        print(f"file_name does not exist")


if __name__ == "__main__":
    main()
