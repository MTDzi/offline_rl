"""
Contains functions needed for training, but also for validating, a model.
"""

from typing import Dict, Tuple, List, Optional, Union, Sequence
from pathlib import Path
from enum import Enum, auto

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.utils.data as data

from racing_utils.torch_related import TensorStandardScaler, calc_progress_and_penalty, scale_batch_and_to_device


class Phase(Enum):
    train = auto()
    valid = auto()


def _traj_act_mse(
        *,
        traj_gammas: torch.Tensor,
        act_gammas: torch.Tensor,
        traj: torch.Tensor,
        traj_pred: torch.Tensor,
        act: torch.Tensor,
        act_pred: torch.Tensor,
):
    return (
        (traj_gammas * (traj - traj_pred) ** 2).mean(),
        (act_gammas * (act - act_pred) ** 2).mean(),
    )


def process_batch(
        model: torch.nn.Module,
        features_scalers: Dict[str, TensorStandardScaler],
        targets_scalers: Dict[str, TensorStandardScaler],
        features_batch: Dict[str, torch.Tensor],
        targets_batch: Dict[str, torch.Tensor],
        traj_gammas: np.array,
        act_gammas: np.array,
        penalty_sigma: float,
        device: str,
        make_plots: bool = False,
        plots_directory: Optional[Union[str, Path]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
    """
    Takes one batch of features and targets, calculates progress and penalty, scales features and targets, produces
    predictions using the model, and calculates the loss for the trajectories and actuators (which is used for backprop),
    but also loss for the penalty and penalty.
    """

    centerline = features_batch['centerline'].to(device)
    left_bound = features_batch['left_bound'].to(device)
    right_bound = features_batch['right_bound'].to(device)
    trajectory = targets_batch['trajectory'].to(device)

    progress, penalty = calc_progress_and_penalty(trajectory, centerline, left_bound, right_bound, penalty_sigma=penalty_sigma)

    features_batch, targets_batch = scale_batch_and_to_device(device, features_scalers, targets_scalers, features_batch, targets_batch)

    preds = model(**features_batch)
    trajectory_pred, actuators_pred = preds['trajectory_pred'], preds['actuators_pred']

    trajectory = targets_batch['trajectory'].squeeze(dim=1)
    actuators = targets_batch['speeds_and_deltas'].squeeze(dim=1)

    traj_mse, act_mse = _traj_act_mse(
        traj_gammas=traj_gammas,
        act_gammas=act_gammas,
        traj=trajectory,
        traj_pred=trajectory_pred,
        act=actuators,
        act_pred=actuators_pred,
    )

    trajectory_pred = targets_scalers['trajectory'].inverse_transform(trajectory_pred)
    progress_pred, penalty_pred = calc_progress_and_penalty(
        trajectory_pred,
        centerline,
        left_bound,
        right_bound,
        penalty_sigma=penalty_sigma,
    )
    progress_mse = float(((progress - progress_pred) ** 2).mean())
    penalty_mse = float(((penalty - penalty_pred) ** 2).mean())

    if make_plots:
        plt.scatter(penalty.detach().cpu().numpy(), penalty_pred.detach().cpu().numpy(), alpha=0.15)
        plt.gca().set_yscale('log')
        plt.gca().set_xscale('log')
        if plots_directory is None:
            plt.show()
        else:
            plt.savefig(f'{plots_directory}/scatter_penalty.png')
            plt.clf()

        plt.scatter(progress.detach().cpu().numpy(), progress_pred.detach().cpu().numpy(), alpha=0.15)
        if plots_directory is None:
            plt.show()
        else:
            plt.savefig(f'{plots_directory}/scatter_progress.png')
            plt.clf()

    return traj_mse, act_mse, progress_mse, penalty_mse


def one_pass_through_data(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        phase: Phase,
        data_loader: data.DataLoader,
        features_scalers: Dict[str, TensorStandardScaler],
        targets_scalers: Dict[str, TensorStandardScaler],
        trajectory_gammas: torch.Tensor,
        actuators_gammas: torch.Tensor,
        penalty_sigma: float,
        traj_mses_for_plot: List[float],
        act_mses_for_plot: List[float],
        progress_mses_for_plot: List[float],
        penalty_mses_for_plot: List[float],
        epoch: int,
        device: str,
):
    """
    Goes through the entire dataset, processing each batch, and storing the loss for each of the
     predicted components.
    """
    cumul_traj_mse = 0
    cumul_act_mse = 0
    cumul_progress_mse = 0
    cumul_penalty_mse = 0

    if phase is Phase.train:
        model.train()
    else:
        model.eval()

    # now for the pass through data itself
    for features_batch, targets_batch in data_loader:
            
        traj_mse, act_mse, progress_mse, penalty_mse = process_batch(
            model,
            features_scalers,
            targets_scalers,
            features_batch,
            targets_batch,
            trajectory_gammas,
            actuators_gammas,
            penalty_sigma,
            device,
        )

        if phase is Phase.train: 
            optimizer.zero_grad()
            (traj_mse + act_mse).backward()
            optimizer.step()

        cumul_traj_mse += float(traj_mse)
        cumul_act_mse += float(act_mse)
        cumul_progress_mse += progress_mse
        cumul_penalty_mse += penalty_mse

    avg_traj_mse = cumul_traj_mse / len(data_loader)
    avg_act_mse = cumul_act_mse / len(data_loader)
    avg_progress_mse = cumul_progress_mse / len(data_loader)
    avg_penalty_mse = cumul_penalty_mse / len(data_loader)

    traj_mses_for_plot.append(avg_traj_mse)
    act_mses_for_plot.append(avg_act_mse)
    progress_mses_for_plot.append(avg_progress_mse)
    penalty_mses_for_plot.append(avg_penalty_mse)
    print(
        f'{epoch}:'
        f' MSE({phase.name}_traj): {avg_traj_mse:.3f},'
        f' MSE({phase.name}_act): {avg_act_mse:.3f},'
        f' MSE({phase.name}_progress): {avg_progress_mse:.3f},'
        f' MSE({phase.name}_penalty): {avg_penalty_mse:.3f}'
    )


def plot_losses_vs_epochs(
        plots_directory: Optional[Union[str, Path]] = None,
        *,
        train_traj_mses: Sequence[float],
        valid_traj_mses: Sequence[float],
        train_act_mses: Sequence[float],
        valid_act_mses: Sequence[float],
        train_progress_mses: Sequence[float],
        valid_progress_mses: Sequence[float],
        train_penalty_mses: Sequence[float],
        valid_penalty_mses: Sequence[float],
):
    """
    Makes four plots with mean squared errors (MSEs) in one row, with their Y axes
     aligned st. the value 0 is the lowest, thus allowing for relative comparisons of the
     loss values.
    """
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(32, 6))

    ax1.plot(train_traj_mses, label='train_traj_mse')
    ax1.plot(valid_traj_mses, label='valid_traj_mse')
    ax1.legend()
    ax1.set_ylim(0, None)

    ax2.plot(train_act_mses, label='train_act_mse')
    ax2.plot(valid_act_mses, label='valid_act_mse')
    ax2.legend()
    ax2.set_ylim(0, None)

    ax3.plot(train_progress_mses, label='train_progress_mse')
    ax3.plot(valid_progress_mses, label='valid_progress_mse')
    ax3.legend()
    ax3.set_ylim(0, None)

    ax4.plot(train_penalty_mses, label='train_penalty_mse')
    ax4.plot(valid_penalty_mses, label='valid_penalty_mse')
    ax4.legend()
    ax4.set_ylim(0, None)

    if plots_directory is None:
        plt.show()
    else:
        plt.savefig(f'{plots_directory}/mses.png')
        plt.clf()
