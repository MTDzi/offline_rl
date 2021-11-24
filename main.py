import argparse
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data

from racing_utils.torch_related import TensorStandardScaler
from racing_utils.models import get_omniward_model
from racing_utils.inference import GradientDriver

from data import RaceDataset
from training import one_pass_through_data, process_batch, Phase, plot_losses_vs_epochs


# Reproducible
# torch.backends.cudnn.determinstic = True
# torch.backends.cudnn.benchmark = False
#
# Or fast
torch.backends.cudnn.benchmark = True

# These parameters seemed so irrelevant (i.e. I no longer see the point in optimizing the model with them)
#  that I didn't want to make them specifiable via argparse
TRAJECTORY_GAMMA = 0.99999
ACTUATORS_GAMMA = 0.8

# This impacts the plots, and then the inference, but not the training itself
PENALTY_SIGMA = 0.3




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-nt', '--num-steps-traj', default=150,
                        help='Number of trajectory steps predicted by the model')
    parser.add_argument('-na', '--num-steps-act', default=10,
                        help='Number of actuator steps predicted by the model')
    parser.add_argument('-nb', '--num-steps-bound', default=150,
                        help='Number of bound steps used for calculating the penalty')
    parser.add_argument('-nc', '--num-steps-cent', default=300,
                        help='Number of centerline steps used by the model as input')
    parser.add_argument('-cd', '--centerline-decimation', default=1,
                        help='Stride step used to decimate the centerline (1 ==> no reduction in size)')
    parser.add_argument('-s', '--dataset-suffix', default='_tiny',
                        help='Suffix of the dataset used for training / validating the model')
    parser.add_argument('-d', '--device', default='cuda:0' if torch.cuda.is_available() else 'cpu',
                        help='Device used byt torch')
    parser.add_argument('-bs', '--batch-size', default=512,
                        help='Batch size used by the data loader')
    parser.add_argument('-fp', '--flip-prob', default=0.5,
                        help='Probability of flipping the observations in the training set (mini-augmentation')
    parser.add_argument('-nw', '--num-workers', default=4,
                        help='Number of workers used by the data loader (passed to Dataloader.__init__)')
    parser.add_argument('-nl', '--num-layers', default=3,
                        help='Number of layers in the model')
    parser.add_argument('-wr', '--width-reduction', default=2,
                        help='A parameter determining the number of neurons in all the layers')
    parser.add_argument('-pd', '--plots-directory', default='./plots',
                        help='Directory to which all plots are going to be saved to')
    
    
    args = parser.parse_args()

    # First, make sure directory for plots exists
    plots_directory = Path(args.plots_directory)
    plots_directory.mkdir(exist_ok=True)

    # Fetch the training dataset
    train_dataset = RaceDataset(
        args.num_steps_traj,
        args.num_steps_act,
        args.num_steps_bound,
        args.num_steps_cent,
        f'./data/train{args.dataset_suffix}',
        args.centerline_decimation,
        flip_prob=args.flip_prob,
    )

    train_loader = data.DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Initilize the scalers
    features, targets = train_dataset[0]
    features_scalers = {key: TensorStandardScaler(args.device) for key in features.keys()}
    targets_scalers = {key: TensorStandardScaler(args.device) for key in targets.keys()}

    # Do a partial fit of the scalers for both the input features, and the targets
    for features_batch, targets_batch in train_loader:
        for feature_name in features:
            # TODO: not all features need rescaling, in particular: left_bound, right_bound
            features_scalers[feature_name].partial_fit(features_batch[feature_name])

        for target_name in targets:
            targets_scalers[target_name].partial_fit(targets_batch[target_name])

    # Move the numpy structures into a torch.tensor
    for feature_name in features:
        features_scalers[feature_name].tensorfy()
    for target_name in targets:
        targets_scalers[target_name].tensorfy()

    # Now for the model
    omniward_model = get_omniward_model(
        args.num_layers,
        args.width_reduction,
        features,
        targets,
        args.device,
    )

    # Now for the validation dataset
    valid_dataset = RaceDataset(
        args.num_steps_traj,
        args.num_steps_act,
        args.num_steps_bound,
        args.num_steps_cent,
        f'./data/valid{args.dataset_suffix}',
        args.centerline_decimation,
        flip_prob=args.flip_prob,
    )

    # And the data loaders
    train_loader, valid_loader = [
        data.DataLoader(dataset, args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        for dataset in [train_dataset, valid_dataset]
    ]

    # The following gammas are needed for the loss function.
    #  Their purpose is to put more weight on: points that are *further* in the trajectory,
    #  and points that are *closer* in the actuators.
    trajectory_size = len(targets['trajectory'])
    actuators_size = len(targets['speeds_and_deltas'])
    trajectory_gammas = (TRAJECTORY_GAMMA ** np.r_[np.arange(trajectory_size // 2), np.arange(trajectory_size // 2)])
    trajectory_gammas = torch.from_numpy(trajectory_gammas[::-1].copy()).to(args.device)
    actuators_gammas = ACTUATORS_GAMMA ** np.r_[np.arange(actuators_size // 2), np.arange(actuators_size // 2)]
    actuators_gammas = torch.from_numpy(actuators_gammas).to(args.device)

    # TODO: this is an important parameter and allows for a bit more flexibility than the learning rate scheduler.
    #  However, as it turned out, I no longer need that flexibility. So this should either be re-written as a callback
    #  (preferably with PyTorch Lightning), or moved to a config file (with Pydantic), or both.
    optimizers = (
        [5, torch.optim.Adam(omniward_model.parameters(), lr=1e-3)],
        [15, torch.optim.Adam(omniward_model.parameters(), lr=1e-4)],
        [5, torch.optim.Adam(omniward_model.parameters(), lr=1e-5)],
    )

    # Bookkeeping for plots
    train_traj_mses_for_plot = []
    train_act_mses_for_plot = []
    train_progress_mses_for_plot = []
    train_penalty_mses_for_plot = []

    valid_traj_mses_for_plot = []
    valid_act_mses_for_plot = []
    valid_progress_mses_for_plot = []
    valid_penalty_mses_for_plot = []

    epoch = 0
    for num_epochs_per_optimizer_round, optimizer in optimizers:
        print(f'Optimizer: {optimizer}')
        for _ in range(num_epochs_per_optimizer_round):
            
            #            #
            #  Training  #
            #            #
            one_pass_through_data(
                model=omniward_model,
                optimizer=optimizer,
                phase=Phase.train,
                data_loader=train_loader,
                features_scalers=features_scalers,
                targets_scalers=targets_scalers,
                trajectory_gammas=trajectory_gammas,
                actuators_gammas=actuators_gammas,
                penalty_sigma=PENALTY_SIGMA,
                traj_mses_for_plot=train_traj_mses_for_plot,
                act_mses_for_plot=train_act_mses_for_plot,
                progress_mses_for_plot=train_progress_mses_for_plot,
                penalty_mses_for_plot=train_penalty_mses_for_plot,
                epoch=epoch,
                device=args.device,
            )
                
            #              #
            #  Validation  #
            #              #
            with torch.inference_mode():
                one_pass_through_data(
                    model=omniward_model,
                    optimizer=None,
                    phase=Phase.valid,
                    data_loader=valid_loader,
                    features_scalers=features_scalers,
                    targets_scalers=targets_scalers,
                    trajectory_gammas=trajectory_gammas,
                    actuators_gammas=actuators_gammas,
                    penalty_sigma=PENALTY_SIGMA,
                    traj_mses_for_plot=valid_traj_mses_for_plot,
                    act_mses_for_plot=valid_act_mses_for_plot,
                    progress_mses_for_plot=valid_progress_mses_for_plot,
                    penalty_mses_for_plot=valid_penalty_mses_for_plot,
                    epoch=epoch,
                    device=args.device,
                )

            epoch += 1
            print()

        # Make scatterplots showing the dependency between ground truth and predicted
        #  panelty and progress losses
        features_batch, targets_batch = next(iter(valid_loader))
        plots_subdirectory = plots_directory / f'epoch={epoch}'
        plots_subdirectory.mkdir(exist_ok=True)
        with torch.inference_mode():
            traj_mse, act_mse, progress_mse, penalty_mse = process_batch(
                omniward_model,
                features_scalers,
                targets_scalers,
                features_batch,
                targets_batch,
                trajectory_gammas,
                actuators_gammas,
                PENALTY_SIGMA,
                args.device,
                make_plots=True,
                plots_directory=plots_subdirectory,
            )

    plot_losses_vs_epochs(
        plots_directory=plots_directory,
        train_traj_mses=train_traj_mses_for_plot,
        valid_traj_mses=valid_traj_mses_for_plot,
        train_act_mses=train_act_mses_for_plot,
        valid_act_mses=valid_act_mses_for_plot,
        train_progress_mses=train_progress_mses_for_plot,
        valid_progress_mses=valid_progress_mses_for_plot,
        train_penalty_mses=train_penalty_mses_for_plot,
        valid_penalty_mses=valid_penalty_mses_for_plot,
    )

    whichever_traj_data = valid_dataset.traj_data[0]
    centerline = whichever_traj_data['centerline']
    left_bound_direction = whichever_traj_data['bound_directions'][0]
    left_bound = valid_dataset.bounds[0][::left_bound_direction]
    right_bound_direction = whichever_traj_data['bound_directions'][1]
    right_bound = valid_dataset.bounds[1][::right_bound_direction]

    contr_params_limits = train_dataset.determine_limits()
    lookahead_distance = (contr_params_limits[0][0] + contr_params_limits[0][1]) / 2
    speed_setpoint = (contr_params_limits[1][0] + contr_params_limits[1][1]) / 2
    tire_force_max = (contr_params_limits[2][0] + contr_params_limits[2][1]) / 2
    print(f'contr_param_limits = \n{contr_params_limits}')

    grad_driver = GradientDriver(
        centerline=centerline,
        num_steps_centerline=args.num_steps_cent,

        left_bound=valid_dataset.bounds[0][::left_bound_direction],
        right_bound=valid_dataset.bounds[1],
        num_steps_ahead_bound=args.num_steps_bound,

        # Controller-related
        init_contr_params=np.r_[lookahead_distance, speed_setpoint, tire_force_max],

        # Model-related
        omniward_model=omniward_model.to('cpu'),
        features_scalers=features_scalers,
        targets_scalers=targets_scalers,

        # Gradient-related
        eta=0.1,
        num_steps_for_grad=4,
        penalty_sigma=PENALTY_SIGMA,
        penalty_scale_coeff=-0.9,
        contr_params_limits=contr_params_limits,

        device='cpu',
    )
