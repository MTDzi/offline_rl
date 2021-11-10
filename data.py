from __future__ import annotations

from typing import Dict, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

import torch.utils.data as data

from racing_utils import (
    rotate_into_map_coord,
    closest_point_idx,
    cyclic_slice,
    determine_direction_of_bound,
    reward_for_getting_far,
    penalty_from_bounds,
)


class RaceDataset(data.Dataset):

    def __init__(
            self,
            num_steps_ahead_traj: int, num_steps_ahead_act: int, num_steps_ahead_bound: int,
            num_steps_centerline: int,
            directory: str,
            centerline_decimation: int = 10,
            prob_flip: float = 0.5,
            full_state_for_trajectory: bool = False,
        ):
        """
        Inputs:
            size - Number of data points we want to generate
        """
        super().__init__()
        
        self.num_steps_ahead_traj = num_steps_ahead_traj
        self.num_steps_ahead_act = num_steps_ahead_act
        self.num_steps_ahead_bound = num_steps_ahead_bound
        self.num_steps_centerline = num_steps_centerline
        self.full_state_for_trajectory = full_state_for_trajectory

        self.size = 0
        self.directory = Path(directory)
        self.bounds = []
        for csv_file in ['interior.csv', 'exterior.csv']:
            bound = pd.read_csv(self.directory.parent / csv_file, header=None).values
            self.bounds.append(bound)
        self.reversed_bounds = [bound[::-1] for bound in self.bounds]
        self.prob_flip = prob_flip
            
        self.fetch_data(centerline_decimation)

    def fetch_data(self, centerline_decimation):
        self.traj_data = []
        self.cumul_sizes = []
        for filename in self.directory.glob('*.pkl'):
            one_piece_of_data = {}
            
            unpickel = pd.read_pickle(filename)
            additional_data = unpickel['additional_data']

            one_race = unpickel['data']
            self.size += len(one_race) - self.num_steps_ahead_traj - 1
            self.cumul_sizes.append(self.size)

            # First, identify the indices of the centerline points that are closest to the positions
            centerline = additional_data['centerline'][::centerline_decimation]
            one_piece_of_data['closest_centerline_idx'] = closest_point_idx(np.stack(one_race['position'].values), centerline)
            one_piece_of_data['centerline'] = centerline

            # While we're at it, let's fetch the rest of the additional data
            one_piece_of_data['lookahead_distance'] = additional_data['lookahead_distance']
            one_piece_of_data['speed_setpoint'] = additional_data['speed_setpoint']
            one_piece_of_data['tire_force_max'] = additional_data['tire_force_max']

            # Now, determine the direction of the bounds
            start_position, end_position = one_race.loc[[0, self.num_steps_ahead_traj], 'position']
            directions = []
            for bound in self.bounds:
                direction = determine_direction_of_bound(bound, start_position, end_position)
                directions.append(direction)
            one_piece_of_data['bound_directions'] = directions
            
            # We'll need the yaws to calcuate the rotations
            one_piece_of_data['yaw'] = one_race['yaw'].values
            
            # States
            one_piece_of_data['position'] = np.stack(one_race['position'].values)
            one_piece_of_data['v_x'] = np.stack(one_race['v_x'].values)
            one_piece_of_data['v_y'] = np.stack(one_race['v_y'].values)
            one_piece_of_data['omega'] = np.stack(one_race['omega'].values)
            
            # Actuators
            one_piece_of_data['delta'] = np.stack(one_race['delta'].values)
            one_piece_of_data['speed_actuator'] = np.stack(one_race['speed_actuator'].values)
            
            self.traj_data.append(one_piece_of_data)
        
    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx) -> Tuple[Dict, Dict]:
        which_race = 0
        idx_shift = 0
        for cum_size in self.cumul_sizes:
            if idx >= cum_size:
                which_race += 1
                idx_shift = cum_size
            else:
                break
                
        one_race = self.traj_data[which_race]
        idx -= idx_shift
        future_slice_traj = slice(idx + 1, idx + self.num_steps_ahead_traj)
        future_slice_act = slice(idx + 1, idx + self.num_steps_ahead_act)

        # Controller's parameters (INPUT FEATURES)
        contr_params = np.r_[one_race['lookahead_distance'], one_race['speed_setpoint'], one_race['tire_force_max']]
        
        # Car's state (INPUT FEATURES)
        position = one_race['position'][idx]
        yaw = one_race['yaw'][idx]
        v_x = one_race['v_x'][idx]
        v_y = one_race['v_y'][idx]
        omega = one_race['omega'][idx]
        # Car's current actuators (INPUT FEATURES)
        delta = one_race['delta'][idx]
        speed_actuator = one_race['speed_actuator'][idx]

        # Centerline (INPUT FEATURES)
        centerline_ahead = cyclic_slice(one_race['centerline'], one_race['closest_centerline_idx'][idx], self.num_steps_centerline)
        centerline_ahead = rotate_into_map_coord(centerline_ahead - position, -yaw)
        
        # Car's states (TARGET)
        future_positions = one_race['position'][future_slice_traj]
        future_positions = rotate_into_map_coord(future_positions - position, -yaw)
        if self.full_state_for_trajectory:
            future_v_xes = one_race['v_x'][future_slice_traj]
            future_v_ys = one_race['v_y'][future_slice_traj]
            future_omegas = one_race['omega'][future_slice_traj]
            future_yaws = one_race['yaw'][future_slice_traj] - yaw
            future_yaws[future_yaws > np.pi] -= (2 * np.pi)
            future_yaws[future_yaws < -np.pi] += (2 * np.pi)

        # Car's actuators (TARGET)
        future_deltas = one_race['delta'][future_slice_act]
        future_speed_actuators = one_race['speed_actuator'][future_slice_act]
        
        # Bounds (INPUT FEATURES)
        bound_slices = []
        for i in range(2):
            # "2" as in: there are two bounds: the interior, and exterior
            bound_direction = one_race['bound_directions'][i]
            if bound_direction == 1:
                bound = self.bounds[i]
            else:
                bound = self.reversed_bounds[i]
                
            closest_idx = closest_point_idx(position, bound)
            bound_slice = cyclic_slice(bound, closest_idx, self.num_steps_ahead_bound)
            bound_slice = rotate_into_map_coord(bound_slice - position, -yaw)
            bound_slices.append(bound_slice)

        # # Reward
        # reward = reward_for_getting_far(future_positions, centerline_ahead)

        # # Penalty
        # penalty = penalty_from_bounds(future_positions, *bound_slices)

        # # Cap on the penalty
        # if penalty > reward:
        #     penalty = reward
        
        # reward_and_penalty = np.array([reward, penalty])

        # A tiny bit of augmentation
        if np.random.uniform() < self.prob_flip:
            delta *= -1
            omega *= -1
            v_y *= -1
            bound_slices[0][:, 1] *= -1
            bound_slices[1][:, 1] *= -1
            centerline_ahead[:, 1] *= -1
            future_deltas = -future_deltas
            future_positions[:, 1] *= -1
            if self.full_state_for_trajectory:
                future_omegas = -future_omegas
                future_yaws = -future_yaws
                future_v_ys = -future_v_ys
        
        # Final preperations for returning the features and the targets 
        car_state = np.r_[v_x, v_y, omega, delta, speed_actuator]
        left_bound = bound_slices[0].flatten()
        right_bound = bound_slices[1].flatten()
        future_speeds_and_deltas = np.r_[future_speed_actuators, future_deltas]
        trajectory = future_positions.flatten()
        if self.full_state_for_trajectory:
            trajectory = np.r_[trajectory, future_v_xes, future_v_ys, future_omegas, future_yaws]
        
        features = {
            # For predicting the trajectory
            'state': car_state,
            'contr_params': contr_params,
            
            # For calculating the reward
            'left_bound': left_bound,
            'right_bound': right_bound,

            # For both
            'centerline': centerline_ahead.flatten(),
        }

        targets = {
            'trajectory': trajectory,
            'speeds_and_deltas': future_speeds_and_deltas,
            # 'reward_and_penalty': reward_and_penalty,
        }

        return features, targets
        

# class OneActuatorDataset(data.Dataset):
#     STATE_COLNAMES = (
#         'position',
#         'velocity',
#         'yaw',
#         'omega'
#     )
#     ACTUATOR_COLNAMES = (
#         'delta',
#         'speed_actuator',
#     )

#     def __init__(self, directory: str, prob_flip: float = 0.5):
#         """
#         Inputs:
#             directory - The directory with .pkl files that are going to be unpacked
#             prob_flip - Probability of doing a vertical flip of the data
#         """
#         super().__init__()
#         self.size = 0
#         self.directory = Path(directory)
#         self.prob_flip = prob_flip
            
#         self.fetch_data()

#     def fetch_data(self):
#         self.traj_data = []
#         for filename in self.directory.glob('*.pkl'):            
#             one_race = pd.read_pickle(filename)

#             self.traj_data.append(one_race[self.STATE_COLNAMES + self.ACTUATOR_COLNAMES])
            
#     def __len__(self) -> int:
#         return self.size

#     def __getitem__(self, idx) -> Tuple[Dict, Dict]:
                
#         if np.random.uniform() < self.prob_flip:
#             delta = -delta
#             omega = -omega
#             bound_slices[0][:, 1] = -bound_slices[0][:, 1]
#             bound_slices[1][:, 1] = -bound_slices[1][:, 1]
#             future_deltas = -future_deltas
#             future_positions[:, 1] = -future_positions[:, 1]
        
#         car_state = np.r_[v_x, omega, delta, speed_actuator]
#         left_bound = bound_slices[0].flatten()
#         right_bound = bound_slices[1].flatten()

#         future_actuators = np.r_[future_deltas, future_speed_actuators]
        
#         return (
#             {'state': car_state, 'left_bound': left_bound, 'right_bound': right_bound},
#             {'trajectory': future_positions.flatten(), 'actuators': future_actuators},
#         )
        