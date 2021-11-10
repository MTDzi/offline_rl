from typing import Dict, Tuple

import numpy as np

import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler


class OmniwardModel(nn.Module):

    FULL_TRAJ_DIM = 6
    YAW_0 = torch.tensor([0.0])

    def __init__(
            self, state_size, controller_params_size, centerline_size,
            centerline_encoder_sizes, middle_sizes, output_sizes,
            trajectory_size, actuators_size
    ):
        super().__init__()

        sizes = [centerline_size] + centerline_encoder_sizes
        self.centerline_encoder = self._stack_layers(sizes, nn.Linear, nn.SiLU)

        sizes = [state_size + centerline_encoder_sizes[-1] + controller_params_size] + middle_sizes
        self.middle_encoder = self._stack_layers(sizes, nn.Linear, nn.SiLU)

        sizes = [middle_sizes[-1]] + output_sizes + [actuators_size]
        self.actuators_predictor = self._stack_layers(sizes, nn.Linear, nn.SiLU, last_linear=True)

        sizes = [middle_sizes[-1]] + output_sizes + [trajectory_size]
        self.trajectory_predictor = self._stack_layers(sizes, nn.Linear, nn.SiLU, last_linear=True)

    @staticmethod
    def _stack_layers(sizes, layer, activation, last_linear=False):
        layers = []
        for enc_size_in, enc_size_out in zip(sizes[:-1], sizes[1:]):
            layers.append(layer(enc_size_in, enc_size_out))
            layers.append(activation(inplace=True))
        if last_linear:
            layers.pop()
        return nn.Sequential(*layers)
        
    def forward(self, state, contr_params, centerline, left_bound, right_bound):
        centerline_enc = self.centerline_encoder(centerline)

        centerline_for_middle = centerline_enc.clone()
        middle = self.middle_encoder(torch.cat([state, contr_params, centerline_for_middle], axis=1))
        middle_for_trajectory = middle.clone()
        middle_for_actuators = middle.clone()
        trajectory_pred = self.trajectory_predictor(middle_for_trajectory)
        actuators_pred = self.actuators_predictor(middle_for_actuators)

        return {
            'trajectory_pred': trajectory_pred,
            'actuators_pred': actuators_pred,
        }


class TrajectoryAndActuatorsForExport:

    def __init__(self, model: nn.Module, features_scalers: Dict[str, StandardScaler], targets_scalers: Dict[str, StandardScaler]):
        self.model = model
        self.features_scalers = features_scalers
        self.targets_scalers = targets_scalers

    def predict(self, bounds: Tuple[np.ndarray, np.ndarray], features: Dict[str, torch.Tensor]) -> Tuple[float, float]:
        with torch.inference_mode:
            preds = self.model(**features)
            actuators_pred = preds['actuators_pred']

        half = len(actuators_pred)
        delta, speed = actuators_pred[0], actuators_pred[half]


# class OneActuatorModel(nn.Module):

#     def __init__(self, num_layers: int = 3, num_neurons: int = 30):
#         super().__init__()
        
#         layers = []
#         for enc_size_in, enc_size_out in zip(bound_encoder_sizes[:-1], bound_encoder_sizes[1:]):
#             layers.append(nn.Linear(enc_size_in, enc_size_out))
#             bound_enc_layers.append(nn.SiLU(inplace=True))
        
#         self.bound_encoder = nn.Sequential(*bound_enc_layers)
            
#         middle_layers = []
#         middle_sizes = [state_size + 2 * enc_size_out] + middle_sizes
#         for in_size, out_size in zip(middle_sizes[:-1], middle_sizes[1:]):
#             middle_layers.append(nn.Linear(in_size, out_size))
#             middle_layers.append(nn.SiLU(inplace=True))
            
#         self.middle_part = nn.Sequential(*middle_layers)
            
#         trajectory_layers = []
#         actuators_layers = []
#         output_sizes = [out_size] + output_sizes
#         for in_size, out_size in zip(output_sizes[:-1], output_sizes[1:]):
#             trajectory_layers.append(nn.Linear(in_size, out_size))
#             trajectory_layers.append(nn.SiLU(inplace=True))
#             actuators_layers.append(nn.Linear(in_size, out_size))
#             actuators_layers.append(nn.SiLU(inplace=True))
            
#         trajectory_layers.append(nn.Linear(out_size, trajectory_size))
#         actuators_layers.append(nn.Linear(out_size, actuators_size))
            
#         self.trajectory_predictor_head = nn.Sequential(*trajectory_layers)
#         self.actuators_predictor_head = nn.Sequential(*actuators_layers)

#     def forward(self, state, left_bound, right_bound):
#         left_bound_enc = self.bound_encoder(left_bound)
#         right_bound_enc = self.bound_encoder(right_bound)
            
#         combined = torch.cat([state, left_bound_enc, right_bound_enc], axis=1)
#         combined = self.middle_part(combined)
            
#         for_trajectory = combined.clone()
#         for_actuators = combined.clone()
#         trajectory_pred = self.trajectory_predictor_head(for_trajectory)
#         actuators_pred = self.actuators_predictor_head(for_actuators)
          
#         return {'trajectory_pred': trajectory_pred, 'actuators_pred': actuators_pred}