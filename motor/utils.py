import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

def get_data(data_dir, data_source):
    inputs = []
    outputs = []
    lookahead = 1  # Number of steps to look ahead for future state
    seq_length = 5  # Sequence length for LSTM input

    for file in data_source:
        file_path = os.path.join(data_dir, file)
        data = pd.read_csv(file_path)

        data['sin_angle'] = np.sin(data['angle'])
        data['cos_angle'] = np.cos(data['angle'])
        count = len(data) - lookahead

        for i in range(count-seq_length):
            sequence_data = data.iloc[i:i+seq_length][['sin_angle', 'cos_angle', 'velocity', 'target_effort']].values # don't need effort?
            start_velocity = data.iloc[i+seq_length-1]['velocity']
            end_velocity = data.iloc[i+seq_length]['velocity']
            if abs(end_velocity - start_velocity) > 5:  # If the velocity change is too large, it might be a reset, so we skip this sequence
                print(f"{file}: Skipping sequence at index {i} due to large velocity change: {start_velocity} -> {end_velocity}")
                continue
            # # Extract current state and control inputs
            # current_state = data.iloc[i][['angle', 'velocity', 'effort']].values
            # control_inputs = data.iloc[i][['target_effort']].values
            # input = np.concatenate((current_state, control_inputs))  # Combine state and control
            
            # # Extract future state after lookahead steps
            # future_state = data.iloc[i + lookahead][['angle', 'velocity', 'effort']].values

            # calculate delta between each sequence data and its future state
            input_data = []
            for j in range(seq_length):
                current_state = data.iloc[i+j][['angle', 'velocity']].values
                prev_state = data.iloc[i+j-1][['angle', 'velocity']].values
                if i+j-1 < 0:
                    prev_state = current_state  # For the first element, there's no previous state, so we can set it to current state
                # delta_data = current_state - prev_state
                # # if delta angle is too large, it might be a reset, so we can cap it to a reasonable range (e.g., -10 to 10 degrees)
                # if abs(delta_data[0]) > 10:
                #     if delta_data[0] > 0:
                #         delta_data[0] -= 4 * np.pi
                #     else:
                #         delta_data[0] += 4 * np.pi
                raw_delta_angle = current_state[0] - prev_state[0]
                delta_velocity = current_state[1] - prev_state[1]
                delta_angle = np.arctan2(np.sin(raw_delta_angle), np.cos(raw_delta_angle))  # Normalize delta angle to [-pi, pi]
                delta_data = np.array([delta_angle, delta_velocity])

                input_data.append(np.append(sequence_data[j], delta_data))  # Add delta to each sequence data

            current_raw_state = data.iloc[i + seq_length - 1][['angle', 'velocity']].values
            future_raw_state = data.iloc[i + seq_length][['angle', 'velocity']].values
            
            raw_target_delta_angle = future_raw_state[0] - current_raw_state[0]
            target_delta_velocity = future_raw_state[1] - current_raw_state[1]
            target_delta_angle = np.arctan2(np.sin(raw_target_delta_angle), np.cos(raw_target_delta_angle))  # Normalize delta angle to [-pi, pi]
            output = np.array([target_delta_angle, target_delta_velocity])  # Calculate deltas

            inputs.append(input_data)  # Append the sequence data for LSTM input
            outputs.append(output)  # Calculate deltas for state variables

    return inputs, outputs


def get_test_data(data_dir, test_data_source):
    inputs = []
    outputs = []
    lookahead = 1  # Number of steps to look ahead for future state
    seq_length = 5  # Sequence length for LSTM input

    file_path = os.path.join(data_dir, test_data_source)
    data = pd.read_csv(file_path)

    data['sin_angle'] = np.sin(data['angle'])
    data['cos_angle'] = np.cos(data['angle'])

    count = len(data) - lookahead

    for i in range(count-seq_length):
            sequence_data = data.iloc[i:i+seq_length][['sin_angle', 'cos_angle', 'velocity', 'target_effort']].values # don't need effort?
            start_velocity = data.iloc[i+seq_length-1]['velocity']
            end_velocity = data.iloc[i+seq_length]['velocity']
            if abs(end_velocity - start_velocity) > 5:  # If the velocity change is too large, it might be a reset, so we skip this sequence
                print(f" Skipping sequence at index {i} due to large velocity change: {start_velocity} -> {end_velocity}")
                continue
            # # Extract current state and control inputs
            # current_state = data.iloc[i][['angle', 'velocity', 'effort']].values
            # control_inputs = data.iloc[i][['target_effort']].values
            # input = np.concatenate((current_state, control_inputs))  # Combine state and control
            
            # # Extract future state after lookahead steps
            # future_state = data.iloc[i + lookahead][['angle', 'velocity', 'effort']].values

            # calculate delta between each sequence data and its future state
            input_data = []
            for j in range(seq_length):
                current_state = data.iloc[i+j][['angle', 'velocity']].values
                prev_state = data.iloc[i+j-1][['angle', 'velocity']].values
                if i+j-1 < 0:
                    prev_state = current_state  # For the first element, there's no previous state, so we can set it to current state
                # delta_data = current_state - prev_state
                # # if delta angle is too large, it might be a reset, so we can cap it to a reasonable range (e.g., -10 to 10 degrees)
                # if abs(delta_data[0]) > 10:
                #     if delta_data[0] > 0:
                #         delta_data[0] -= 4 * np.pi
                #     else:
                #         delta_data[0] += 4 * np.pi
                raw_delta_angle = current_state[0] - prev_state[0]
                delta_velocity = current_state[1] - prev_state[1]
                delta_angle = np.arctan2(np.sin(raw_delta_angle), np.cos(raw_delta_angle))  # Normalize delta angle to [-pi, pi]
                delta_data = np.array([delta_angle, delta_velocity])

                input_data.append(np.append(sequence_data[j], delta_data))  # Add delta to each sequence data

            current_raw_state = data.iloc[i + seq_length - 1][['angle', 'velocity']].values
            future_raw_state = data.iloc[i + seq_length][['angle', 'velocity']].values
            
            raw_target_delta_angle = future_raw_state[0] - current_raw_state[0]
            target_delta_velocity = future_raw_state[1] - current_raw_state[1]
            target_delta_angle = np.arctan2(np.sin(raw_target_delta_angle), np.cos(raw_target_delta_angle))  # Normalize delta angle to [-pi, pi]
            output = np.array([target_delta_angle, target_delta_velocity])  # Calculate deltas

            inputs.append(input_data)  # Append the sequence data for LSTM input
            outputs.append(output)  # Calculate deltas for state variables

    return inputs, outputs


class WeightedMSELoss(nn.Module):
    def __init__(self, weights):
        super(WeightedMSELoss, self).__init__()
        self.weights = weights

    def forward(self, predictions, targets):
        # Calculate squared error
        squared_errors = (predictions - targets) ** 2
        # Apply weights
        weighted_errors = squared_errors * self.weights.to(predictions.device)
        return torch.mean(weighted_errors)