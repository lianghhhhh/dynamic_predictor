import torch
import torch.nn as nn


# class MotorDynamicModel(nn.Module):
#     def __init__(self, input_size=4, hidden_size=128, output_size=3):
#         super(MotorDynamicModel, self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_size, hidden_size),
#             nn.SiLU(), # SiLU/Swish is often smoother for physical dynamics than ReLU
#             nn.Linear(hidden_size, hidden_size),
#             nn.SiLU(),
#             nn.Linear(hidden_size, output_size) # Output: delta angle, delta velocity, delta effort
#         )

#     def forward(self, x):
#         return self.net(x)



class MotorDynamicModel(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=2, output_size=2):
        super(MotorDynamicModel, self).__init__()
        # batch_first=True means inputs are (batch_size, sequence_length, features)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(), # ?
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x):
        # Pass through LSTM
        lstm_out, _ = self.lstm(x)
        
        # Extract the output from the last time step in the sequence
        last_step_out = lstm_out[:, -1, :] 
        
        # Pass through the fully connected layers to get delta predictions
        return self.fc(last_step_out)