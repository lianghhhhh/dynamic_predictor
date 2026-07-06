import torch.nn as nn


class CarDynamicModel(nn.Module):
    """State-space style dynamics model.

    input_size = len(CONTROL_COLS) + N_ABS_STATE + len(STATE_COLS)
               = 4 (control: effort commands only)
                 + 17 (absolute state, EXCLUDING x/y position -- only delta
                       position is fed in, see utils.ABS_STATE_MASK)
                 + 19 (delta state, includes delta x/y)
               = 40
               NOTE: update this if you change CONTROL_COLS/STATE_COLS in utils.py.
    output_size = len(STATE_COLS) = 19  (predicted delta-state, includes
                  delta x/y position so rollout can reconstruct absolute
                  position by accumulating deltas)

    Output dimension intentionally matches the state portion of the input,
    so a prediction can be fed straight back in as the next state for
    closed-loop / multi-step rollout.
    """

    def __init__(self, input_size=40, hidden_size=128, num_layers=3, output_size=19, dropout=0.1):
        super(CarDynamicModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc = nn.Sequential( # 3 layers 
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_step_out = lstm_out[:, -1, :]
        return self.fc(last_step_out)