import torch.nn as nn

class CarDynamicModel(nn.Module):
    def __init__(self, input_size=140, output_size=7):
        super(CarDynamicModel, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LayerNorm(256),
            nn.Softplus(),
            
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.Softplus(),
            
            nn.Linear(64, output_size),
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.reshape(x.shape[0], -1)
        return self.net(x)