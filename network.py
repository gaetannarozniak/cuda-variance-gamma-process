import torch.nn as nn

class VG_nn(nn.Module):
    def __init__(self):
        super().__init__()
        h1, h2 = 128, 128
        self.net = nn.Sequential(
            nn.Linear(5, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 1),
            nn.ReLU()
        )

    def forward(self, X):
        return self.net(X)