import torch
from torch import nn


class Recognizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(400, 192), nn.ReLU(),
                                 nn.Linear(192, 96), nn.ReLU(), nn.Linear(96, 10))

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def probabilities(model, x):
    model.eval()
    return torch.sigmoid(model(torch.as_tensor(x))).cpu().numpy()
