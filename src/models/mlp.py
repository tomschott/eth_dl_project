import torch.nn as nn
import torch.nn.functional as F

class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.flatten = nn.Flatten()
        # Similar simple structure to paper, just smaller
        self.l1 = nn.Linear(28*28, 12, bias=False)
        self.l2 = nn.Linear(12, 10, bias=False)

    def forward(self, x):
        x = self.flatten(x)
        return self.l2(F.relu(self.l1(x)))


class ThreeLayerMLP(nn.Module):
    def __init__(self):
        super(ThreeLayerMLP, self).__init__()
        self.flatten = nn.Flatten()
        # Similar simple structure to paper, just smaller
        self.l1 = nn.Linear(28*28, 15, bias=False)
        self.l2 = nn.Linear(15, 15, bias=False)
        self.l3 = nn.Linear(15, 15, bias=False)
        self.l4 = nn.Linear(15, 10, bias=False)

    def forward(self, x):
        x = self.flatten(x)
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h3 = F.relu(self.l3(h2))
        return self.l4(h3)


class ConfigurableMLP(nn.Module):
    def __init__(self, num_hidden_layers=3, hidden_dim=15):
        super(ConfigurableMLP, self).__init__()
        if num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be >= 1")
        self.flatten = nn.Flatten()
        input_dim = 28 * 28
        output_dim = 10

        layers = []
        in_dim = input_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim, bias=False))
            in_dim = hidden_dim
        self.hidden_layers = nn.ModuleList(layers)
        self.out = nn.Linear(in_dim, output_dim, bias=False)

    def forward(self, x):
        x = self.flatten(x)
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        return self.out(x)
