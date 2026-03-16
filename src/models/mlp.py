from torch import nn


class MLP(nn.Module):
    def __init__(self, input_size: int, num_classes: int = 10):
        super().__init__()
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)

        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)

        self.fc_out = nn.Linear(128, num_classes)

        self.dropout = nn.Dropout(0.3)
        self.silu = nn.SiLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.silu(self.bn1(self.fc1(x)))
        x = self.dropout(x)

        x = self.silu(self.bn2(self.fc2(x)))
        x = self.dropout(x)

        x = self.silu(self.bn3(self.fc3(x)))
        x = self.fc_out(x)
        return x
