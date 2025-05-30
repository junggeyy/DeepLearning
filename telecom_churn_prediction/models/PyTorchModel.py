import torch

class PyTorchModel(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.all_layers= torch.nn.Sequential(

            torch.nn.Linear(num_features, 64),
            torch.nn.ReLU(),

            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),

            torch.nn.Linear(32, num_classes)
        )

    def forward(self, x):
        logits = self.all_layers(x).squeeze(1)
        return logits