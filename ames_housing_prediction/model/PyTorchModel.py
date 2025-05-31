import torch

class PyTorchModel(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.all_layers= torch.nn.Sequential(

            torch.nn.Linear(num_features, 64),
            torch.nn.ReLU(),

            torch.nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        logits = self.all_layers(x)
        return logits