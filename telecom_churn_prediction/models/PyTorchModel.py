import torch

class PyTorchModel(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.all_layers= torch.nn.Sequential(

            torch.nn.Linear(num_features, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),

            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),

            torch.nn.Linear(64, num_classes)
        )

    def forward(self, x):
        logits = self.all_layers(x).squeeze(1)
        return logits