from ames_housing_prediction.data.dataloader import AmesHousingDataModule
from ames_housing_prediction.model.AmesHousingPredictor import AmesHousingPredictor
from ames_housing_prediction.model.PyTorchModel import PyTorchModel
import lightning as L
import torch
import os

def train():
    # get data
    dm = AmesHousingDataModule()

    # setup hyperparameters
    num_features = 3
    num_classes = 1 #regression model
    learning_rate = 0.05

    # setup model
    base_model = PyTorchModel(num_features, num_classes)
    lightning_model = AmesHousingPredictor(base_model, learning_rate)

    # setup trainier
    trainer = L.Trainer(
        max_epochs=30,
        accelerator="auto", 
        devices="auto", 
        deterministic=True
    )

    trainer.fit(model=lightning_model, datamodule=dm)

    train_mse = trainer.validate(dataloaders=dm.train_dataloader())[0]["val_mse"]
    val_mse = trainer.validate(datamodule=dm)[0]["val_mse"]
    test_mse = trainer.test(datamodule=dm)[0]["test_mse"]

    print(
        f"Train MSE: {train_mse:.4f}"
        f" | Val MSE: {val_mse:.4f}"
        f" | Test MSE: {test_mse:.4f}"
    )

    os.makedirs("saved_models", exist_ok=True)
    torch.save(base_model.state_dict(), "saved_models/mlp-symbols.pt")

if __name__ == "__main__":
    torch.manual_seed(123)
    train()

# Train MSE: 0.1488 | Val MSE: 0.1547 | Test MSE: 0.1678