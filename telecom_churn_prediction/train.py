import os, torch
from telecom_churn_prediction.data.dataloader import load_and_process_data
from telecom_churn_prediction.data.my_dataset import MyDataset
from telecom_churn_prediction.models.PyTorchModel import PyTorchModel
from telecom_churn_prediction.models.ChurnPredictor import ChurnPredictor
from torch.utils.data import DataLoader
from lightning import Trainer


def train():
    # get processed data 
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_process_data("telecom_churn_prediction/data/churn_dataset.csv")

    # create datasets and data loaders
    train_ds = MyDataset(X_train, y_train)
    val_ds = MyDataset(X_val, y_val)
    test_ds = MyDataset(X_test, y_test)

    BATCH_SIZE = 32
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # setup hyperparameters
    num_features = X_train.shape[1]
    num_classes = 1
    learning_rate = 0.005

    base_model = PyTorchModel(num_features, num_classes)
    lightning_model = ChurnPredictor(base_model, learning_rate)

    print(f"Training model with {num_features}")

    # defining our Trainer 
    trainer = Trainer(
        max_epochs=20,
        accelerator="auto",
        devices="auto",
    )

    # training loop
    trainer.fit(
        model=lightning_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    train_acc = trainer.test(dataloaders=train_loader)[0]["test_acc"]
    val_acc = trainer.test(dataloaders=val_loader)[0]["test_acc"]
    test_acc = trainer.test(dataloaders=test_loader)[0]["test_acc"]

    print(
        f"Train Acc {train_acc*100:.2f}%"
        f" | Val Acc {val_acc*100:.2f}%"
        f" | Test Acc {test_acc*100:.2f}%"
    )

    os.makedirs("saved_models", exist_ok=True)
    torch.save(base_model.state_dict(), "saved_models/mlp-symbols.pt")
    print("Model weights saved to saved_models/mlp-symbols.pt")


if __name__ == "__main__":
    train()


