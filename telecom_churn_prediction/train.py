import torch
from data.dataloader import load_and_process_data, get_dataloaders
from data.my_dataset import MyDataset
from torch.utils.data import DataLoader


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


if __name__ == "__main__":
    train()


