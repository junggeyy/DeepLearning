import lightning as L
import torch, torchmetrics

class AmesHousingPredictor(L.LightningModule):
    def __init__(self, model, learning_rate):
        super().__init__()

        self.model = model
        self.learning_rate = learning_rate
        self.criterion = torch.nn.MSELoss()

        self.train_mse = torchmetrics.MeanSquaredError()
        self.val_mse = torchmetrics.MeanSquaredError()
        self.test_mse = torchmetrics.MeanSquaredError()

    def forward(self, x):
        return self.model(x)
    
    def _get_loss(self, batch):
        features, true_label = batch

        preds = self(features)
        loss = self.criterion(preds, true_label)

        return loss, preds, true_label

    def training_step(self, batch):
        loss, preds, true_label = self._get_loss(batch)

        self.log("train_loss", loss)
        self.train_mse(preds, true_label)
        self.log(
            "train_mse", self.train_mse, prog_bar=True, on_epoch=True, on_step=False
        )
        return loss      

    def validation_step(self, batch):
        loss, preds, true_label = self._get_loss(batch)

        self.log("val_loss", loss)
        self.val_mse(preds, true_label)
        self.log(
            "val_mse", self.val_mse, prog_bar=True, on_epoch=True, on_step=False
        )  

    def test_step(self, batch):
        loss, preds, true_label = self._get_loss(batch)

        self.log("test_loss", loss)
        self.test_mse(preds, true_label)
        self.log(
            "test_mse", self.test_mse, prog_bar=True, on_epoch=True, on_step=False
        )  

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        return optimizer