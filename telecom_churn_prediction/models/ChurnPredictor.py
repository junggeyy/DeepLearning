import lightning as L
import torch
from torchmetrics.classification import BinaryAccuracy

class ChurnPredictor(L.LightningModule):
    def __init__(self, model, learning_rate):
        super().__init__()

        self.model = model
        self.learning_rate = learning_rate
        self.criterion = torch.nn.BCEWithLogitsLoss()

        self.train_accuracy = BinaryAccuracy()
        self.val_accuracy = BinaryAccuracy()
        self.test_accuracy = BinaryAccuracy()

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch):
        features, true_label = batch

        logits = self(features)
        loss = self.criterion(logits, true_label)
        preds = torch.sigmoid(logits) > 0.5

        self.train_accuracy.update(preds, true_label.int())

        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log('train_acc', self.train_accuracy, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch):
        features, true_label = batch

        logits = self(features)
        loss = self.criterion(logits, true_label)
        preds = torch.sigmoid(logits) > 0.5

        self.val_accuracy.update(preds, true_label.int())

        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_acc', self.val_accuracy, on_step=False, on_epoch=True)
    
    def test_step(self, batch):
        features, true_label = batch

        logits = self(features)
        loss = self.criterion(logits, true_label)
        preds = torch.sigmoid(logits) > 0.5

        self.test_accuracy.update(preds, true_label.int())

        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', self.test_accuracy, on_step=False, on_epoch=True)  

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.learning_rate)    