
import torchvision
import torch
import torch.nn as nn
import torchmetrics
import torchmetrics.classification
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR  


# Modify the pre-existing Resnet architecture from TorchVision. 
# The pre-existing architecture is based on ImageNet images (224x224) as input.
# So we need to modify it for CIFAR10 images (32x32).
def create_model():
    model = torchvision.models.resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    return model


class LightResnet(LightningModule):
    def __init__(self, lr=0.002, steps_per_epoch = 512):
        super().__init__()

        self.steps_per_epochs = steps_per_epoch
        self.save_hyperparameters()
        self.model = create_model()
        self.train_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=10)
        self.test_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=10)
        self.valid_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=10)

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        predictions = self(inputs)
        loss = F.nll_loss(predictions, labels)
        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.train_acc(predictions, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True )
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        output = self(inputs)
        loss = F.nll_loss(output, labels)
        preds = torch.argmax(output, dim=1)
        self.valid_acc(preds, labels)
        self.log(f"valid_loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('valid_acc', self.valid_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)


    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        output = self(inputs)
        loss = F.nll_loss(output, labels)
        preds = torch.argmax(output, dim=1)
        self.test_acc(preds, labels)
        self.log(f"test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
        steps_per_epoch = self.steps_per_epochs # BATCH_SIZE
        scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer,
                0.1,
                total_steps=self.trainer.estimated_stepping_batches,
                steps_per_epoch=steps_per_epoch,
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}