
import torchvision
import torch
import torch.nn as nn
import torchmetrics
import torchmetrics.classification
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR  
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.utilities import model_summary


class LightResnet(LightningModule):
    def __init__(self, model, lr=0.002, batch_size = 512):
        super().__init__()

        self.model = model
        self._batch_size = batch_size
        self.save_hyperparameters(ignore=['model'])
        
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
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4, momentum=0.9)
        print(f"lr: {self.hparams.lr}")
        steps_per_epoch = self._batch_size # BATCH_SIZE
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


class ModelHandler:

    def __init__(self, batch_size) -> None:
        self._batch_size = batch_size

    # Modify the pre-existing Resnet architecture from TorchVision. 
    # The pre-existing architecture is based on ImageNet images (224x224) as input.
    # So we need to modify it for CIFAR10 images (32x32).
    def get_base_model_instance(self):
        model = torchvision.models.resnet18(num_classes=10)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        model.maxpool = nn.Identity()
        return model
    
    def get_lightning_model_instance(self, saved_model = None):
        model = self.get_base_model_instance()
        lightning_model = LightResnet(model=model, batch_size=self._batch_size)

        if(saved_model is not None):
            lightning_model.load_state_dict(torch.load(saved_model))        
                  
        return lightning_model
    
    def show_model_summary(self, lightning_model):
        return model_summary.ModelSummary(lightning_model, max_depth=2)  


