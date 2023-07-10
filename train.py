import os

import lightning as L
import torch
import torchmetrics
from pytorch_lightning.loggers import WandbLogger
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
BATCH_SIZE = 256 if torch.cuda.is_available() else 64


class MNISTModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)
        self.accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=10
        )

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_nb):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.accuracy.update(preds, y)

    def on_test_epoch_end(self):
        self.log("test_acc_epoch", self.accuracy.compute())

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


# Init our model
model = MNISTModel()

# Init DataLoader from MNIST Dataset
train_ds = MNIST(
    PATH_DATASETS, train=True, download=True, transform=transforms.ToTensor()
)
val_ds = MNIST(
    PATH_DATASETS, train=False, download=True, transform=transforms.ToTensor()
)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# Initialize a trainer
wandb_logger = WandbLogger(project="mlops-wandb")
trainer = L.Trainer(
    accelerator="auto",
    devices=1,
    max_epochs=3,
    logger=wandb_logger,
)

# Train the model ⚡
trainer.fit(model, train_loader)

# Evaluate the model ⚡
trainer.test(dataloaders=val_loader)
