import os

import lightning as L
import torch
import torchmetrics
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


import lightning.pytorch as pl
from torch.utils.data import random_split, DataLoader

# Note - you must have torchvision installed for this example
from torchvision.datasets import MNIST
from torchvision import transforms


class MNISTDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "./"):
        super().__init__()
        self.data_dir = os.environ.get("PATH_DATASETS", ".")
        self.batch_size = 256 if torch.cuda.is_available() else 64
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.mnist_test = MNIST(
                self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)


class Model(L.LightningModule):
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
model = Model()
dm = MNISTDataModule()

# Initialize a trainer
wandb_logger = WandbLogger(project="mlops-wandb", log_model=True)
# log model only if `val_accuracy` increases
checkpoint_callback = ModelCheckpoint(monitor="train_loss", mode="min")
trainer = L.Trainer(
    accelerator="auto",
    devices=1,
    max_epochs=3,
    logger=wandb_logger,
    callbacks=[checkpoint_callback],
)


# Train the model ⚡
trainer.fit(model, dm)

# Evaluate the model ⚡
trainer.test(model, dm)
