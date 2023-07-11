import os

import lightning as L
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
BATCH_SIZE = 256 if torch.cuda.is_available() else 64

val_ds = MNIST(
    PATH_DATASETS, train=False, download=True, transform=transforms.ToTensor()
)

val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# Evaluate the model ⚡
trainer.test(dataloaders=val_loader)
