import os
import shutil
from pathlib import Path

import lightning as L
import torch
import torchmetrics
import torchvision.transforms as transforms
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

import wandb


def unzip_directory(directory_path):
    directory = Path(directory_path)
    for file in directory.glob("*.zip"):
        shutil.unpack_archive(str(file), extract_dir=str(file.parent))


def handle_dataset_artifact(wandb_logger, artifact_ref, artifact_name, local_path):
    try:
        artifact_dir = wandb_logger.download_artifact(
            artifact_ref, artifact_type="dataset"
        )
        unzip_directory(artifact_dir)
        return artifact_dir
    except wandb.CommError as comm_error:
        print(f"wandb raised exception: {comm_error}.")
        print("Dataset not available from wandb, uploading new dataset.")

        artifact = wandb.Artifact(name=artifact_name, type="dataset")
        artifact.add_file(local_path=local_path)
        wandb_logger.experiment.log_artifact(artifact)

        artifact_dir = wandb_logger.download_artifact(
            artifact_ref, artifact_type="dataset"
        )
        unzip_directory(artifact_dir)
        return artifact_dir


class AerialCactusDataset(Dataset):
    def __init__(self, root_dir, train=True, transform=None):
        super(AerialCactusDataset, self).__init__()

        self.root_dir = root_dir + "/data/train" if train else root_dir + "/data/test"
        self.transform = transform

        self.classes = os.listdir(self.root_dir)  # Get list of classes
        self.files = []
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.root_dir, class_name)
            class_files = [
                (os.path.join(class_dir, f), class_idx)
                for f in os.listdir(class_dir)
                if os.path.isfile(os.path.join(class_dir, f))
            ]
            self.files.extend(class_files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path, label = self.files[idx]
        img = Image.open(img_path).convert("RGB")  # Convert to RGB

        if self.transform:
            img = self.transform(img)

        return img, label


class AerialCactusDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "data"):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = 256 if torch.cuda.is_available() else 32
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def prepare_data(self):
        # download
        AerialCactusDataset(self.data_dir, train=True)
        AerialCactusDataset(self.data_dir, train=False)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            dataset_full = AerialCactusDataset(
                self.data_dir, train=True, transform=self.transform
            )
            self.dataset_train, self.dataset_val = random_split(
                dataset_full, [0.7, 0.3]
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.dataset_test = AerialCactusDataset(
                self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size)


class Model(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(3 * 32 * 32, 2)
        self.accuracy = torchmetrics.classification.Accuracy(task="binary")

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
