import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from utils import AerialCactusDataModule, Model, handle_dataset_artifact


def main():
    project_name = "mlops-wandb"
    user_name = "bonnefond-virginie"
    artifact_name = "cactus"

    # Initialize a trainer
    wandb_logger = WandbLogger(
        project=project_name, offline=False, log_model=True, job_type="train"
    )

    # Init our model
    model = Model()

    artifact_dir = handle_dataset_artifact(
        wandb_logger,
        artifact_ref=f"{user_name}/{project_name}/{artifact_name}:latest",
        artifact_name=artifact_name,
        local_path="/Users/virginie/repos/github-actions-hello-world/data.zip",
    )

    dm = AerialCactusDataModule(data_dir=artifact_dir)

    # log model only if `val_accuracy` increases
    checkpoint_callback = ModelCheckpoint(monitor="train_loss", mode="min")
    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=2,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
    )

    # Train the model ⚡
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
