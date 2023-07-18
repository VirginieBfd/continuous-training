import lightning as L
from pytorch_lightning.loggers import WandbLogger

from utils import AerialCactusDataModule, Model, handle_dataset_artifact


def main():
    project_name = "mlops-wandb"
    user_name = "bonnefond-virginie"
    artifact_name = "cactus"

    # Initialize a trainer
    wandb_logger = WandbLogger(project=project_name, job_type="eval")

    # Get model from registry
    artifact_dir = wandb_logger.download_artifact(
        f"{user_name}/model-registry/{project_name}:latest", artifact_type="model"
    )
    model = Model.load_from_checkpoint(f"{artifact_dir}/model.ckpt")

    artifact_dir = handle_dataset_artifact(
        wandb_logger,
        artifact_ref=f"{user_name}/{project_name}/{artifact_name}:latest",
        artifact_name=artifact_name,
        local_path="/Users/virginie/repos/github-actions-hello-world/data.zip",
    )

    dm = AerialCactusDataModule(data_dir=artifact_dir)

    # Evaluate the model ⚡
    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        logger=wandb_logger,
    )
    trainer.test(model, dm)


if __name__ == "__main__":
    main()
