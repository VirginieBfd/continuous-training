import wandb
from io_utils import unzip_directory


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
