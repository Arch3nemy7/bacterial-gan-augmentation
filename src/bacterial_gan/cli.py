from pathlib import Path

import typer

from bacterial_gan import __version__
from bacterial_gan.config import get_settings
from bacterial_gan.pipelines import evaluate_pipeline, generate_data_pipeline, train_pipeline

app = typer.Typer(
    name="Bacterial GAN Augmentation CLI",
    help="CLI for training and using StyleGAN2-ADA models for bacterial image augmentation.",
    add_completion=False,
)


@app.callback()
def main():
    """Workflow management for bacterial image augmentation with StyleGAN2-ADA."""
    pass


@app.command()
def version():
    """Display application version."""
    typer.echo(f"🔬 Bacterial GAN Augmentation CLI Version: {__version__}")


@app.command()
def train(
    config_path: str = typer.Option(
        default="configs/config.yaml", help="Path to YAML configuration file"
    ),
    resume_from_checkpoint: str = typer.Option(
        default=None, help="Full path to checkpoint file to resume training"
    ),
    resume_from_run_id: str = typer.Option(
        default=None, help="MLflow run ID to resume training (use with --resume-from-epoch)"
    ),
    resume_from_epoch: int = typer.Option(
        default=None, help="Epoch number to resume from (use with --resume-from-run-id)"
    ),
):
    """
    Run StyleGAN2-ADA training pipeline.

    Examples:
        bacterial-gan train
        bacterial-gan train --resume-from-checkpoint models/RUN_ID/checkpoint_epoch_0150.npy
        bacterial-gan train --resume-from-run-id RUN_ID --resume-from-epoch 150
    """
    checkpoint_path = None

    if resume_from_checkpoint and (resume_from_run_id or resume_from_epoch):
        typer.secho(
            "❌ Cannot use --resume-from-checkpoint with --resume-from-run-id/--resume-from-epoch",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    if resume_from_checkpoint:
        checkpoint_path = Path(resume_from_checkpoint)
        if not checkpoint_path.exists():
            typer.secho(f"❌ Checkpoint not found: {checkpoint_path}", fg=typer.colors.RED)
            raise typer.Exit(code=1)
        typer.echo(f"📂 Resuming from: {checkpoint_path}")
        checkpoint_path = str(checkpoint_path)

    elif resume_from_run_id and resume_from_epoch:
        checkpoint_path = Path(
            f"models/{resume_from_run_id}/checkpoint_epoch_{resume_from_epoch:04d}.npy"
        )
        if not checkpoint_path.exists():
            typer.secho(f"❌ Checkpoint not found: {checkpoint_path}", fg=typer.colors.RED)
            raise typer.Exit(code=1)
        typer.echo(f"📂 Resuming: run={resume_from_run_id}, epoch={resume_from_epoch}")
        checkpoint_path = str(checkpoint_path)

    elif resume_from_run_id or resume_from_epoch:
        typer.secho(
            "❌ --resume-from-run-id and --resume-from-epoch must be used together",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)
    else:
        typer.echo(f"🚀 Starting training with: {config_path}")

    settings = get_settings(config_path)
    train_pipeline.run(settings, resume_from_checkpoint=checkpoint_path)
    typer.secho("✅ Training completed!", fg=typer.colors.GREEN)


@app.command()
def evaluate(
    run_id: str = typer.Argument(default=None, help="MLflow run ID to evaluate"),
    config_path: str = typer.Option(
        default="configs/config.yaml", help="Path to configuration file"
    ),
):
    """Evaluate a trained model."""
    if run_id is None:
        typer.secho("❌ run_id is required", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    typer.echo(f"📊 Evaluating model: {run_id}")
    settings = get_settings(config_path)
    evaluate_pipeline.run(settings, run_id)
    typer.secho("✅ Evaluation completed!", fg=typer.colors.GREEN)


@app.command()
def generate_data(
    run_id: str = typer.Argument(default=None, help="MLflow run ID of model to use"),
    num_images: int = typer.Option(default=100, help="Number of images to generate"),
    config_path: str = typer.Option(
        default="configs/config.yaml", help="Path to configuration file"
    ),
):
    """Generate synthetic bacterial images."""
    if run_id is None:
        typer.secho("❌ run_id is required", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    typer.echo(f"🎨 Generating {num_images} images from: {run_id}")
    settings = get_settings(config_path)
    generate_data_pipeline.run(settings, run_id, num_images)
    typer.secho("✅ Generation completed!", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
