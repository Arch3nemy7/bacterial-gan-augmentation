import typer
from pathlib import Path

from bacterial_gan import __version__
from bacterial_gan.config import get_settings
from bacterial_gan.pipelines import train_pipeline, evaluate_pipeline, generate_data_pipeline

app = typer.Typer(
    name="Bacterial GAN Augmentation CLI",
    help="CLI tool for training, evaluating, and using cGAN models for bacterial image augmentation.",
    add_completion=False,
)

@app.callback()
def main():
    """
    Workflow management for bacterial image augmentation project with Conditional GAN.
    """
    pass

@app.command()
def version():
    """Display application version."""
    typer.echo(f"🔬 Bacterial GAN Augmentation CLI Version: {__version__}")

@app.command()
def train(
    config_path: str = typer.Option(default="configs/config.yaml", help="Path to YAML configuration file"),
    resume_from_checkpoint: str = typer.Option(default=None, help="Full path to checkpoint file to resume training"),
    resume_from_run_id: str = typer.Option(default=None, help="MLflow run ID to resume training (use with --resume-from-epoch)"),
    resume_from_epoch: int = typer.Option(default=None, help="Epoch number to resume from (use with --resume-from-run-id)")
):
    """
    Run the complete cGAN model training pipeline.

    Usage Examples:

    1. Start new training:
       bacterial-gan train

    2. Resume from checkpoint (direct path):
       bacterial-gan train --resume-from-checkpoint models/RUN_ID/checkpoint_epoch_0150.npy

    3. Resume from run ID and epoch (recommended):
       bacterial-gan train --resume-from-run-id RUN_ID --resume-from-epoch 150
    """
    # Determine checkpoint path
    checkpoint_path = None

    # Validate resume parameters
    if resume_from_checkpoint and (resume_from_run_id or resume_from_epoch):
        typer.secho(
            "❌ Error: Cannot use --resume-from-checkpoint with --resume-from-run-id/--resume-from-epoch together",
            fg=typer.colors.RED
        )
        typer.echo("   Choose one method:")
        typer.echo("   1. --resume-from-checkpoint PATH")
        typer.echo("   2. --resume-from-run-id RUN_ID --resume-from-epoch EPOCH")
        raise typer.Exit(code=1)

    if resume_from_checkpoint:
        # Option 1: Direct checkpoint path
        checkpoint_path = Path(resume_from_checkpoint)

        if not checkpoint_path.exists():
            typer.secho(f"❌ Error: Checkpoint file not found: {checkpoint_path}", fg=typer.colors.RED)
            raise typer.Exit(code=1)

        typer.echo(f"📂 Resuming training from checkpoint: {checkpoint_path}")
        checkpoint_path = str(checkpoint_path)

    elif resume_from_run_id and resume_from_epoch:
        # Option 2: Run ID + Epoch
        checkpoint_path = Path(f"models/{resume_from_run_id}/checkpoint_epoch_{resume_from_epoch:04d}.npy")

        if not checkpoint_path.exists():
            typer.secho(f"❌ Error: Checkpoint file not found: {checkpoint_path}", fg=typer.colors.RED)
            typer.echo(f"\n💡 Tip: Check available checkpoints with:")
            typer.echo(f"   ls models/{resume_from_run_id}/")
            raise typer.Exit(code=1)

        typer.echo(f"📂 Resuming training from:")
        typer.echo(f"   🆔 Run ID: {resume_from_run_id}")
        typer.echo(f"   📊 Epoch: {resume_from_epoch}")
        typer.echo(f"   📁 Checkpoint: {checkpoint_path}")
        checkpoint_path = str(checkpoint_path)

    elif resume_from_run_id or resume_from_epoch:
        # Error: Only one parameter provided
        typer.secho("❌ Error: --resume-from-run-id and --resume-from-epoch must be used together", fg=typer.colors.RED)
        typer.echo("\n💡 Example:")
        typer.echo("   bacterial-gan train --resume-from-run-id 456de292f6bf403d95357bf1554b32e9 --resume-from-epoch 150")
        raise typer.Exit(code=1)
    else:
        # New training
        typer.echo(f"🚀 Starting new training with configuration from: {config_path}")

    # Load settings and run training
    settings = get_settings(config_path)
    train_pipeline.run(settings, resume_from_checkpoint=checkpoint_path)
    typer.secho("✅ Training pipeline completed successfully!", fg=typer.colors.GREEN)

@app.command()
def evaluate(
    run_id: str = typer.Argument(default=None, help="MLflow run ID of the model to evaluate"),
    config_path: str = typer.Option(default="configs/base.yaml", help="Path to YAML configuration file")
):
    """
    Evaluate a trained model from a specific MLflow run.
    """
    if run_id is None:
        typer.secho("❌ Error: run_id is required", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    typer.echo(f"📊 Starting evaluation for model from MLflow run ID: {run_id}")
    settings = get_settings(config_path)
    evaluate_pipeline.run(settings, run_id)
    typer.secho("✅ Evaluation completed successfully!", fg=typer.colors.GREEN)

@app.command()
def generate_data(
    run_id: str = typer.Argument(default=None, help="MLflow run ID of the model to use"),
    num_images: int = typer.Option(default=100, help="Number of synthetic images to generate"),
    config_path: str = typer.Option(default="configs/base.yaml", help="Path to YAML configuration file")
):
    """
    Generate synthetic bacterial images using a trained model.
    """
    if run_id is None:
        typer.secho("❌ Error: run_id is required", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    typer.echo(f"🎨 Generating {num_images} synthetic images using model from run ID: {run_id}")
    settings = get_settings(config_path)
    generate_data_pipeline.run(settings, run_id, num_images)
    typer.secho("✅ Synthetic data generation completed successfully!", fg=typer.colors.GREEN)

if __name__ == "__main__":
    app()