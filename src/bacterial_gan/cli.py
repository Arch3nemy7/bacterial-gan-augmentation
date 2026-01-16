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
        bacterial-gan train --resume-from-checkpoint mlruns/.../checkpoints/checkpoint_epoch_0150.npy
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
        import glob

        mlruns_pattern = f"mlruns/*/{ resume_from_run_id}/artifacts/checkpoints/checkpoint_epoch_{resume_from_epoch:04d}.npy"
        matches = glob.glob(mlruns_pattern)
        if matches:
            checkpoint_path = Path(matches[0])
        else:
            typer.secho(f"❌ Checkpoint not found in MLflow artifacts", fg=typer.colors.RED)
            typer.secho(f"   Searched: {mlruns_pattern}", fg=typer.colors.YELLOW)
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


@app.command()
def reduce_dataset(
    input_dir: str = typer.Argument(..., help="Input directory containing images"),
    output_dir: str = typer.Argument(..., help="Output directory for reduced dataset"),
    target_size: int = typer.Option(
        default=5000, help="Target TOTAL number of images"
    ),
    with_splits: bool = typer.Option(
        default=False, help="Create train/val/test splits (70/15/15)"
    ),
    train_ratio: float = typer.Option(default=0.7, help="Train split ratio (with --with-splits)"),
    val_ratio: float = typer.Option(default=0.15, help="Val split ratio (with --with-splits)"),
    test_ratio: float = typer.Option(default=0.15, help="Test split ratio (with --with-splits)"),
    seed: int = typer.Option(default=42, help="Random seed for reproducibility"),
    dry_run: bool = typer.Option(
        default=False, help="Only report what would be done, don't copy files"
    ),
):
    """
    Reduce dataset size while preserving diversity using feature clustering.
    
    Useful for reducing training time and server costs while maintaining
    a representative sample of the original dataset.
    
    Examples:
        # Reduce single folder
        bacterial-gan reduce-dataset data/02_processed/train data/reduced --target-size 5000
        
        # Reduce entire dataset with train/val/test splits (70/15/15)
        bacterial-gan reduce-dataset data/02_processed data/reduced --target-size 5000 --with-splits
    """
    typer.echo(f"🔄 Reducing dataset: {input_dir} → {output_dir}")
    typer.echo(f"   Target size: {target_size} images")
    
    if with_splits:
        from bacterial_gan.utils.reduce_dataset import reduce_dataset_with_splits
        
        typer.echo(f"   Splits: train={train_ratio*100:.0f}%, val={val_ratio*100:.0f}%, test={test_ratio*100:.0f}%")
        
        result = reduce_dataset_with_splits(
            input_dir=Path(input_dir),
            output_dir=Path(output_dir),
            target_size=target_size,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_seed=seed,
            dry_run=dry_run,
        )
        
        if result["reduced"] > 0:
            typer.secho(
                f"\n✅ Reduced: {result['original']} → {result['reduced']} images",
                fg=typer.colors.GREEN,
            )
            typer.secho(
                f"   train: {result['train']}, val: {result['val']}, test: {result['test']}",
                fg=typer.colors.GREEN,
            )
    else:
        from bacterial_gan.utils.reduce_dataset import reduce_dataset as run_reduce
        
        result = run_reduce(
            input_dir=Path(input_dir),
            output_dir=Path(output_dir),
            target_size=target_size,
            random_seed=seed,
            dry_run=dry_run,
        )
        
        if result["reduced"] > 0:
            pct_reduction = (1 - result["reduced"] / result["original"]) * 100
            typer.secho(
                f"✅ Reduced: {result['original']} → {result['reduced']} images ({pct_reduction:.1f}% reduction)",
                fg=typer.colors.GREEN,
            )
        else:
            typer.secho("❌ No images found or reduction failed", fg=typer.colors.RED)


if __name__ == "__main__":
    app()
