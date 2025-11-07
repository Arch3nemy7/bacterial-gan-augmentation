import typer
from typing_extensions import Annotated

from bacterial_gan import __version__
from bacterial_gan.config import get_settings
from bacterial_gan.pipelines import train_pipeline, evaluate_pipeline, generate_data_pipeline

app = typer.Typer(
    name="Bacterial GAN Augmentation CLI",
    help="Alat CLI untuk melatih, mengevaluasi, dan menggunakan model cGAN untuk augmentasi data citra bakteri.",
    add_completion=False,
)

def version_callback(value: bool):
    """Callback untuk menampilkan versi aplikasi dan keluar."""
    if value:
        print(f"Bacterial GAN Augmentation CLI Version: {__version__}")
        raise typer.Exit()

@app.callback()
def main(
    version: Annotated = False,
):
    """
    Manajemen alur kerja untuk proyek augmentasi data bakteri dengan cGAN.
    """
    pass

@app.command()
def train(
    config_path: Annotated = "configs/config.yaml"
):
    """
    Menjalankan pipeline pelatihan model cGAN secara penuh.
    """
    typer.echo(f"Memulai pipeline pelatihan menggunakan konfigurasi dari: {config_path}")
    settings = get_settings(config_path)
    train_pipeline.run(settings)
    typer.secho("Pipeline pelatihan selesai dengan sukses.", fg=typer.colors.GREEN)

@app.command()
def evaluate(
    run_id: Annotated,
    config_path: Annotated = "configs/base.yaml"
):
    """
    Mengevaluasi model yang telah dilatih dari run MLflow tertentu.
    """
    typer.echo(f"Memulai evaluasi untuk model dari MLflow run ID: {run_id}")
    settings = get_settings(config_path)
    evaluate_pipeline.run(settings, run_id)
    typer.secho("Evaluasi selesai.", fg=typer.colors.GREEN)

@app.command()
def generate_data(
    run_id: Annotated,
    num_images: Annotated[int, typer.Option(help="Jumlah gambar sintetis yang akan dibuat.")],
    config_path: Annotated = "configs/base.yaml"
):
    """
    Menghasilkan data sintetis menggunakan model yang telah dilatih.
    """
    typer.echo(f"Menghasilkan {num_images} gambar sintetis menggunakan model dari run ID: {run_id}")
    settings = get_settings(config_path)
    generate_data_pipeline.run(settings, run_id, num_images)
    typer.secho("Proses pembuatan data sintetis selesai.", fg=typer.colors.GREEN)

if __name__ == "__main__":
    app()