import mlflow
import mlflow.tensorflow
from bacterial_gan.config import Settings
from bacterial_gan.models import architecture  # Fixed: models not model
#... import lainnya

def run(settings: Settings):
    """Menjalankan pipeline pelatihan lengkap dengan pelacakan MLflow."""
    
    mlflow.set_experiment("Bacterial GAN Augmentation")

    with mlflow.start_run() as run:
        # 1. Log konfigurasi/parameter
        mlflow.log_params(settings.training.model_dump())
        mlflow.log_params(settings.data.model_dump())
        mlflow.log_param("run_name", run.info.run_name)
        
        #... (Kode untuk memuat data)
        # train_dataset, val_dataset =...

        #... (Kode untuk menginisialisasi model)
        generator = architecture.build_generator(settings.training.image_size)
        discriminator = architecture.build_discriminator(settings.training.image_size)
        
        #... (Loop pelatihan utama)
        for epoch in range(settings.training.epochs):
            #... (Logika satu epoch pelatihan)
            gen_loss, disc_loss =... # Hitung loss
            
            # 2. Log metrik per epoch
            mlflow.log_metric("generator_loss", gen_loss, step=epoch)
            mlflow.log_metric("discriminator_loss", disc_loss, step=epoch)

        # 3. Log artefak dan model setelah pelatihan selesai
        # Simpan beberapa contoh gambar sintetis sebagai artefak
        # synthetic_images = generator.predict(...)
        # mlflow.log_images(synthetic_images, "generated_examples")
        
        # Simpan model menggunakan format MLflow Model
        # Ini akan mengemas model, dependensi, dan kelas Predictor
        # yang akan kita gunakan untuk deployment.
        # 'signature' mendefinisikan skema input/output model.
        # 'artifacts' memungkinkan kita untuk mengemas file tambahan (misalnya, color normalizer)
        
        # Contoh sederhana untuk logging model Keras
        mlflow.tensorflow.log_model(
            model=generator,
            artifact_path="generator_model",
            registered_model_name="bacterial-gan-generator" # Opsional, untuk Model Registry
        )
        
        print(f"Pelatihan selesai. Run ID: {run.info.run_id}")