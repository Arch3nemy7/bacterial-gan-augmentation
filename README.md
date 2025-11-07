# Bacterial GAN Augmentation

Proyek untuk augmentasi data citra bakteri menggunakan Conditional Generative Adversarial Networks (cGAN) untuk meningkatkan kualitas klasifikasi bakteri Gram-positif dan Gram-negatif.

## 📋 Deskripsi Proyek

README ini harus berisi:

### 🎯 Tujuan Proyek
- Problem statement yang jelas tentang keterbatasan dataset bakteri
- Solusi yang ditawarkan menggunakan cGAN
- Target metrics dan expected outcomes

### 🏗️ Arsitektur Sistem
- Overview arsitektur cGAN yang digunakan
- Pipeline data processing dan training
- Deployment architecture untuk API
- Integration dengan MLflow untuk experiment tracking

### 📊 Dataset
- Deskripsi dataset bakteri yang digunakan
- Preprocessing steps dan normalisasi warna Macenko
- Data splits dan augmentation strategies
- Class distribution dan balancing approaches

### 🚀 Quick Start
```bash
# Installation
git clone <repository-url>
cd bacterial-gan-augmentation
make install

# Training
make train

# Generate synthetic data
bacterial-gan generate-data --run-id <mlflow-run-id> --num-images 1000

# Run API
make run-api
```

### 📁 Struktur Proyek
```
bacterial-gan-augmentation/
├── src/                    # Source code
│   ├── models/            # Model architectures
│   ├── data/              # Data handling
│   ├── pipelines/         # Training & evaluation pipelines
│   └── utils.py           # Utility functions
├── app/                   # FastAPI application
├── tests/                 # Unit tests
├── configs/               # Configuration files
├── scripts/               # Execution scripts
└── docs/                  # Documentation
```

### 🔧 Konfigurasi
- Environment setup dan dependencies
- Configuration files explanation
- MLflow setup dan tracking
- GPU requirements dan setup

### 📈 Model Performance
- Evaluation metrics yang digunakan (FID, IS, Classification accuracy)
- Benchmark results vs baseline methods
- Expert evaluation results
- Computational efficiency metrics

### 🛠️ Development
- Development workflow dan best practices
- Testing strategy
- CI/CD pipeline setup
- Contributing guidelines

### 📚 API Documentation
- Endpoints overview
- Authentication (jika ada)
- Request/response examples
- Rate limiting dan usage guidelines

### 🔬 Research Background
- Literature review tentang GAN untuk medical imaging
- Macenko color normalization explanation
- Class conditioning strategies
- Loss function design rationale

### 📄 License dan Citation
- License information
- How to cite this work
- Acknowledgments
