# Menggunakan bash dengan flag untuk menghentikan eksekusi jika terjadi kesalahan
SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

# Target default yang akan dijalankan jika 'make' dipanggil tanpa argumen
.DEFAULT_GOAL := help

# Menggunakan.PHONY untuk target yang bukan merupakan file
.PHONY: help install format lint test train generate-data run-api clean

help:
	@echo "Perintah yang tersedia:"
	@echo "  install        -> Menginstal dependensi proyek menggunakan Poetry."
	@echo "  format         -> Memformat seluruh basis kode dengan black dan isort."
	@echo "  lint           -> Menjalankan linter (flake8) pada kode sumber."
	@echo "  test           -> Menjalankan semua pengujian menggunakan pytest."
	@echo "  train          -> Menjalankan pipeline pelatihan model menggunakan DVC."
	@echo "  generate-data  -> Menjalankan pipeline generasi data sintetis menggunakan DVC."
	@echo "  run-api        -> Menjalankan server API FastAPI untuk pengembangan lokal."
	@echo "  clean          -> Menghapus file sementara dan cache."

install:
	@echo ">>> Menginstal dependensi dari poetry.lock..."
	poetry install

format:
	@echo ">>> Memformat kode dengan black dan isort..."
	poetry run black .
	poetry run isort .

lint:
	@echo ">>> Menjalankan linter flake8..."
	poetry run flake8 src/ app/ tests/

test:
	@echo ">>> Menjalankan pengujian dengan pytest..."
	poetry run pytest

train:
	@echo ">>> Menjalankan pipeline pelatihan DVC..."
	poetry run dvc repro train # 'train' adalah nama stage di dvc.yaml

generate-data:
	@echo ">>> Menjalankan pipeline generasi data DVC..."
	poetry run dvc repro generate # 'generate' adalah nama stage di dvc.yaml

run-api:
	@echo ">>> Menjalankan server FastAPI di http://127.0.0.1:8000..."
	poetry run uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

clean:
	@echo ">>> Membersihkan file cache Python..."
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete