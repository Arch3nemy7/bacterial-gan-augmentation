"""
Main FastAPI application for bacterial GAN augmentation API.

This API should provide endpoints for:
1. Model inference (generate synthetic images)
2. Model management (list available models, model info)
3. Health checks and monitoring
4. File upload for custom generation
5. Evaluation results retrieval
6. Real-time generation progress tracking
"""

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / "src"))

from config import settings
from app.api.router import api_router
from app.core.logging_config import setup_api_logging
from app.core.dependencies import get_model_registry

app = FastAPI(
    title=settings.app.title,
    version=settings.app.version,
    description="API for generating synthetic bacterial images using trained GAN models",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")

@app.on_event("startup")
async def startup_event():
    """Initialize services on application startup."""
    setup_api_logging()
    logging.info("Bacterial GAN API starting up...")
    logging.info("API startup completed")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on application shutdown."""
    logging.info("Bacterial GAN API shutting down...")

@app.get("/")
async def root():
    """Root endpoint with basic API information."""
    return {
        "message": "Bacterial GAN Augmentation API",
        "version": settings.app.version,
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "version": settings.app.version,
        "models_available": True,
        "database_connected": True
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
