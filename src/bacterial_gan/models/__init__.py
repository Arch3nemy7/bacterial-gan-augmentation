"""GAN Models for bacterial image augmentation using StyleGAN2-ADA."""

from .losses import (
    PerceptualLoss,
    StyleGAN2ADALoss,
    get_loss_functions,
    get_regularization_functions,
    logistic_nonsaturating_discriminator_loss,
    logistic_nonsaturating_generator_loss,
    path_length_regularization,
    r1_regularization,
)
from .stylegan2_ada import (
    MappingNetwork,
    SimplifiedStyleGAN2Discriminator,
    SimplifiedStyleGAN2Generator,
    StyleGAN2Discriminator,
    StyleGAN2Generator,
    SynthesisNetwork,
    build_stylegan2_ada,
)
from .stylegan2_wrapper import StyleGAN2ADA

__all__ = [
    "build_stylegan2_ada",
    "StyleGAN2Generator",
    "StyleGAN2Discriminator",
    "SimplifiedStyleGAN2Generator",
    "SimplifiedStyleGAN2Discriminator",
    "MappingNetwork",
    "SynthesisNetwork",
    "StyleGAN2ADA",
    "StyleGAN2ADALoss",
    "get_loss_functions",
    "get_regularization_functions",
    "logistic_nonsaturating_generator_loss",
    "logistic_nonsaturating_discriminator_loss",
    "r1_regularization",
    "path_length_regularization",
    "PerceptualLoss",
]
