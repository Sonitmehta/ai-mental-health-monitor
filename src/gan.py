import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Dict, List


class FeatureGenerator(nn.Module):
    """Deep Neural Generator for synthesizing tabular / TF-IDF text feature distributions."""

    def __init__(self, latent_dim: int, feature_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, feature_dim),
            nn.Sigmoid()  # Matches normalized TF-IDF feature range [0, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class FeatureDiscriminator(nn.Module):
    """Adversarial Discriminator that distinguishes authentic from generated synthetic feature vectors."""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_feature_gan(
    real_features: np.ndarray,
    epochs: int = 150,
    batch_size: int = 16,
    latent_dim: int = 32,
    lr: float = 0.0002
) -> Tuple[FeatureGenerator, Dict[str, List[float]]]:
    """
    Trains a Generative Adversarial Network (GAN) on minority/high-risk feature vectors
    to learn their underlying latent distribution for data augmentation.
    """
    real_tensor = torch.tensor(real_features, dtype=torch.float32)
    dataset_size, feature_dim = real_features.shape

    generator = FeatureGenerator(latent_dim, feature_dim)
    discriminator = FeatureDiscriminator(feature_dim)

    criterion = nn.BCELoss()
    opt_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    history = {"d_loss": [], "g_loss": []}
    actual_batch = min(batch_size, max(4, dataset_size))

    for epoch in range(epochs):
        perm = torch.randperm(dataset_size)
        epoch_d_loss = []
        epoch_g_loss = []

        for i in range(0, dataset_size, actual_batch):
            indices = perm[i:i + actual_batch]
            batch_real = real_tensor[indices]
            cur_bs = batch_real.size(0)
            if cur_bs < 2:
                continue

            # 1. Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
            real_labels = torch.ones((cur_bs, 1), dtype=torch.float32) * 0.9  # Label smoothing
            fake_labels = torch.zeros((cur_bs, 1), dtype=torch.float32)

            opt_d.zero_grad()
            d_real_out = discriminator(batch_real)
            d_loss_real = criterion(d_real_out, real_labels)

            noise = torch.randn(cur_bs, latent_dim)
            generator.eval()
            with torch.no_grad():
                fake_data = generator(noise)
            discriminator.train()
            d_fake_out = discriminator(fake_data)
            d_loss_fake = criterion(d_fake_out, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            opt_d.step()

            # 2. Train Generator: max log(D(G(z)))
            generator.train()
            opt_g.zero_grad()
            noise = torch.randn(cur_bs, latent_dim)
            gen_fake = generator(noise)
            d_gen_out = discriminator(gen_fake)
            g_loss = criterion(d_gen_out, torch.ones((cur_bs, 1), dtype=torch.float32))

            g_loss.backward()
            opt_g.step()

            epoch_d_loss.append(d_loss.item())
            epoch_g_loss.append(g_loss.item())

        if epoch_d_loss and epoch_g_loss:
            history["d_loss"].append(float(np.mean(epoch_d_loss)))
            history["g_loss"].append(float(np.mean(epoch_g_loss)))

    return generator, history


def generate_synthetic_features(
    generator: FeatureGenerator,
    n_samples: int,
    latent_dim: int = 32
) -> np.ndarray:
    """Generate n_samples synthetic feature vectors from the trained generator."""
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(n_samples, latent_dim)
        synthetic = generator(noise).cpu().numpy()
    return synthetic
