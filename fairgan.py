import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


"""Model Architecture

FairGAN consists of:
- Generator
- Discriminator
- Fairness discriminator """


# Generator
class Generator(nn.Module):
    def __init__(self, input_dim: int, noise_dim=100):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 128), 
            nn.ReLU(), 
            nn.Linear(128, input_dim)
        )

    def forward(self, z):
        return self.model(z)


# Real/Fake Discriminator (D1)
class Discriminator(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


# Fairness Discriminator (D2)
class FairnessDiscriminator(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


class FairGAN:
    """FairGAN implementation for tabular data.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data to be modeled.
        sensitive_feature (str): Name of the column representing the sensitive attribute (e.g. race)
        target_feature (str): Name of the column representing the target attribute (e.g., is_recid)
        epochs (int): Number of training epochs (default: 200)
        noise_dim (int): Dimensionality of the noise vector for the generator (default: 100)
        lr (float): Learning rate for the optimizers (default: 0.001)
        verbose (bool): Whether to have print statements for progress results. (default: False)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        sensitive_feature: str,
        target_feature: str,
        epochs=200,
        noise_dim=100,
        lr=0.001,
        verbose=False,
    ):
        self.epochs = epochs
        self.noise_dim = noise_dim
        self._lr = lr
        self.sensitive_feature = sensitive_feature
        self.target_feature = target_feature
        self.df = df.copy()
        self.df["race"] = (self.df["race"] == "African-American").astype(int)
        self.verbose = verbose

        # Store Metadata
        self.encoders = {}
        self.integer_columns = []
        self.binary_columns = []
        self.categorical_columns = []

        # for col in self.df.columns:
        #     if self.df[col].dtype == "object":
        #         self.df[col] = LabelEncoder().fit_transform(self.df[col])

        # Lable Encode Categorical Columns
        for col in self.df.columns:
            if self.df[col].dtype == "object":
                self.categorical_columns.append(col)
                le = LabelEncoder()
                self.df[col] = le.fit_transform(
                    self.df[col].astype(str)
                )
                self.encoders[col] = le

        # Identify Column types
        for col in self.df.columns:
            unique_vals = self.df[col].nunique()
            if unique_vals == 2:
                self.binary_columns.append(col)
            elif pd.api.types.is_integer_dtype(self.df[col]):
                self.integer_columns.append(col)

        # Scale data
        self.scaler = StandardScaler()
        self.numeric_columns = []
        for col in self.df.columns:
            if (
                col not in self.categorical_columns
                and col not in self.binary_columns
            ):
                self.numeric_columns.append(col)
        self.df[self.numeric_columns] = self.scaler.fit_transform(
            self.df[self.numeric_columns]
        )

    def train(
        self,
        lambda_fair=0.05,
        lr=None,
    ):
        if lr is None:
            lr = self._lr

        S_index = self.df.columns.get_loc(self.sensitive_feature)
        #Y_index = self.df.columns.get_loc(self.target_feature)

        self._generator = Generator(
            noise_dim=self.noise_dim, input_dim=self.df.shape[1]
        )
        self._discriminator = Discriminator(input_dim=self.df.shape[1])
        self._fairness_discriminator = FairnessDiscriminator(
            input_dim=self.df.shape[1]
        )

        criterion = nn.BCELoss()

        g_optimizer = optim.Adam(self._generator.parameters(), lr=lr)
        d1_optim = optim.Adam(self._discriminator.parameters(), lr=lr)
        d2_optim = optim.Adam(self._fairness_discriminator.parameters(), lr=lr)

        data = torch.tensor(self.df.values, dtype=torch.float32)

        for epoch in range(self.epochs):

            # Train D1 (real vs fake)
            real_labels = torch.ones(len(data), 1)
            fake_labels = torch.zeros(len(data), 1)

            z = torch.randn(len(data), self.noise_dim)
            fake_data = self._generator(z)

            d1_real = self._discriminator(data)
            d1_fake = self._discriminator(fake_data.detach())

            d1_loss = criterion(d1_real, real_labels) + criterion(
                d1_fake, fake_labels
            )

            d1_optim.zero_grad()
            d1_loss.backward()
            d1_optim.step()

            # Train D2 (predict S)
            S_real = data[:, S_index].unsqueeze(1)

            d2_pred = self._fairness_discriminator(data)
            d2_loss = criterion(d2_pred, S_real)

            d2_optim.zero_grad()
            d2_loss.backward()
            d2_optim.step()

            # Train Generator
            z = torch.randn(len(data), self.noise_dim)
            fake_data = self._generator(z)

            # GAN loss (fool D1)
            g_loss = criterion(self._discriminator(fake_data), real_labels)

            # FAIRNESS loss (Xu-style)
            S_fake_pred = self._fairness_discriminator(fake_data)

            # push D2 to be uncertain (0.5)
            target_uncertain = torch.full_like(S_fake_pred, 0.5)
            fair_loss = criterion(S_fake_pred, target_uncertain)

            # TOTAL LOSS
            total_g_loss = g_loss + lambda_fair * fair_loss

            g_optimizer.zero_grad()
            total_g_loss.backward()
            g_optimizer.step()

            if self.verbose and epoch % 10 == 0:
                print(
                    f"Epoch {epoch} | D1: {d1_loss.item():.4f} | D2: {d2_loss.item():.4f} | G: {g_loss.item():.4f} | Fair: {fair_loss.item():.4f}"
                )

    def generate(self, num_samples: int) -> pd.DataFrame: 
        self._generator.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.noise_dim)
            synthetic = self._generator(z).detach().numpy()

        fairgan_synthetic_data = pd.DataFrame(
            synthetic, columns=self.df.columns
        )

        # Reverse Scaling
        fairgan_synthetic_data[self.numeric_columns] = (
            self.scaler.inverse_transform(
                fairgan_synthetic_data[self.numeric_columns]
            )
        )

        # Handling integer columns
        for col in self.integer_columns:
            fairgan_synthetic_data[col] = (
                fairgan_synthetic_data[col].round().astype(int)
            )

        # Handling binary columns
        for col in self.binary_columns:
            fairgan_synthetic_data[col] = (fairgan_synthetic_data[col] > 0.5).astype(int)

        # Reverse lable encoding
        for col, le in self.encoders.items():
            fairgan_synthetic_data[col] = (fairgan_synthetic_data[col].round())
            fairgan_synthetic_data[col] = (
                fairgan_synthetic_data[col].clip(0, len(le.classes_) - 1)
            )
            fairgan_synthetic_data[col] = le.inverse_transform(
                fairgan_synthetic_data[col].astype(int)
            )

        return fairgan_synthetic_data
