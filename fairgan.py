import random

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn, optim

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


"""Model Architecture

FairGAN consists of:
- Generator
- Discriminator
- Fairness discriminator """


# Encoder
class Encoder(nn.Module):
    """Encodes (x, y, s) -> z.
    s: protected attribute
    z: noice variable
    """

    def __init__(self, x_dim: int, noise_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(x_dim + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, noise_dim),
        )

    def forward(self, x, y, s):
        return self.model(torch.cat([x, y, s], dim=1))


# Generator
class Generator(nn.Module):
    """GDec(z, s) -> (x_hat, y_hat_logit). Conditioned on the sensitive attribute."""

    def __init__(self, x_dim: int, noise_dim: int = 100, hidden_dim: int = 128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim + 1, hidden_dim),
            nn.ReLU(),
        )
        self.x_head = nn.Linear(hidden_dim, x_dim)
        self.y_head = nn.Linear(hidden_dim, 1)

    def forward(self, z, s):
        h = self.model(torch.cat([z, s], dim=1))
        x_hat = self.x_head(h)
        y_hat_logit = self.y_head(h)
        return x_hat, y_hat_logit


# Real/Fake Discriminator (D1)
class Discriminator(nn.Module):
    """D1: real (x, y, s) vs. fake (x_hat, y_hat, s_hat)."""

    def __init__(self, x_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(x_dim + 2, 128),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, y, s):
        return self.model(torch.cat([x, y, s], dim=1))


# Fairness Discriminator (D2)
class FairnessDiscriminator(nn.Module):
    """D2: predicts s from (x_hat, y_hat) -- no access to s."""

    def __init__(self, x_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(x_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, y):
        return self.model(torch.cat([x, y], dim=1))


class FairGAN:
    """FairGAN implementation for tabular data.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data to be modeled.
        sensitive_feature (str): Column name of the protected attribute.
            Must be binary, or reducible to binary.
        target_feature (str): Column name of the target attribute.
           Must be binary, or reducible to binary.
        epochs (int): Number of adversarial training epochs (default: 200)
        pretrain_epochs (int): Autoencoder pre-training epochs (default: 20)
        noise_dim (int): Dimensionality of the generator's noise input (default: 100)
        lr (float): Learning rate for all optimizers (default: 0.001)
        verbose (bool): Print progress during training (default: False)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        sensitive_feature: str,
        target_feature: str,
        epochs=200,
        pretrain_epochs=20,
        noise_dim=100,
        lr=0.001,
        verbose=False,
    ):
        self.epochs = epochs
        self.pretrain_epochs = pretrain_epochs
        self.noise_dim = noise_dim
        self._lr = lr
        self.sensitive_feature = sensitive_feature
        self.target_feature = target_feature
        self.df = df.copy()
        self.verbose = verbose

        # Store Metadata
        self.encoders = {}
        self.integer_columns = []
        self.binary_columns = []
        self.categorical_columns = []

        # Label-encode categorical columns
        for col in self.df.columns:
            if self.df[col].dtype == "object":
                self.categorical_columns.append(col)
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))
                self.encoders[col] = le

        # Identify column types
        for col in self.df.columns:
            unique_vals = self.df[col].nunique()
            if unique_vals == 2:
                self.binary_columns.append(col)
            elif pd.api.types.is_integer_dtype(self.df[col]):
                self.integer_columns.append(col)

        # Check that sensitive and target features are binary
        for feature in [sensitive_feature, target_feature]:
            if self.df[feature].nunique() != 2:
                raise ValueError(f"'{feature}' must be binary.")

        # Scale numeric columns
        self.scaler = StandardScaler()
        self.numeric_columns = [
            col
            for col in self.df.columns
            if col not in self.categorical_columns and col not in self.binary_columns
        ]
        if self.numeric_columns:
            self.df[self.numeric_columns] = self.scaler.fit_transform(
                self.df[self.numeric_columns]
            )

        # x/y/s column split (paper structure: everything except the
        # sensitive attribute and the label is treated as "x")
        self.feature_columns = [
            c for c in self.df.columns if c not in (sensitive_feature, target_feature)
        ]
        self.x_dim = len(self.feature_columns)

    # ---- internal tensors -------------------------------------------------

    def _tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert the DataFrame to PyTorch tensors.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the input features (x), target variable (y), and sensitive attribute (s).
        """
        x = torch.tensor(self.df[self.feature_columns].values, dtype=torch.float32)
        y = torch.tensor(self.df[[self.target_feature]].values, dtype=torch.float32)
        s = torch.tensor(self.df[[self.sensitive_feature]].values, dtype=torch.float32)
        return x, y, s

    def _sample_s(self, n: int, p_s1: float) -> torch.Tensor:
        return (torch.rand(n, 1) < p_s1).float()

    # ---- training -----------------------------------------------------

    def pretrain(self, y_recon_weight: float = 1.0, lr: float | None = None):
        """Pre-train Encoder + Generator (GDec) as an autoencoder.

        Args:
            y_recon_weight (float, optional): Weight for the reconstruction loss of the target variable y. Defaults to 1.0.
            lr (float, optional): Learning rate. Defaults to None.
        """
        if lr is None:
            lr = self._lr

        self._encoder = Encoder(self.x_dim, self.noise_dim)
        self._generator = Generator(self.x_dim, self.noise_dim)

        opt = optim.Adam(
            list(self._encoder.parameters()) + list(self._generator.parameters()), lr=lr
        )
        mse = nn.MSELoss()
        bce = nn.BCEWithLogitsLoss()

        x, y, s = self._tensors()

        for epoch in range(self.pretrain_epochs):
            opt.zero_grad()
            z = self._encoder(x, y, s)
            x_hat, y_hat_logit = self._generator(z, s)
            loss = mse(x_hat, x) + y_recon_weight * bce(y_hat_logit, y)
            loss.backward()
            opt.step()

            if self.verbose and epoch % 5 == 0:
                print(f"[Pretrain] epoch {epoch} | loss {loss.item():.4f}")

    def train(
        self,
        lambda_fair=0.05,
        lr=None,
        log_every=10,
    ):
        if lr is None:
            lr = self._lr

        S_index = self.df.columns.get_loc(self.sensitive_feature)
        # Y_index = self.df.columns.get_loc(self.target_feature)

        self._discriminator = Discriminator(x_dim=self.x_dim)
        self._fairness_discriminator = FairnessDiscriminator(x_dim=self.x_dim)

        criterion = nn.BCELoss()

        g_optimizer = optim.Adam(self._generator.parameters(), lr=lr)
        d1_optim = optim.Adam(self._discriminator.parameters(), lr=lr)
        d2_optim = optim.Adam(self._fairness_discriminator.parameters(), lr=lr)

        x, y, s = self._tensors()
        n = x.shape[0]
        p_s1 = s.mean().item()

        for epoch in range(self.epochs):
            # TODO: Where does lambda fair go
            real_labels = torch.ones(n, 1)
            fake_labels = torch.zeros(n, 1)

            # --- sample fake batch (x_hat, y_hat, s_hat) ~ P_G ---
            z = torch.randn(n, self.noise_dim)
            s_hat = self._sample_s(n, p_s1)
            x_hat, y_hat_logit = self._generator(z, s_hat)
            y_hat = torch.sigmoid(y_hat_logit)

            # === Update D1 (ascend) ===
            d1_real = self._discriminator(x, y, s)
            d1_fake = self._discriminator(
                x_hat.detach(), y_hat.detach(), s_hat.detach()
            )
            d1_loss = criterion(d1_real, real_labels) + criterion(d1_fake, fake_labels)

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
            z = torch.randn(n, self.noise_dim)
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

            if self.verbose and epoch % log_every == 0:
                print(
                    f"Epoch {epoch} | D1: {d1_loss.item():.4f} | D2: {d2_loss.item():.4f} | G: {g_loss.item():.4f} | Fair: {fair_loss.item():.4f}"
                )

    def generate(self, num_samples: int, p_s1: float | None = None) -> pd.DataFrame:
        """Generate a synthetic DataFrame.

        Args:
            num_samples (int): number of rows to generate.
            p_s1 (float, optional): probability of sampling s_hat=1. Defaults to the
                empirical P(s=1) from the training data.

        Returns:
            pd.DataFrame: Synthetic Dataframe with the same columns as the original DataFrame.
        """
        if p_s1 is None:
            p_s1 = self.df[self.sensitive_feature].mean()

        self._generator.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.noise_dim)
            s_hat = self._sample_s(num_samples, p_s1)
            x_hat, y_hat_logit = self._generator(z, s_hat)
            y_hat = (torch.sigmoid(y_hat_logit) > 0.5).float()

        synthetic = pd.DataFrame(x_hat.numpy(), columns=self.feature_columns)
        synthetic[self.target_feature] = y_hat.numpy()
        synthetic[self.sensitive_feature] = s_hat.numpy()
        synthetic = synthetic[self.df.columns]  # restore original column order

        # Reverse Scaling
        if self.numeric_columns:
            synthetic[self.numeric_columns] = self.scaler.inverse_transform(
                synthetic[self.numeric_columns]
            )

        # Handling integer columns
        for col in self.integer_columns:
            synthetic[col] = synthetic[col].round().astype(int)

        # Handling binary columns
        for col in self.binary_columns:
            synthetic[col] = (synthetic[col] > 0.5).astype(int)
        synthetic[self.target_feature] = synthetic[self.target_feature].astype(int)
        synthetic[self.sensitive_feature] = synthetic[self.sensitive_feature].astype(
            int
        )

        # Reverse label encoding
        for col, le in self.encoders.items():
            synthetic[col] = synthetic[col].round()
            synthetic[col] = synthetic[col].clip(0, len(le.classes_) - 1)
            synthetic[col] = le.inverse_transform(synthetic[col].astype(int))

        return synthetic
