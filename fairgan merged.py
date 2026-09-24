"""
FairGAN
=======

Merges:
  - The class-based `FairGAN(df, sensitive_feature, target_feature, ...)`
    interface, label-encoding / scaling preprocessing, and
    `generate()` -> DataFrame reconstruction logic from the existing code.
  - The architecture and training procedure from the FairGAN paper
    (Figure 2 / Algorithm 1):
        * GDec(z, s) is conditioned on the protected attribute, not just
          noise -- it generates (x_hat, y_hat) for a chosen/sampled s.
        * An Encoder + GDec autoencoder is pre-trained first to
          initialize the generator.
        * D1 discriminates real (x, y, s) vs. fake (x_hat, y_hat, s_hat).
        * D2 discriminates GENERATED (x_hat, y_hat) grouped by s_hat=1 vs
          s_hat=0 (not real data), and GDec is trained adversarially to
          fool it -- pushing the generated data toward demographic parity.
"""

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


# --------------------------------------------------------------------------
# Networks
# --------------------------------------------------------------------------


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


class Generator(nn.Module):
    """GDec(z, s) -> (x_hat, y_hat_logit). Conditioned on the sensitive attribute."""

    def __init__(self, x_dim: int, noise_dim: int = 100, hidden_dim: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(noise_dim + 1, hidden_dim),
            nn.ReLU(),
        )
        self.x_head = nn.Linear(hidden_dim, x_dim)
        self.y_head = nn.Linear(hidden_dim, 1)

    def forward(self, z, s):
        h = self.trunk(torch.cat([z, s], dim=1))
        x_hat = self.x_head(h)
        y_hat_logit = self.y_head(h)
        return x_hat, y_hat_logit


class Discriminator(nn.Module):
    """D1: real (x, y, s) vs. fake (x_hat, y_hat, s_hat)."""

    def __init__(self, x_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(x_dim + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, y, s):
        return self.model(torch.cat([x, y, s], dim=1))


class FairnessDiscriminator(nn.Module):
    """D2: predicts s from (x_hat, y_hat) alone -- no access to s."""

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


# --------------------------------------------------------------------------
# FairGAN
# --------------------------------------------------------------------------


class FairGAN:
    """FairGAN implementation for tabular data.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data to be modeled.
        sensitive_feature (str): Column name of the protected attribute
            (e.g. "race"). Must be binary, or reducible to binary -- if it
            has more than 2 categories, binarize it yourself before passing
            it in (e.g. df["race"] = (df["race"] == "Group A").astype(int)).
        target_feature (str): Column name of the label (e.g. "is_recid").
            Must be binary.
        epochs (int): Number of adversarial training epochs (default: 200)
        pretrain_epochs (int): Autoencoder pre-training epochs (default: 20)
        noise_dim (int): Dimensionality of the generator's noise input z (default: 100)
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

        # Store metadata
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

    def _tensors(self):
        x = torch.tensor(self.df[self.feature_columns].values, dtype=torch.float32)
        y = torch.tensor(self.df[[self.target_feature]].values, dtype=torch.float32)
        s = torch.tensor(self.df[[self.sensitive_feature]].values, dtype=torch.float32)
        return x, y, s

    def _sample_s(self, n: int, p_s1: float) -> torch.Tensor:
        return (torch.rand(n, 1) < p_s1).float()

    # ---- training -----------------------------------------------------

    def pretrain(self, y_recon_weight: float = 1.0, lr=None):
        """Pre-train Encoder + Generator (GDec) as an autoencoder (Algorithm 1, lines 1-4)."""
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

    def train(self, lr=None, log_every=10):
        """
        Adversarial training following Algorithm 1 (lines 5-13):
          1. Update D1 on real (x,y,s) vs fake (x_hat,y_hat,s_hat).
          2. Update GDec to fool D1.
          3. Update D2 on fake (x_hat,y_hat) split by s_hat=1 vs s_hat=0.
          4. Update GDec to fool D2.
        """
        if lr is None:
            lr = self._lr

        if not hasattr(self, "_generator"):
            self.pretrain(lr=lr)

        self._discriminator = Discriminator(self.x_dim)
        self._fairness_discriminator = FairnessDiscriminator(self.x_dim)

        bce = nn.BCELoss()

        g_optim = optim.Adam(self._generator.parameters(), lr=lr)
        d1_optim = optim.Adam(self._discriminator.parameters(), lr=lr)
        d2_optim = optim.Adam(self._fairness_discriminator.parameters(), lr=lr)

        x, y, s = self._tensors()
        n = x.shape[0]
        p_s1 = s.mean().item()

        for epoch in range(self.epochs):
            real_labels = torch.ones(n, 1)
            fake_labels = torch.zeros(n, 1)

            # --- sample fake batch (x_hat, y_hat, s_hat) ~ P_G ---
            z = torch.randn(n, self.noise_dim)
            s_hat = self._sample_s(n, p_s1)
            x_hat, y_hat_logit = self._generator(z, s_hat)
            y_hat = torch.sigmoid(y_hat_logit)

            # === Update D1 (ascend) ===
            d1_real = self._discriminator(x, y, s)
            d1_fake = self._discriminator(x_hat.detach(), y_hat.detach(), s_hat.detach())
            d1_loss = bce(d1_real, real_labels) + bce(d1_fake, fake_labels)

            d1_optim.zero_grad()
            d1_loss.backward()
            d1_optim.step()

            # === Update GDec to fool D1 (descend) ===
            z = torch.randn(n, self.noise_dim)
            s_hat = self._sample_s(n, p_s1)
            x_hat, y_hat_logit = self._generator(z, s_hat)
            y_hat = torch.sigmoid(y_hat_logit)
            g1_loss = bce(self._discriminator(x_hat, y_hat, s_hat), real_labels)

            g_optim.zero_grad()
            g1_loss.backward()
            g_optim.step()

            # --- sample fake batches split by s_hat, for D2 ---
            z1 = torch.randn(n, self.noise_dim)
            s1 = torch.ones(n, 1)
            x1, y1_logit = self._generator(z1, s1)
            y1 = torch.sigmoid(y1_logit)

            z0 = torch.randn(n, self.noise_dim)
            s0 = torch.zeros(n, 1)
            x0, y0_logit = self._generator(z0, s0)
            y0 = torch.sigmoid(y0_logit)

            # === Update D2 (ascend) on GENERATED data grouped by s_hat ===
            d2_pred_1 = self._fairness_discriminator(x1.detach(), y1.detach())
            d2_pred_0 = self._fairness_discriminator(x0.detach(), y0.detach())
            d2_loss = bce(d2_pred_1, real_labels) + bce(d2_pred_0, fake_labels)

            d2_optim.zero_grad()
            d2_loss.backward()
            d2_optim.step()

            # === Update GDec to fool D2 (descend) ===
            z1b = torch.randn(n, self.noise_dim)
            x1b, y1b_logit = self._generator(z1b, torch.ones(n, 1))
            y1b = torch.sigmoid(y1b_logit)

            z0b = torch.randn(n, self.noise_dim)
            x0b, y0b_logit = self._generator(z0b, torch.zeros(n, 1))
            y0b = torch.sigmoid(y0b_logit)

            # GDec wants D2 to mix the groups up -- push each group's
            # prediction toward the *other* group's label.
            g2_loss = bce(self._fairness_discriminator(x1b, y1b), fake_labels) + bce(
                self._fairness_discriminator(x0b, y0b), real_labels
            )

            g_optim.zero_grad()
            g2_loss.backward()
            g_optim.step()

            if self.verbose and epoch % log_every == 0:
                print(
                    f"Epoch {epoch} | D1: {d1_loss.item():.4f} | G(D1): {g1_loss.item():.4f} "
                    f"| D2: {d2_loss.item():.4f} | G(D2): {g2_loss.item():.4f}"
                )

    # ---- generation -----------------------------------------------------

    def generate(self, num_samples: int, p_s1: float = None) -> pd.DataFrame:
        """
        Generate a synthetic DataFrame.

        Args:
            num_samples: number of rows to generate.
            p_s1: probability of sampling s_hat=1. Defaults to the
                empirical P(s=1) from the training data.
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

        # Reverse scaling
        if self.numeric_columns:
            synthetic[self.numeric_columns] = self.scaler.inverse_transform(
                synthetic[self.numeric_columns]
            )

        # Handling integer columns
        for col in self.integer_columns:
            synthetic[col] = synthetic[col].round().astype(int)

        # Handling binary columns (includes target_feature / sensitive_feature)
        for col in self.binary_columns:
            synthetic[col] = (synthetic[col] > 0.5).astype(int)
        synthetic[self.target_feature] = synthetic[self.target_feature].astype(int)
        synthetic[self.sensitive_feature] = synthetic[self.sensitive_feature].astype(int)

        # Reverse label encoding
        for col, le in self.encoders.items():
            synthetic[col] = synthetic[col].round()
            synthetic[col] = synthetic[col].clip(0, len(le.classes_) - 1)
            synthetic[col] = le.inverse_transform(synthetic[col].astype(int))

        return synthetic

    # ---- evaluation -----------------------------------------------------

    def demographic_parity_gap(self, n: int = 5000) -> float:
        """|P(y_hat=1 | s_hat=1) - P(y_hat=1 | s_hat=0)| on generated data."""
        self._generator.eval()
        with torch.no_grad():
            z1 = torch.randn(n, self.noise_dim)
            _, y1_logit = self._generator(z1, torch.ones(n, 1))
            z0 = torch.randn(n, self.noise_dim)
            _, y0_logit = self._generator(z0, torch.zeros(n, 1))
            p1 = torch.sigmoid(y1_logit).mean().item()
            p0 = torch.sigmoid(y0_logit).mean().item()
        return abs(p1 - p0)


# --------------------------------------------------------------------------
# Demo
# --------------------------------------------------------------------------


def _make_synthetic_biased_dataframe(n: int = 5000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    s = rng.binomial(1, 0.5, size=n)
    x1 = rng.normal(loc=s * 1.5, scale=1.0, size=n)
    x2 = rng.normal(loc=0.0, scale=1.0, size=n)
    logit = 0.3 * x1 + 1.5 * s - 0.5
    p_y = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, p_y)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y, "s": s})


def main():
    df = _make_synthetic_biased_dataframe(8000)
    real_gap = abs(df.loc[df.s == 1, "y"].mean() - df.loc[df.s == 0, "y"].mean())
    print(f"Real-data demographic parity gap: {real_gap:.3f}")

    model = FairGAN(
        df,
        sensitive_feature="s",
        target_feature="y",
        epochs=300,
        pretrain_epochs=20,
        noise_dim=16,
        lr=1e-3,
        verbose=True,
    )
    print(f"DP gap before training: {model.demographic_parity_gap():.3f}")
    model.train(log_every=50)
    print(f"DP gap after training:  {model.demographic_parity_gap():.3f}")

    synthetic_df = model.generate(2000)
    print(synthetic_df.head())


if __name__ == "__main__":
    main()
