"""
FairGAN
=======

Implementation of the FairGAN algorithm as published in the paper:
Xu, D., Yuan, S., Zhang, L., & Wu, X. (2018). FairGAN: Fairness-aware Generative Adversarial Networks (1805.11202v1).
arXiv. https://arxiv.org/abs/1805.11202v1


PyTorch implementation of FairGAN, following:

    Structure (Figure 2):
        - Encoder + GDec (Generator/Decoder) form an autoencoder used to
          pre-train the generator's decoder.
        - GDec(z, s) generates fake (x_hat, y_hat) conditioned on noise z
          and a (re-sampled) protected attribute s.
        - D1 is a discriminator that tries to distinguish real (x, y, s)
          triples from fake (x_hat, y_hat, s_hat) triples.
        - D2 is a discriminator that tries to predict the protected
          attribute s from the generated (x_hat, y_hat) alone (i.e. with
          s removed). GDec is trained to fool D2, which pushes the
          generated (x, y) distribution to be independent of s
          (statistical/demographic parity in the synthetic data).

    Training (Algorithm 1):
        1. Pre-train the Autoencoder (Encoder + GDec) to reconstruct x.
        2. Alternate:
             a. Update D1 on real (x, y, s) vs. fake (x_hat, y_hat, s_hat).
             b. Update GDec (generator) to fool D1.
             c. Update D2 on fake (x_hat, y_hat) grouped by s_hat = 1 vs
                s_hat = 0.
             d. Update GDec (generator) to fool D2.
"""

import dataclasses
import random
from dataclasses import dataclass

import torch
from numpy import np
from pandas import pd
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import F
from torch.utils.data import DataLoader, Dataset

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# --------------------------------------------------------------------------
# 1. Data handling
# --------------------------------------------------------------------------


@dataclass
class FairGANData:
    """Holds preprocessed tensors and the scaler used to build them."""

    x: torch.Tensor  # (N, x_dim) continuous/normalized features
    y: torch.Tensor  # (N, 1) binary label in {0, 1}
    s: torch.Tensor  # (N, 1) binary protected attribute in {0, 1}
    scaler: StandardScaler
    feature_columns: list


class TabularFairDataset(Dataset):
    """Wraps (x, y, s) tensors as a torch Dataset."""

    def __init__(self, data: FairGANData):
        self.x = data.x
        self.y = data.y
        self.s = data.s

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.s[idx]


def prepare_dataframe(
    df: pd.DataFrame,
    feature_columns: list,
    label_column: str,
    protected_column: str,
) -> FairGANData:
    """
    Turn a pandas DataFrame into normalized tensors (x, y, s).

    - `feature_columns` are standardized with sklearn's StandardScaler.
    - `label_column` and `protected_column` must already be binary
      (0/1). Use pandas.get_dummies / LabelEncoder beforehand if needed.
    """
    scaler = StandardScaler()
    x = scaler.fit_transform(df[feature_columns].values.astype(np.float32))
    y = df[label_column].values.astype(np.float32).reshape(-1, 1)
    s = df[protected_column].values.astype(np.float32).reshape(-1, 1)

    assert set(np.unique(y)).issubset({0.0, 1.0}), "label_column must be binary 0/1"
    assert set(np.unique(s)).issubset({0.0, 1.0}), "protected_column must be binary 0/1"

    return FairGANData(
        x=torch.tensor(x, dtype=torch.float32),
        y=torch.tensor(y, dtype=torch.float32),
        s=torch.tensor(s, dtype=torch.float32),
        scaler=scaler,
        feature_columns=feature_columns,
    )


# --------------------------------------------------------------------------
# 2. Networks
# --------------------------------------------------------------------------
def _mlp(
    input_dim: int,
    hidden_dims: list,
    output_dim: int,
    out_activation: nn.Module | None = None,
):
    layers = []
    prev = input_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), nn.ReLU(), nn.Linear(h, h)]
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    if out_activation is not None:
        layers.append(out_activation)
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    """Encodes (x, y, s) -> z. Used only to pre-train GDec."""

    def __init__(self, x_dim: int, z_dim: int, hidden_dims: list):
        super().__init__()
        self.net = _mlp(x_dim + 1 + 1, hidden_dims, z_dim)

    def forward(self, x, y, s):
        inp = torch.cat([x, y, s], dim=1)
        return self.net(inp)


class GDec(nn.Module):
    """
    Generator / decoder: GDec(z, s) -> (x_hat, y_hat).

    This is the ONLY network kept after pre-training; it is the FairGAN
    generator used in both the adversarial (D1) and fairness (D2) games.
    """

    def __init__(self, z_dim: int, x_dim: int, hidden_dims: list):
        super().__init__()
        self.trunk = _mlp(
            z_dim + 1, hidden_dims, hidden_dims[-1] if hidden_dims else z_dim
        )
        trunk_out = hidden_dims[-1] if hidden_dims else z_dim
        self.x_head = nn.Linear(trunk_out, x_dim)
        self.y_head = nn.Linear(trunk_out, 1)

    def forward(self, z, s):
        inp = torch.cat([z, s], dim=1)
        h = self.trunk(inp)
        # continuous features (no squashing)
        x_hat = self.x_head(h)
        # logit; sigmoid applied where needed
        y_hat_logit = self.y_head(h)
        return x_hat, y_hat_logit


class D1(nn.Module):
    """Discriminator: real (x, y, s) vs. fake (x_hat, y_hat, s_hat)."""

    def __init__(self, x_dim: int, hidden_dims: list):
        super().__init__()
        self.net = _mlp(x_dim + 1 + 1, hidden_dims, 1)

    def forward(self, x, y, s):
        inp = torch.cat([x, y, s], dim=1)
        return self.net(inp)  # logit


class D2(nn.Module):
    """Discriminator: predicts protected attribute s from (x_hat, y_hat)."""

    def __init__(self, x_dim: int, hidden_dims: list):
        super().__init__()
        self.net = _mlp(x_dim + 1, hidden_dims, 1)

    def forward(self, x, y):
        inp = torch.cat([x, y], dim=1)
        return self.net(inp)  # logit


# --------------------------------------------------------------------------
# 3. FairGAN model wrapper
# --------------------------------------------------------------------------


@dataclass
class FairGANConfig:
    x_dim: int
    z_dim: int = 32
    hidden_dims: list = dataclasses.field(default_factory=lambda: [128, 64])
    lr: float = 2e-4
    betas: tuple = (0.5, 0.999)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class FairGAN:
    def __init__(self, config: FairGANConfig):
        self.cfg = config
        dev = config.device

        self.encoder = Encoder(config.x_dim, config.z_dim, config.hidden_dims).to(dev)
        self.gdec = GDec(config.z_dim, config.x_dim, config.hidden_dims).to(dev)
        self.d1 = D1(config.x_dim, config.hidden_dims).to(dev)
        self.d2 = D2(config.x_dim, config.hidden_dims).to(dev)

        self.opt_ae = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.gdec.parameters()),
            lr=config.lr,
            betas=config.betas,
        )
        self.opt_g = torch.optim.Adam(
            self.gdec.parameters(), lr=config.lr, betas=config.betas
        )
        self.opt_d1 = torch.optim.Adam(
            self.d1.parameters(), lr=config.lr, betas=config.betas
        )
        self.opt_d2 = torch.optim.Adam(
            self.d2.parameters(), lr=config.lr, betas=config.betas
        )

        self.bce = nn.BCEWithLogitsLoss()

    # ---- sampling helpers -------------------------------------------------

    def sample_z(self, n: int) -> torch.Tensor:
        return torch.randn(n, self.cfg.z_dim, device=self.cfg.device)

    def sample_s(self, n: int, p_s1: float) -> torch.Tensor:
        """Sample s ~ P_G(s), i.e. from the empirical marginal of s."""
        s = (torch.rand(n, 1, device=self.cfg.device) < p_s1).float()
        return s

    def generate(self, n: int, p_s1: float):
        z = self.sample_z(n)
        s_hat = self.sample_s(n, p_s1)
        x_hat, y_logit = self.gdec(z, s_hat)
        return x_hat, y_logit, s_hat

    # ---- training steps -----------------------------------------------

    def pretrain_autoencoder(
        self, loader: DataLoader, epochs: int, y_recon_weight: float = 1.0
    ):
        """Algorithm 1, lines 1-4: pre-train Encoder + GDec as an autoencoder."""
        dev = self.cfg.device
        self.encoder.train()
        self.gdec.train()
        for epoch in range(epochs):
            total_loss = 0.0
            n_batches = 0
            for x, y, s in loader:
                x, y, s = x.to(dev), y.to(dev), s.to(dev)
                self.opt_ae.zero_grad()
                z = self.encoder(x, y, s)
                x_hat, y_hat_logit = self.gdec(z, s)
                # Reconstruction loss (paper specifies ||x' - x||^2; we
                # optionally also reconstruct y with a BCE term since GDec
                # produces both outputs and needs both heads trained).
                loss = F.mse_loss(x_hat, x)
                if y_recon_weight > 0:
                    loss = loss + y_recon_weight * self.bce(y_hat_logit, y)
                loss.backward()
                self.opt_ae.step()
                total_loss += loss.item()
                n_batches += 1
            print(
                f"[Pretrain AE] epoch {epoch + 1}/{epochs} - loss {total_loss / n_batches:.4f}"
            )

    def train(self, loader: DataLoader, epochs: int, p_s1: float, log_every: int = 1):
        """Algorithm 1, lines 5-13: adversarial training of D1, D2, GDec."""
        dev = self.cfg.device
        for epoch in range(epochs):
            d1_losses, g1_losses, d2_losses, g2_losses = [], [], [], []
            for x, y, s in loader:
                m = x.shape[0]
                x, y, s = x.to(dev), y.to(dev), s.to(dev)

                # --- line 7: sample fake batch (x_hat, y_hat, s_hat) ~ P_G ---
                x_hat, y_hat_logit, s_hat = self.generate(m, p_s1)
                y_hat = torch.sigmoid(y_hat_logit)

                # === line 8: update D1 (ascend) ===
                self.d1.train()
                self.opt_d1.zero_grad()
                real_logits = self.d1(x, y, s)
                fake_logits = self.d1(x_hat.detach(), y_hat.detach(), s_hat.detach())
                d1_loss = self.bce(real_logits, torch.ones_like(real_logits)) + self.bce(
                    fake_logits, torch.zeros_like(fake_logits)
                )
                d1_loss.backward()
                self.opt_d1.step()
                d1_losses.append(d1_loss.item())

                # === line 9: update GDec to fool D1 (descend) ===
                self.gdec.train()
                self.opt_g.zero_grad()
                x_hat2, y_hat_logit2, s_hat2 = self.generate(m, p_s1)
                y_hat2 = torch.sigmoid(y_hat_logit2)
                fool_d1_logits = self.d1(x_hat2, y_hat2, s_hat2)
                # non-saturating variant of "descend log(1 - D1(fake))":
                # equivalent minimization target, avoids vanishing gradients.
                g1_loss = self.bce(fool_d1_logits, torch.ones_like(fool_d1_logits))
                g1_loss.backward()
                self.opt_g.step()
                g1_losses.append(g1_loss.item())

                # --- lines 10: sample fake batches split by s_hat ---
                n1 = m
                n0 = m
                x1, y1_logit, s1 = self.generate(n1, p_s1=1.0)  # s_hat = 1
                x0, y0_logit, s0 = self.generate(n0, p_s1=0.0)  # s_hat = 0
                y1 = torch.sigmoid(y1_logit)
                y0 = torch.sigmoid(y0_logit)

                # === line 11: update D2 (ascend) ===
                self.d2.train()
                self.opt_d2.zero_grad()
                d2_logits_1 = self.d2(x1.detach(), y1.detach())
                d2_logits_0 = self.d2(x0.detach(), y0.detach())
                d2_loss = self.bce(d2_logits_1, torch.ones_like(d2_logits_1)) + self.bce(
                    d2_logits_0, torch.zeros_like(d2_logits_0)
                )
                d2_loss.backward()
                self.opt_d2.step()
                d2_losses.append(d2_loss.item())

                # === line 12: update GDec to fool D2 (descend) ===
                self.opt_g.zero_grad()
                x1b, y1b_logit, _ = self.generate(n1, p_s1=1.0)
                x0b, y0b_logit, _ = self.generate(n0, p_s1=0.0)
                y1b = torch.sigmoid(y1b_logit)
                y0b = torch.sigmoid(y0b_logit)
                fool_d2_logits_1 = self.d2(x1b, y1b)
                fool_d2_logits_0 = self.d2(x0b, y0b)
                # GDec wants D2 to be unable to tell the groups apart, i.e.
                # push both towards the opposite label (fool it both ways).
                g2_loss = self.bce(
                    fool_d2_logits_1, torch.zeros_like(fool_d2_logits_1)
                ) + self.bce(fool_d2_logits_0, torch.ones_like(fool_d2_logits_0))
                g2_loss.backward()
                self.opt_g.step()
                g2_losses.append(g2_loss.item())

            if (epoch + 1) % log_every == 0:
                print(
                    f"[Train] epoch {epoch + 1}/{epochs} "
                    f"D1 {np.mean(d1_losses):.4f} G(D1) {np.mean(g1_losses):.4f} "
                    f"D2 {np.mean(d2_losses):.4f} G(D2) {np.mean(g2_losses):.4f}"
                )

    # ---- generation / evaluation ---------------------------------------

    @torch.no_grad()
    def sample_fair_dataset(
        self, n: int, p_s1: float, scaler: StandardScaler | None = None
    ):
        """Generate a synthetic (x, y, s) dataset from the trained GDec."""
        self.gdec.eval()
        x_hat, y_logit, s_hat = self.generate(n, p_s1)
        y_hat = (torch.sigmoid(y_logit) > 0.5).float()
        x_np = x_hat.cpu().numpy()
        if scaler is not None:
            x_np = scaler.inverse_transform(x_np)
        return x_np, y_hat.cpu().numpy(), s_hat.cpu().numpy()

    @torch.no_grad()
    def demographic_parity_gap(self, n: int = 5000) -> float:
        """
        |P(y_hat=1 | s_hat=1) - P(y_hat=1 | s_hat=0)| on generated data.
        Should shrink over training as D2 is fooled.
        """
        self.gdec.eval()
        _, y1_logit, _ = self.generate(n, p_s1=1.0)
        _, y0_logit, _ = self.generate(n, p_s1=0.0)
        p1 = torch.sigmoid(y1_logit).mean().item()
        p0 = torch.sigmoid(y0_logit).mean().item()
        return abs(p1 - p0)


# --------------------------------------------------------------------------
# 4. Demo / example usage
# --------------------------------------------------------------------------


def _make_synthetic_biased_dataframe(n: int = 5000, seed: int = 0) -> pd.DataFrame:
    """
    Builds a toy biased dataset: protected attribute s in {0,1}, features x
    correlated with s, and label y whose probability depends on s (i.e. an
    unfair generating process), to demonstrate FairGAN reducing that gap.
    """
    rng = np.random.default_rng(seed)
    s = rng.binomial(1, 0.5, size=n)
    x1 = rng.normal(loc=s * 1.5, scale=1.0, size=n)
    x2 = rng.normal(loc=0.0, scale=1.0, size=n)
    x3 = rng.normal(loc=-s * 0.5, scale=1.0, size=n)
    logit = 0.3 * x1 - 0.2 * x3 + 1.5 * s - 0.5
    p_y = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, p_y)
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "y": y, "s": s})


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    df = _make_synthetic_biased_dataframe(n=8000)
    feature_columns = ["x1", "x2", "x3"]
    data = prepare_dataframe(df, feature_columns, label_column="y", protected_column="s")

    p_s1 = df["s"].mean()
    print(f"Empirical P(s=1) = {p_s1:.3f}")
    real_dp_gap = abs(df.loc[df.s == 1, "y"].mean() - df.loc[df.s == 0, "y"].mean())
    print(f"Real-data demographic parity gap: {real_dp_gap:.3f}")

    dataset = TabularFairDataset(data)
    loader = DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True)

    config = FairGANConfig(x_dim=len(feature_columns), z_dim=16, hidden_dims=[64, 64])
    model = FairGAN(config)

    model.pretrain_autoencoder(loader, epochs=15)
    print(f"DP gap before adversarial training: {model.demographic_parity_gap():.3f}")

    model.train(loader, epochs=30, p_s1=p_s1, log_every=5)
    print(f"DP gap after adversarial training:  {model.demographic_parity_gap():.3f}")

    x_syn, y_syn, s_syn = model.sample_fair_dataset(2000, p_s1=p_s1, scaler=data.scaler)
    syn_df = pd.DataFrame(x_syn, columns=feature_columns)
    syn_df["y"] = y_syn
    syn_df["s"] = s_syn
    print(syn_df.head())


if __name__ == "__main__":
    main()
