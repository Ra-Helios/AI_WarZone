"""
gan.py — City Grid GAN (from scratch, pure PyTorch, no pretrained models)
Generates city layout grids and places mini-AIs using learned distributions.

Architecture:
  Generator  : noise (z_dim) → MLP → flattened grid probabilities
  Discriminator: flattened grid → MLP → real/fake score
"""

import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ── Config ────────────────────────────────────────────────────────────────
GRID_ROWS = 10
GRID_COLS = 6
GRID_SIZE = GRID_ROWS * GRID_COLS  # 100 cells
Z_DIM = 64  # noise vector size
G_HIDDEN = 256
D_HIDDEN = 256
LR = 0.0002
BETA1 = 0.5
EPOCHS = 500
BATCH_SIZE = 32
AI_COUNT = 10  # number of mini-AIs per grid


# ── Generator ─────────────────────────────────────────────────────────────
class Generator(nn.Module):
    """
    Maps random noise → city grid (flat, values 0.0–1.0 = AI probability per cell)
    """

    def __init__(self, z_dim=Z_DIM, hidden=G_HIDDEN, out_size=GRID_SIZE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden * 2),
            nn.BatchNorm1d(hidden * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden * 2, hidden),
            nn.BatchNorm1d(hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, out_size),
            nn.Sigmoid(),  # output: per-cell AI probability
        )

    def forward(self, z):
        return self.net(z)


# ── Discriminator ──────────────────────────────────────────────────────────
class Discriminator(nn.Module):
    """
    Maps flattened city grid → scalar (real layout vs generated layout)
    """

    def __init__(self, in_size=GRID_SIZE, hidden=D_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_size, hidden),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


# ── Synthetic training data ────────────────────────────────────────────────
def generate_synthetic_cities(n=2000, rows=GRID_ROWS, cols=GRID_COLS, n_ais=AI_COUNT):
    """
    Generate synthetic city grids for training.
    AIs tend to cluster in urban centres (biased placement).
    Returns tensor of shape (n, rows*cols)
    """
    data = []
    for _ in range(n):
        grid = np.zeros(rows * cols, dtype=np.float32)

        # Bias: pick 2–3 "hotspot" centres
        n_centres = np.random.randint(2, 4)
        centres = [
            (np.random.randint(1, rows - 1), np.random.randint(1, cols - 1))
            for _ in range(n_centres)
        ]

        placed = 0
        attempts = 0
        while placed < n_ais and attempts < 1000:
            # Choose a centre, add Gaussian noise
            cr, cc = centres[np.random.randint(len(centres))]
            r = int(np.clip(np.random.normal(cr, rows * 0.2), 0, rows - 1))
            c = int(np.clip(np.random.normal(cc, cols * 0.2), 0, cols - 1))
            idx = r * cols + c
            if grid[idx] == 0:
                grid[idx] = 1.0
                placed += 1
            attempts += 1

        data.append(grid)
    return torch.tensor(np.array(data))


# ── Training loop ──────────────────────────────────────────────────────────
def train_gan(save_path="models/gan_weights.pt", epochs=EPOCHS):
    os.makedirs("models", exist_ok=True)

    G = Generator()
    D = Discriminator()

    opt_G = optim.Adam(G.parameters(), lr=LR, betas=(BETA1, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=LR, betas=(BETA1, 0.999))
    criterion = nn.BCELoss()

    print(f"Generating synthetic training data...")
    real_data = generate_synthetic_cities(n=2000)
    dataset = torch.utils.data.TensorDataset(real_data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f"Training GAN for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        d_losses, g_losses = [], []

        for (real_batch,) in loader:
            bs = real_batch.size(0)

            # ── Train Discriminator ──────────────────────────────────────
            real_labels = torch.ones(bs, 1)
            fake_labels = torch.zeros(bs, 1)

            z = torch.randn(bs, Z_DIM)
            fake = G(z).detach()

            loss_real = criterion(D(real_batch), real_labels)
            loss_fake = criterion(D(fake), fake_labels)
            loss_D = (loss_real + loss_fake) / 2

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # ── Train Generator ──────────────────────────────────────────
            z = torch.randn(bs, Z_DIM)
            fake = G(z)
            # Generator wants D to output 1 (real) for its fakes
            loss_G = criterion(D(fake), real_labels)

            # AI count regularisation — keep number of AIs close to target
            ai_counts = fake.sum(dim=1)
            count_penalty = ((ai_counts - AI_COUNT) ** 2).mean() * 0.1
            loss_G = loss_G + count_penalty

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            d_losses.append(loss_D.item())
            g_losses.append(loss_G.item())

        if epoch % 50 == 0:
            print(
                f"Epoch {epoch:4d}/{epochs}  D={np.mean(d_losses):.4f}  G={np.mean(g_losses):.4f}"
            )

    torch.save({"G": G.state_dict(), "D": D.state_dict()}, save_path)
    print(f"✅ GAN saved to {save_path}")
    return G, D


# ── Inference ──────────────────────────────────────────────────────────────
class GANInference:
    """
    Loads trained GAN and generates city grids for the game.
    """

    def __init__(
        self,
        weights_path="models/gan_weights.pt",
        rows=GRID_ROWS,
        cols=GRID_COLS,
        n_ais=AI_COUNT,
    ):
        self.rows = rows
        self.cols = cols
        self.n_ais = n_ais
        self.G = Generator(out_size=rows * cols)
        self.G.eval()

        if os.path.exists(weights_path):
            ckpt = torch.load(weights_path, map_location="cpu")
            self.G.load_state_dict(ckpt["G"])
            print(f"✅ GAN loaded from {weights_path}")
        else:
            print(f"⚠️  No weights found at {weights_path} — using random generator")

    def generate_city(self):
        """
        Returns a city grid dict:
          {
            "rows": 10, "cols": 10,
            "ai_positions": [(r,c), ...],   # hidden from player
            "probability_map": [[float]]    # raw GAN output, used by VAE
          }
        """
        with torch.no_grad():
            z = torch.randn(1, Z_DIM)
            prob = self.G(z).squeeze(0).numpy()  # shape: (rows*cols,)

        prob_2d = prob.reshape(self.rows, self.cols)

        # Convert probabilities → exactly n_ais positions using top-k sampling
        flat_probs = prob.copy()
        ai_indices = self._sample_positions(flat_probs, self.n_ais)
        ai_positions = [
            (int(idx // self.cols), int(idx % self.cols)) for idx in ai_indices
        ]

        return {
            "rows": self.rows,
            "cols": self.cols,
            "ai_positions": ai_positions,
            "probability_map": prob_2d.tolist(),
        }

    def _sample_positions(self, probs, n):
        """Sample n positions weighted by GAN probabilities (no replacement)."""
        probs = np.clip(probs, 1e-6, None)
        probs = probs / probs.sum()
        return np.random.choice(len(probs), size=n, replace=False, p=probs).tolist()


# ── CLI ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--train", action="store_true", help="Train the GAN")
    p.add_argument("--test", action="store_true", help="Test generate a city")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    args = p.parse_args()

    if args.train:
        train_gan(epochs=args.epochs)

    if args.test:
        gan = GANInference()
        city = gan.generate_city()
        print(f"\nGenerated city ({city['rows']}×{city['cols']}):")
        print(f"AI positions ({len(city['ai_positions'])}): {city['ai_positions']}")
        grid = [["." for _ in range(city["cols"])] for _ in range(city["rows"])]
        for r, c in city["ai_positions"]:
            grid[r][c] = "X"
        for row in grid:
            print(" ".join(row))
