"""
vae.py — City Node VAE (from scratch, pure PyTorch, no pretrained models)
Trained on CLEAN (no-AI) node feature vectors.
Detects AI presence as high reconstruction error (anomaly = deviation from normal).

Node features per cell (8 values):
  [traffic_flow, latency, packet_loss, cpu_load,
   memory_usage, connection_count, error_rate, uptime]
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

# ── Config ────────────────────────────────────────────────────────────────
INPUT_DIM  = 8     # node feature vector size
LATENT_DIM = 4     # compressed representation
HIDDEN_DIM = 32
LR         = 1e-3
EPOCHS     = 300
BATCH_SIZE = 64
KL_WEIGHT  = 0.5   # β for β-VAE (higher = more disentangled latent space)


# ── Encoder ───────────────────────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, in_dim=INPUT_DIM, hidden=HIDDEN_DIM, latent=LATENT_DIM):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mu_head      = nn.Linear(hidden, latent)
        self.logvar_head  = nn.Linear(hidden, latent)

    def forward(self, x):
        h      = self.shared(x)
        mu     = self.mu_head(h)
        logvar = self.logvar_head(h)
        return mu, logvar


# ── Decoder ───────────────────────────────────────────────────────────────
class Decoder(nn.Module):
    def __init__(self, latent=LATENT_DIM, hidden=HIDDEN_DIM, out_dim=INPUT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
            nn.Sigmoid()   # features are normalised 0–1
        )

    def forward(self, z):
        return self.net(z)


# ── Full VAE ──────────────────────────────────────────────────────────────
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparameterise(self, mu, logvar):
        """z = mu + eps * std  (reparameterisation trick)"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu   # deterministic at inference

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z          = self.reparameterise(mu, logvar)
        recon      = self.decoder(z)
        return recon, mu, logvar

    def reconstruction_error(self, x):
        """
        Returns per-sample MSE reconstruction error.
        High error = anomaly = likely AI-infected node.
        """
        self.eval()
        with torch.no_grad():
            recon, _, _ = self(x)
            return torch.mean((x - recon) ** 2, dim=1)


# ── ELBO Loss ─────────────────────────────────────────────────────────────
def vae_loss(recon, x, mu, logvar, kl_weight=KL_WEIGHT):
    """
    ELBO = Reconstruction loss + β * KL divergence
    Recon: MSE (features are continuous)
    KL:    -0.5 * sum(1 + logvar - mu² - exp(logvar))
    """
    recon_loss = nn.functional.mse_loss(recon, x, reduction="mean")
    kl_loss    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_weight * kl_loss, recon_loss.item(), kl_loss.item()


# ── Synthetic training data ────────────────────────────────────────────────
def generate_clean_nodes(n=5000):
    """
    Synthetic CLEAN node features — no AI present.
    Normal operating ranges (all values 0–1 normalised).
    """
    data = np.zeros((n, INPUT_DIM), dtype=np.float32)

    # traffic_flow: moderate, bell-curve around 0.5
    data[:, 0] = np.clip(np.random.normal(0.50, 0.12, n), 0, 1)
    # latency: low is good, most nodes have low latency
    data[:, 1] = np.clip(np.random.normal(0.20, 0.08, n), 0, 1)
    # packet_loss: very low in clean nodes
    data[:, 2] = np.clip(np.random.normal(0.05, 0.03, n), 0, 1)
    # cpu_load: moderate
    data[:, 3] = np.clip(np.random.normal(0.45, 0.15, n), 0, 1)
    # memory_usage: moderate
    data[:, 4] = np.clip(np.random.normal(0.50, 0.12, n), 0, 1)
    # connection_count: moderate
    data[:, 5] = np.clip(np.random.normal(0.40, 0.15, n), 0, 1)
    # error_rate: very low
    data[:, 6] = np.clip(np.random.normal(0.04, 0.02, n), 0, 1)
    # uptime: high
    data[:, 7] = np.clip(np.random.normal(0.92, 0.06, n), 0, 1)

    return torch.tensor(data)


def generate_infected_nodes(n=1000):
    """
    Synthetic AI-INFECTED node features (used ONLY for evaluation, NOT training).
    Pattern: high traffic, high errors, low uptime — AI draining resources.
    """
    data = np.zeros((n, INPUT_DIM), dtype=np.float32)

    data[:, 0] = np.clip(np.random.normal(0.85, 0.08, n), 0, 1)  # high traffic
    data[:, 1] = np.clip(np.random.normal(0.70, 0.12, n), 0, 1)  # high latency
    data[:, 2] = np.clip(np.random.normal(0.40, 0.15, n), 0, 1)  # high packet loss
    data[:, 3] = np.clip(np.random.normal(0.90, 0.07, n), 0, 1)  # maxed CPU
    data[:, 4] = np.clip(np.random.normal(0.85, 0.08, n), 0, 1)  # high memory
    data[:, 5] = np.clip(np.random.normal(0.80, 0.10, n), 0, 1)  # many connections
    data[:, 6] = np.clip(np.random.normal(0.60, 0.20, n), 0, 1)  # high error rate
    data[:, 7] = np.clip(np.random.normal(0.30, 0.12, n), 0, 1)  # low uptime

    return torch.tensor(data)


# ── Training loop ──────────────────────────────────────────────────────────
def train_vae(save_path="models/vae_weights.pt", epochs=EPOCHS):
    os.makedirs("models", exist_ok=True)

    model     = VAE()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print("Generating clean training data for VAE...")
    clean_data = generate_clean_nodes(n=5000)
    dataset    = torch.utils.data.TensorDataset(clean_data)
    loader     = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f"Training VAE for {epochs} epochs on CLEAN data only...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = recon_sum = kl_sum = 0

        for (batch,) in loader:
            recon, mu, logvar = model(batch)
            loss, recon_l, kl_l = vae_loss(recon, batch, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            recon_sum  += recon_l
            kl_sum     += kl_l

        if epoch % 50 == 0:
            n_batches = len(loader)
            print(f"Epoch {epoch:4d}/{epochs}  "
                  f"Loss={total_loss/n_batches:.4f}  "
                  f"Recon={recon_sum/n_batches:.4f}  "
                  f"KL={kl_sum/n_batches:.4f}")

    # Evaluate: can it distinguish clean vs infected?
    print("\nEvaluating anomaly detection...")
    model.eval()
    clean_errors   = model.reconstruction_error(generate_clean_nodes(500)).numpy()
    infected_errors = model.reconstruction_error(generate_infected_nodes(500)).numpy()

    threshold = np.percentile(clean_errors, 95)   # 95th percentile of clean = threshold
    tp_rate   = (infected_errors > threshold).mean()
    fp_rate   = (clean_errors    > threshold).mean()
    print(f"  Threshold (95th pct clean): {threshold:.4f}")
    print(f"  True positive rate (AI detected): {tp_rate*100:.1f}%")
    print(f"  False positive rate: {fp_rate*100:.1f}%")

    torch.save({
        "model":     model.state_dict(),
        "threshold": float(threshold),
    }, save_path)
    print(f"✅ VAE saved to {save_path}")
    return model, threshold


# ── Inference ──────────────────────────────────────────────────────────────
class VAEInference:
    """
    Loads trained VAE and provides anomaly scores for game cells.
    """
    def __init__(self, weights_path="models/vae_weights.pt"):
        self.model     = VAE()
        self.threshold = 0.05   # default; overridden by saved value
        self.model.eval()

        if os.path.exists(weights_path):
            ckpt = torch.load(weights_path, map_location="cpu")
            self.model.load_state_dict(ckpt["model"])
            self.threshold = ckpt.get("threshold", self.threshold)
            print(f"✅ VAE loaded from {weights_path} (threshold={self.threshold:.4f})")
        else:
            print(f"⚠️  No weights found at {weights_path} — using untrained VAE")

    def score_cell(self, features: list) -> dict:
        """
        Score a single cell.
        features: list of 8 floats [traffic, latency, packet_loss, cpu, mem, conns, errors, uptime]

        Returns:
          {
            "anomaly_score": float 0–1,   # higher = more suspicious
            "is_anomaly": bool,           # True if > threshold
            "confidence": str             # "LOW" / "MEDIUM" / "HIGH"
          }
        """
        x     = torch.tensor([features], dtype=torch.float32)
        error = self.model.reconstruction_error(x).item()

        # Normalise to 0–1 range using threshold as reference
        score = min(error / (self.threshold * 3), 1.0)

        if score < 0.35:
            confidence = "LOW"
        elif score < 0.70:
            confidence = "MEDIUM"
        else:
            confidence = "HIGH"

        return {
            "anomaly_score": round(score, 3),
            "is_anomaly":    error > self.threshold,
            "confidence":    confidence,
        }

    def scan_grid(self, probability_map: list) -> list:
        """
        Score all cells in the grid using GAN probability map as proxy features.
        probability_map: list of lists (rows × cols) with GAN-generated probabilities.

        Returns list of lists with anomaly scores for each cell.
        """
        rows  = len(probability_map)
        cols  = len(probability_map[0])
        scores = []

        for r in range(rows):
            row_scores = []
            for c in range(cols):
                p = probability_map[r][c]

                # Build synthetic feature vector from GAN probability
                # High p → simulate infected node characteristics
                features = [
                    min(0.3 + p * 0.7,  1.0),   # traffic_flow
                    min(0.1 + p * 0.65, 1.0),   # latency
                    min(0.02 + p * 0.45, 1.0),  # packet_loss
                    min(0.4 + p * 0.55, 1.0),   # cpu_load
                    min(0.4 + p * 0.5,  1.0),   # memory_usage
                    min(0.3 + p * 0.55, 1.0),   # connection_count
                    min(0.02 + p * 0.65, 1.0),  # error_rate
                    max(0.95 - p * 0.7,  0.0),  # uptime (inverted)
                ]
                result = self.score_cell(features)
                row_scores.append(result["anomaly_score"])
            scores.append(row_scores)

        return scores

    def single_reveal(self, ai_positions: list, rows: int, cols: int) -> tuple:
        """
        Anomaly detection purchase mechanic:
        Costs points, reveals ONE confirmed AI position.
        Returns (row, col) of revealed AI.
        """
        import random
        return random.choice(ai_positions)


# ── CLI ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--train", action="store_true")
    p.add_argument("--test",  action="store_true")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    args = p.parse_args()

    if args.train:
        train_vae(epochs=args.epochs)

    if args.test:
        vae = VAEInference()
        # Test clean node
        clean    = [0.5, 0.2, 0.05, 0.45, 0.5, 0.4, 0.04, 0.92]
        infected = [0.85, 0.70, 0.40, 0.90, 0.85, 0.80, 0.60, 0.30]
        print("Clean node:", vae.score_cell(clean))
        print("Infected node:", vae.score_cell(infected))
