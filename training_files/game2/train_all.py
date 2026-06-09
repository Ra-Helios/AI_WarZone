"""
train_all.py — Train GAN and VAE for City Sweep in one command.
No pretrained models. No external APIs. Pure PyTorch from scratch.

Usage:
  python train_all.py                          # default epochs
  python train_all.py --gan-epochs 300         # custom GAN epochs
  python train_all.py --vae-epochs 200         # custom VAE epochs
  python train_all.py --quick                  # fast test (50 epochs each)
"""

import argparse
import os
import time


def main():
    parser = argparse.ArgumentParser(description="Train City Sweep models")
    parser.add_argument("--gan-epochs", type=int, default=500)
    parser.add_argument("--vae-epochs", type=int, default=300)
    parser.add_argument("--quick",      action="store_true",
                        help="Quick test run (50 epochs each)")
    args = parser.parse_args()

    if args.quick:
        args.gan_epochs = 50
        args.vae_epochs = 50

    os.makedirs("models", exist_ok=True)

    print("=" * 60)
    print("  CITY SWEEP — Model Training")
    print("  No pretrained models. No APIs. Pure PyTorch.")
    print("=" * 60)

    # ── Train VAE ────────────────────────────────────────────────────
    print(f"\n[1/2] Training VAE (anomaly detector) — {args.vae_epochs} epochs")
    print("      Trains on CLEAN node data only.")
    print("      AI nodes detected as high reconstruction error.\n")
    t0 = time.time()
    from vae import train_vae
    model_vae, threshold = train_vae(
        save_path="models/vae_weights.pt",
        epochs=args.vae_epochs
    )
    print(f"      Done in {time.time()-t0:.1f}s\n")

    # ── Train GAN ────────────────────────────────────────────────────
    print(f"[2/2] Training GAN (city generator) — {args.gan_epochs} epochs")
    print("      Learns to generate realistic city grid layouts.")
    print("      AI positions cluster in urban hotspots.\n")
    t0 = time.time()
    from gan import train_gan
    G, D = train_gan(
        save_path="models/gan_weights.pt",
        epochs=args.gan_epochs
    )
    print(f"      Done in {time.time()-t0:.1f}s\n")

    print("=" * 60)
    print("  ✅ Training complete!")
    print("  Run:  python server.py")
    print("  Then open: http://localhost:5000")
    print("=" * 60)

    # Quick smoke test
    print("\nRunning smoke test...")
    from gan import GANInference
    from vae import VAEInference
    gan = GANInference()
    vae = VAEInference()
    city = gan.generate_city()
    scores = vae.scan_grid(city["probability_map"])
    print(f"  GAN: generated {city['rows']}×{city['cols']} city with {len(city['ai_positions'])} AIs ✅")
    print(f"  VAE: scored {len(scores)}×{len(scores[0])} cells ✅")
    print("  Smoke test passed ✅")


if __name__ == "__main__":
    main()
