"""Reproducible MNIST VQ-VAE with a 32 x 7 x 7 latent and 64 codewords."""
import argparse
import json
import math
import os
import random
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import MNIST
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


class Residual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(nn.ReLU(), nn.Conv2d(channels, 32, 3, padding=1),
                                 nn.ReLU(), nn.Conv2d(32, channels, 1))

    def forward(self, x):
        return x + self.net(x)


class VQVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(), Residual(128),
            nn.ReLU(), nn.Conv2d(128, 32, 1))
        self.codebook = nn.Embedding(64, 32)
        nn.init.uniform_(self.codebook.weight, -1 / 64, 1 / 64)
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 128, 3, padding=1), Residual(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 4, 2, 1))

    def quantize(self, z, mode="nearest", generator=None):
        flat = z.permute(0, 2, 3, 1).reshape(-1, 32)
        codes = self.codebook.weight
        distances = (flat.square().sum(1, keepdim=True)
                     + codes.square().sum(1) - 2 * flat @ codes.T)
        if mode == "random":
            indices = torch.randint(64, (len(flat),), generator=generator).to(z.device)
        elif mode in ("nearest", "second"):
            rank = int(mode == "second")
            indices = distances.argsort(dim=1, stable=True)[:, rank]
        else:
            raise ValueError("Unknown quantization mode")
        quantized = self.codebook(indices).reshape(z.shape[0], 7, 7, 32)
        return quantized.permute(0, 3, 1, 2).contiguous(), indices.reshape(-1, 7, 7)

    def forward(self, x, mode="nearest", generator=None):
        z = self.encoder(x)
        q, indices = self.quantize(z, mode, generator)
        codebook_loss = F.mse_loss(q, z.detach())
        commitment_loss = F.mse_loss(z, q.detach())
        straight_through = z + (q - z).detach()
        return self.decoder(straight_through), codebook_loss, commitment_loss, indices


def load_data(root, train):
    dataset = MNIST(root=str(root), train=train, download=True)
    return dataset.data.unsqueeze(1).float() / 255, dataset.targets


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def code_stats(counts):
    p = counts.double() / counts.sum()
    entropy = -(p[p > 0] * p[p > 0].log()).sum().item()
    return {"active_codes": int((counts > 0).sum()), "perplexity": math.exp(entropy)}


def train(args, device):
    images, _ = load_data(args.data, True)
    # Keep normalized images in host memory; copy only a batch to the device.
    model = VQVAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=args.lr / 10)
    history = []
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(images))
        sums = torch.zeros(3, device=device)
        counts = torch.zeros(64, device=device, dtype=torch.long)
        lr = optimizer.param_groups[0]["lr"]
        for ids in order.split(args.batch_size):
            x = images[ids].to(device)
            optimizer.zero_grad(set_to_none=True)
            recon, code, commit, indices = model(x)
            rec = F.l1_loss(recon, x)
            loss = rec + code + args.beta * commit
            loss.backward()
            optimizer.step()
            sums += torch.stack([rec, code, commit]).detach() * len(ids)
            counts += torch.bincount(indices.flatten(), minlength=64)
        rec, code, commit = (sums / len(images)).tolist()
        row = dict(epoch=epoch + 1, lr=lr, l1=rec, codebook=code,
                   commitment=commit, total=rec + code + args.beta * commit,
                   **code_stats(counts))
        history.append(row)
        scheduler.step()
        print(f"Epoch {epoch + 1:02d}/{args.epochs}: L1={rec:.6f}, "
              f"active={row['active_codes']}, perplexity={row['perplexity']:.2f}", flush=True)
        write_json(args.output / "history.json", history)
        config = dict(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                      beta=args.beta, seed=args.seed, train_samples=len(images))
        torch.save({"state_dict": model.state_dict(), "config": config,
                    "completed_epochs": epoch + 1}, args.output / "model.pt")
    return model, config


def image_metrics(recon, reference):
    error = (recon - reference).flatten(1)
    mse = error.square().mean(1)
    return {"mae": error.abs().mean(1).tolist(),
            "mse": mse.tolist(), "psnr_db": (-10 * mse.clamp_min(1e-12).log10()).tolist()}


def summarize_metrics(values):
    return {key: float(np.mean(value)) for key, value in values.items()}


def grid(path, rows, labels, titles):
    fig, axes = plt.subplots(len(rows), len(titles), squeeze=False,
                             figsize=(1.65 * len(titles), 1.95 * len(rows)))
    for r, row in enumerate(rows):
        for c, img in enumerate(row):
            ax = axes[r, c]
            ax.imshow(img.squeeze(), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(titles[c], fontsize=10)
            if c == 0:
                ax.set_ylabel(labels[r], fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


@torch.inference_mode()
def evaluate(model, config, args, device):
    model.eval()
    figs = args.output / "figures"
    figs.mkdir(exist_ok=True)
    metrics = {"config": config, "parameter_count": sum(p.numel() for p in model.parameters()),
               "noise_sigma": 5 / 255, "noise_clipped": False,
               "noise_seed": config["seed"] + 1, "random_seed": config["seed"] + 2,
               "random_trials": args.trials, "random_mode": "iid_uniform_trained_codewords",
               "random_codebook_seed": config["seed"] + 3}
    untrained = VQVAE().to(device)
    untrained.load_state_dict(model.state_dict())
    fresh_codes = torch.empty(64, 32).uniform_(-1 / 64, 1 / 64,
                  generator=torch.Generator().manual_seed(config["seed"] + 3))
    untrained.codebook.weight.copy_(fresh_codes.to(device))
    untrained.eval()
    arrays = {}
    for split, is_train in [("train", True), ("test", False)]:
        images, labels = load_data(args.data, is_train)
        first = [int((labels == digit).nonzero()[0]) for digit in range(10)]
        x = images[first].to(device)
        recon, _, _, indices = model(x)
        clean = image_metrics(recon.cpu(), images[first])
        metrics[split] = {"first_indices": first, "first_metrics": clean}
        arrays[f"{split}_input"] = images[first].numpy()
        arrays[f"{split}_reconstruction"] = recon.cpu().numpy()
        arrays[f"{split}_indices"] = indices.cpu().numpy()
        grid(figs / f"{split}_digits.png", [images[first], recon.cpu()],
             ["Input", "Reconstruction"], [str(i) for i in range(10)])
        five = images[first[5]:first[5] + 1]
        noise = torch.randn(five.shape, generator=torch.Generator().manual_seed(config["seed"] + 1))
        noisy = five + (5 / 255) * noise
        noisy_rec, _, _, noisy_idx = model(noisy.to(device))
        second, _, _, second_idx = model(five.to(device), "second")
        generator = torch.Generator().manual_seed(config["seed"] + 2)
        random_rec, _, _, random_idx = model(five.to(device), "random", generator)
        fresh_rec, _, _, fresh_idx = untrained(five.to(device))
        clean_five = recon[5:6].cpu()
        metrics[split]["five"] = {
            "clean": summarize_metrics(image_metrics(clean_five, five)),
            "noisy_input": summarize_metrics(image_metrics(noisy, five)),
            "noisy_reconstruction": summarize_metrics(image_metrics(noisy_rec.cpu(), five)),
            "second": summarize_metrics(image_metrics(second.cpu(), five)),
            "random": summarize_metrics(image_metrics(random_rec.cpu(), five)),
            "random_codebook": summarize_metrics(image_metrics(fresh_rec.cpu(), five)),
            "noise_sample_std": noise.std(unbiased=False).item() * 5 / 255,
            "noise_index_change_fraction": (noisy_idx.cpu() != indices[5:6].cpu()).float().mean().item(),
            "second_index_change_fraction": (second_idx.cpu() != indices[5:6].cpu()).float().mean().item()}
        draws, _, _, _ = model(five.repeat(args.trials, 1, 1, 1).to(device), "random",
                              torch.Generator().manual_seed(config["seed"] + 2))
        trial_mae = np.asarray(image_metrics(draws.cpu(), five)["mae"])
        metrics[split]["five"]["random_trials"] = {
            "mae_mean": float(trial_mae.mean()), "mae_std": float(trial_mae.std(ddof=1)),
            "mae_min": float(trial_mae.min()), "mae_max": float(trial_mae.max())}
        metrics[split]["five"]["black_baseline_mae"] = float(five.abs().mean())
        grid(figs / f"{split}_noise.png", [torch.cat([five, noisy, clean_five, noisy_rec.cpu()])],
             ["Digit 5"], ["Clean", "Gaussian noise", "Clean\nreconstruction", "Noisy\nreconstruction"])
        grid(figs / f"{split}_quantization.png",
             [torch.cat([five, clean_five, second.cpu(), random_rec.cpu(), fresh_rec.cpu()])],
             ["Digit 5"], ["Input", "Nearest", "Second nearest", "Random indices", "Random codebook"])
        maps = torch.cat([indices[5:6].cpu(), noisy_idx.cpu(), second_idx.cpu(), random_idx.cpu()])
        fig, axes = plt.subplots(1, 4, figsize=(9, 2.7), layout="constrained")
        for ax, ids, title in zip(axes, maps, ["Nearest", "Noisy nearest", "Second nearest", "Random"]):
            im = ax.imshow(ids, vmin=0, vmax=63, cmap="viridis", interpolation="nearest")
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.colorbar(im, ax=axes, label="Code index", shrink=0.8)
        fig.savefig(figs / f"{split}_indices.png", dpi=200)
        plt.close(fig)
        for name, value in dict(noisy=noisy, noise_reconstruction=noisy_rec.cpu(),
                                second=second.cpu(), random=random_rec.cpu(), maps=maps,
                                random_codebook=fresh_rec.cpu(), random_codebook_indices=fresh_idx.cpu()).items():
            arrays[f"{split}_{name}"] = value.numpy()
        if not is_train:
            full = {key: [] for key in ["mae", "mse", "psnr_db"]}
            counts = torch.zeros(64, dtype=torch.long)
            for batch in images.split(args.batch_size):
                pred, _, _, ids = model(batch.to(device))
                for key, values in image_metrics(pred.cpu(), batch).items():
                    full[key].extend(values)
                counts += torch.bincount(ids.cpu().flatten(), minlength=64)
            metrics["test"]["all"] = {"samples": len(images), **summarize_metrics(full),
                "psnr_from_global_mse_db": -10 * math.log10(float(np.mean(full["mse"]))),
                "mae_standard_error": float(np.std(full["mae"], ddof=1) / math.sqrt(len(images))),
                "black_baseline_mae": float(images.mean()), **code_stats(counts)}
            metrics["test"]["per_digit_mae"] = {
                str(d): float(np.asarray(full["mae"])[labels.numpy() == d].mean()) for d in range(10)}
            metrics["test"]["code_counts"] = counts.tolist()
            fig, ax = plt.subplots(figsize=(9, 3))
            ax.bar(np.arange(64), counts.numpy() / counts.sum().item(), color="#456e9b")
            ax.set(xlabel="Code index", ylabel="Assignment frequency", xlim=(-1, 64))
            fig.tight_layout()
            fig.savefig(figs / "code_usage.png", dpi=200)
            plt.close(fig)
    history = json.loads((args.output / "history.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    epochs = [row["epoch"] for row in history]
    for key, label in [("l1", "L1"), ("codebook", "Codebook MSE"), ("total", "Total")]:
        axes[0].plot(epochs, [row[key] for row in history], label=label)
    axes[0].set(xlabel="Epoch", ylabel="Training loss", yscale="log")
    axes[0].legend()
    for key, label in [("perplexity", "Perplexity"), ("active_codes", "Active codes")]:
        axes[1].plot(epochs, [row[key] for row in history], label=label)
    axes[1].set(xlabel="Epoch", ylabel="Codebook utilization", ylim=(0, 66))
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(figs / "training.png", dpi=200)
    plt.close(fig)
    np.savez_compressed(args.output / "examples.npz", **arrays)
    write_json(args.output / "metrics.json", metrics)
    print(json.dumps(metrics["test"]["all"], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["train", "evaluate", "all"], default="all")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.trials < 2 or args.lr <= 0 or args.beta < 0:
        parser.error("Require positive epochs, batch size and lr; trials >= 2 and beta >= 0")
    args.output.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if args.mode in ("train", "all"):
        model, config = train(args, device)
    else:
        checkpoint = torch.load(args.output / "model.pt", map_location=device, weights_only=True)
        if checkpoint["completed_epochs"] != checkpoint["config"]["epochs"]:
            raise ValueError("Training is incomplete; evaluation requires the final checkpoint")
        model = VQVAE().to(device)
        model.load_state_dict(checkpoint["state_dict"])
        config = checkpoint["config"]
    if args.mode in ("evaluate", "all"):
        evaluate(model, config, args, device)


if __name__ == "__main__":
    main()
