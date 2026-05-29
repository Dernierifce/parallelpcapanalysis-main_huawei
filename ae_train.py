"""
ae_train.py
Treino rápido de Autoencoder (dense) sobre o cache de features.
Gera um pickle com tempos de treino e inferência e scores (erro de reconstrução).

Uso:
    python ae_train.py --cache-file ./data/results/features_cache.pkl --outdir ./data/results --outfile ae_results.pkl

Dependências: torch, numpy
"""
import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class AE(nn.Module):
    def __init__(self, n_features, latent=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, max(64, n_features)),
            nn.ReLU(),
            nn.Linear(max(64, n_features), latent),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, max(64, n_features)),
            nn.ReLU(),
            nn.Linear(max(64, n_features), n_features),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def train_ae(X: np.ndarray, device: str = "cpu", epochs=20, batch_size=1024, lr=1e-3):
    n, d = X.shape
    model = AE(d, latent=min(64, d // 2)).to(device)
    ds = TensorDataset(torch.from_numpy(X.astype(np.float32)))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    t0 = time.perf_counter()
    model.train()
    for _ in range(epochs):
        for (batch,) in loader:
            batch = batch.to(device)
            recon = model(batch)
            loss = loss_fn(recon, batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
    train_time = time.perf_counter() - t0

    # inference
    t1 = time.perf_counter()
    model.eval()
    with torch.no_grad():
        X_t = torch.from_numpy(X.astype(np.float32)).to(device)
        recon = model(X_t)
        errs = ((recon - X_t) ** 2).mean(dim=1).cpu().numpy()
    infer_time = time.perf_counter() - t1

    return errs, train_time, infer_time


def main():
    parser = argparse.ArgumentParser(description="Autoencoder benchmark sobre cache de features")
    parser.add_argument("--cache-file", required=True)
    parser.add_argument("--outdir", default=str(Path(__file__).resolve().parent / "data" / "results"))
    parser.add_argument("--outfile", default="ae_results.pkl")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--anom-percentile", type=float, default=95.0,
                        help="Percentil para marcar anomalias pelo erro de reconstrução (default: 95)")
    args = parser.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    with open(args.cache_file, "rb") as f:
        cache = pickle.load(f)

    X = cache["X_scaled"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    errs, train_time, infer_time = train_ae(
        X, device=device, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr
    )

    # reconstruction error thresholds
    p = float(args.anom_percentile)
    perc_thr = np.percentile(errs, p)
    n_anom_perc = int((errs > perc_thr).sum())

    # alternative threshold: mean + 3*std
    mean_thr = errs.mean() + 3 * errs.std()
    n_anom_mean3std = int((errs > mean_thr).sum())

    out = {
        "method": "autoencoder",
        "device": device,
        "args": vars(args),
        "n_flows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_anomalies": n_anom_perc,
        "anom_rate": float(n_anom_perc / X.shape[0]),
        "n_anomalies_mean3std": n_anom_mean3std,
        "anom_rate_mean3std": float(n_anom_mean3std / X.shape[0]),
        "thresholds": {"percentile": float(perc_thr), "mean_3std": float(mean_thr), "percentile_value": p},
        "times": {"train_s": train_time, "infer_s": infer_time, "classification_s": train_time + infer_time},
        "scores": errs,
    }

    out_path = Path(args.outdir) / args.outfile
    with open(out_path, "wb") as f:
        pickle.dump(out, f)

    print(f"Autoencoder salvo em: {out_path}")
    print(f"Treino: {train_time:.2f}s | Infer: {infer_time:.2f}s | Total: {train_time+infer_time:.2f}s")
    print(f"Anomalias (percentil {p}%): {n_anom_perc} ({n_anom_perc/X.shape[0]*100:.2f}%) | Threshold: {perc_thr:.6g}")
    print(f"Anomalias (mean+3std): {n_anom_mean3std} ({n_anom_mean3std/X.shape[0]*100:.2f}%) | Threshold: {mean_thr:.6g}")


if __name__ == '__main__':
    main()
