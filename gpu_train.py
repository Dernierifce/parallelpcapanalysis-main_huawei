"""
gpu_train.py
Benchmark em GPU usando Autoencoder em PyTorch.

Uso recomendado:
    python gpu_train.py --cache-file ./data/results/features_cache.pkl --outdir ./data/results --outfile gpu_results.pkl

Modo legado:
    python gpu_train.py --shards ./data/pcaps/*.pcapng --outdir ./data/results --outfile gpu_results.pkl
"""

import argparse
import glob
import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from feature_extractor import FEATURE_COLS, extract_flows
from log_utils import emit_report, setup_run_logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SHARDS = sorted(glob.glob(str(BASE_DIR / "data" / "pcaps" / "*.pcapng")))
DEFAULT_OUTDIR = str(BASE_DIR / "data" / "results")
DEFAULT_OUTFILE = "gpu_results.pkl"


class Autoencoder(nn.Module):
    def __init__(self, n_features: int, latent_dim: int):
        super().__init__()
        hidden_dim = max(64, n_features)
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_features),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def _load_features(cache_file: str | None, shards: list[str]) -> tuple[np.ndarray, float, list[dict], str]:
    if cache_file:
        with open(cache_file, "rb") as f:
            cache = pickle.load(f)
        return cache["X_scaled"], 0.0, cache.get("shard_stats", []), "cache"

    extraction_start = time.perf_counter()
    all_dfs = []
    shard_stats = []

    for shard in sorted(shards):
        t0 = time.perf_counter()
        df = extract_flows(shard, anonymize=True)
        elapsed = time.perf_counter() - t0
        all_dfs.append(df)
        shard_stats.append({"shard_path": str(shard), "n_flows": int(len(df)), "extract_s": elapsed})
        print(f"  [EXTRACT] {Path(shard).name:30s} | flows={len(df):,} | t={elapsed:.1f}s")

    if not all_dfs:
        raise ValueError("Nenhum shard informado/encontrado. Verifique --shards.")

    merged = np.vstack([df[FEATURE_COLS].fillna(0).values for df in all_dfs])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(merged)
    return X_scaled, time.perf_counter() - extraction_start, shard_stats, "full"


def train_autoencoder(X_scaled: np.ndarray, epochs: int, batch_size: int, lr: float, latent_dim: int, device: str):
    n_features = X_scaled.shape[1]
    model = Autoencoder(n_features, latent_dim).to(device)
    dataset = TensorDataset(torch.from_numpy(X_scaled.astype(np.float32)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    t_train_start = time.perf_counter()
    model.train()
    for _ in range(epochs):
        for (batch,) in loader:
            batch = batch.to(device)
            recon = model(batch)
            loss = loss_fn(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    train_time = time.perf_counter() - t_train_start

    t_infer_start = time.perf_counter()
    model.eval()
    with torch.no_grad():
        X_t = torch.from_numpy(X_scaled.astype(np.float32)).to(device)
        recon = model(X_t)
        errors = ((recon - X_t) ** 2).mean(dim=1).cpu().numpy()
    infer_time = time.perf_counter() - t_infer_start

    return errors, train_time, infer_time


def main():
    parser = argparse.ArgumentParser(description="Benchmark GPU com Autoencoder em PyTorch")
    parser.add_argument("--shards", nargs="+", default=DEFAULT_SHARDS)
    parser.add_argument("--cache-file", default=None, help="Cache gerado por preprocess_features.py")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument("--outfile", default=DEFAULT_OUTFILE)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--anom-percentile", type=float, default=95.0)
    parser.add_argument("--log-file", default=None, help="Arquivo de log opcional")
    args = parser.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    log_path = setup_run_logging(args.outdir, "gpu_train", args.log_file)

    print("=" * 70)
    print("  Benchmark GPU — Autoencoder")
    print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Log: {log_path}")
    print("=" * 70)

    X_scaled, extraction_time, shard_stats, mode = _load_features(args.cache_file, args.shards)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    classification_start = time.perf_counter()
    errors, train_time, infer_time = train_autoencoder(
        X_scaled,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        latent_dim=args.latent_dim,
        device=device,
    )
    classification_time = time.perf_counter() - classification_start
    total_time = extraction_time + classification_time

    percentile = float(args.anom_percentile)
    threshold = float(np.percentile(errors, percentile))
    labels = np.where(errors > threshold, -1, 1)
    n_anom = int((labels == -1).sum())

    out = {
        "method": "autoencoder",
        "backend": device,
        "mode": mode,
        "args": vars(args),
        "n_flows": int(X_scaled.shape[0]),
        "n_features": int(X_scaled.shape[1]),
        "n_anomalies": n_anom,
        "anom_rate": float(n_anom / X_scaled.shape[0]),
        "thresholds": {"percentile": percentile, "value": threshold},
        "times": {
            "extract_s": extraction_time,
            "train_s": train_time,
            "infer_s": infer_time,
            "classification_s": classification_time,
            "total_s": total_time,
        },
        "shards": shard_stats,
        "scores": errors,
        "labels": labels,
    }

    out_path = Path(args.outdir) / args.outfile
    with open(out_path, "wb") as f:
        pickle.dump(out, f)

    emit_report(
        "Relatório detalhado — gpu_train",
        {
            "Configuração": {
                "metodo": out["method"],
                "backend": device,
                "mode": mode,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "latent_dim": args.latent_dim,
                "anom_percentile": percentile,
            },
            "Volume processado": {
                "fluxos": int(X_scaled.shape[0]),
                "features": int(X_scaled.shape[1]),
                "anomalias": f"{n_anom:,}",
                "taxa_anomalia_pct": f"{n_anom / X_scaled.shape[0] * 100:.2f}%",
            },
            "Tempos": {
                "extracao_s": round(extraction_time, 3),
                "treino_s": round(train_time, 3),
                "inferencia_s": round(infer_time, 3),
                "classificacao_s": round(classification_time, 3),
                "total_s": round(total_time, 3),
            },
            "Shards": [
                f"{Path(item['shard_path']).name}: fluxos={item['n_flows']:,} | extração={item.get('extract_s', 0.0):.2f}s"
                for item in shard_stats
            ] or ["Sem shards detalhados disponíveis"],
            "Resultado": {
                "threshold": f"p{percentile:.1f} -> {threshold:.6g}",
                "arquivo_saida": out_path,
                "classificacao_metrica_principal": f"{classification_time:.1f}s (train + infer)",
            },
        },
    )


if __name__ == "__main__":
    main()
