from __future__ import annotations

import argparse
import csv
import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover - handled at runtime
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_FILE = BASE_DIR / "data" / "results" / "features_cache.pkl"
DEFAULT_OUTDIR = BASE_DIR
DEFAULT_SAMPLE_SIZE = 5000
DEFAULT_SYNTHETIC_SAMPLES = 6000
DEFAULT_SYNTHETIC_FEATURES = 16
DEFAULT_AE_EPOCHS = 12
DEFAULT_AE_BATCH_SIZE = 256
DEFAULT_AE_LATENT_DIM = 8
DEFAULT_KMEANS_CLUSTERS = 8
DEFAULT_KMEANS_MAX_ITER = 300


@dataclass
class BenchmarkRow:
    experiment: str
    hardware: str
    train_s: float | None
    infer_s: float | None
    classification_s: float | None
    speedup: float | None
    status: str
    notes: str

if nn is not None:
    class Autoencoder(nn.Module):
        def __init__(self, n_features: int, latent_dim: int):
            super().__init__()
            hidden_dim = max(32, n_features * 2)
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
else:
    Autoencoder = None


def _sync_torch(device: str) -> None:
    if torch is not None and device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def _load_cache(cache_file: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with open(cache_file, "rb") as handle:
        data = pickle.load(handle)

    if "X_scaled" in data:
        X = np.asarray(data["X_scaled"], dtype=np.float32)
    elif "X" in data:
        X = np.asarray(data["X"], dtype=np.float32)
    else:
        raise KeyError("Cache file does not contain X_scaled or X")

    meta = {
        "source": str(cache_file),
        "n_features": int(X.shape[1]),
        "n_rows": int(X.shape[0]),
        "mode": "cache",
    }
    return X, meta


def _load_dataset(cache_file: Path | None, sample_size: int, synthetic_samples: int, synthetic_features: int) -> tuple[np.ndarray, dict[str, Any]]:
    if cache_file is not None and cache_file.exists():
        X, meta = _load_cache(cache_file)
        sampled = False
        if sample_size > 0 and X.shape[0] > sample_size:
            rng = np.random.default_rng(42)
            indices = rng.choice(X.shape[0], size=sample_size, replace=False)
            X = X[indices]
            sampled = True
        meta["sampled"] = sampled
        meta["n_rows_used"] = int(X.shape[0])
        return X, meta

    X, _ = make_blobs(
        n_samples=synthetic_samples,
        n_features=synthetic_features,
        centers=max(2, synthetic_features // 2),
        cluster_std=2.2,
        random_state=42,
    )
    X = StandardScaler().fit_transform(X).astype(np.float32)
    meta = {
        "source": "synthetic",
        "n_features": int(X.shape[1]),
        "n_rows": int(X.shape[0]),
        "n_rows_used": int(X.shape[0]),
        "mode": "synthetic",
        "sampled": False,
    }
    return X, meta


def _run_autoencoder(X: np.ndarray, device: str, epochs: int, batch_size: int, latent_dim: int, lr: float) -> tuple[float, float, np.ndarray]:
    if torch is None or nn is None or DataLoader is None or TensorDataset is None:
        raise RuntimeError("PyTorch is required to run the Autoencoder experiments.")

    model = Autoencoder(X.shape[1], latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    tensor_X = torch.from_numpy(X.astype(np.float32))
    train_loader = DataLoader(
        TensorDataset(tensor_X),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )
    infer_loader = DataLoader(
        TensorDataset(tensor_X),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    _sync_torch(device)
    train_start = time.perf_counter()
    model.train()
    for _ in range(epochs):
        for (batch,) in train_loader:
            batch = batch.to(device, non_blocking=(device == "cuda"))
            recon = model(batch)
            loss = loss_fn(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    _sync_torch(device)
    train_s = time.perf_counter() - train_start

    _sync_torch(device)
    infer_start = time.perf_counter()
    model.eval()
    errors: list[np.ndarray] = []
    with torch.no_grad():
        for (batch,) in infer_loader:
            batch = batch.to(device, non_blocking=(device == "cuda"))
            recon = model(batch)
            batch_errors = ((recon - batch) ** 2).mean(dim=1).detach().cpu().numpy()
            errors.append(batch_errors)
    _sync_torch(device)
    infer_s = time.perf_counter() - infer_start

    return train_s, infer_s, np.concatenate(errors)


def _run_kmeans_cpu(X: np.ndarray, clusters: int, max_iter: int) -> tuple[float, float, np.ndarray]:
    model = KMeans(n_clusters=clusters, max_iter=max_iter, n_init=10, random_state=42)

    train_start = time.perf_counter()
    model.fit(X)
    train_s = time.perf_counter() - train_start

    infer_start = time.perf_counter()
    labels = model.predict(X)
    infer_s = time.perf_counter() - infer_start

    return train_s, infer_s, labels


def _run_kmeans_gpu(X: np.ndarray, clusters: int, max_iter: int) -> tuple[float | None, float | None, np.ndarray | None, str]:
    try:
        from cuml.cluster import KMeans as CuMLKMeans
    except ImportError as exc:
        return None, None, None, f"cuML unavailable: {exc}"

    X_gpu = np.asarray(X, dtype=np.float32)
    model = CuMLKMeans(n_clusters=clusters, max_iter=max_iter, random_state=42)

    train_start = time.perf_counter()
    model.fit(X_gpu)
    train_s = time.perf_counter() - train_start

    infer_start = time.perf_counter()
    labels = model.predict(X_gpu)
    infer_s = time.perf_counter() - infer_start

    labels_np = np.asarray(labels)
    return train_s, infer_s, labels_np, "ok"


def _format_seconds(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}s"


def _build_table(rows: list[BenchmarkRow]) -> tuple[str, list[dict[str, str]]]:
    headers = ["Experimento", "Hardware", "Treino", "Inferencia", "Classificacao", "Speedup", "Status", "Notas"]
    csv_rows: list[dict[str, str]] = []
    table_lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]

    for row in rows:
        csv_row = {
            "experiment": row.experiment,
            "hardware": row.hardware,
            "train_s": _format_seconds(row.train_s),
            "infer_s": _format_seconds(row.infer_s),
            "classification_s": _format_seconds(row.classification_s),
            "speedup": "n/a" if row.speedup is None else f"{row.speedup:.2f}x",
            "status": row.status,
            "notes": row.notes,
        }
        csv_rows.append(csv_row)
        table_lines.append(
            "| "
            + " | ".join(
                [
                    row.experiment,
                    row.hardware,
                    _format_seconds(row.train_s),
                    _format_seconds(row.infer_s),
                    _format_seconds(row.classification_s),
                    "n/a" if row.speedup is None else f"{row.speedup:.2f}x",
                    row.status,
                    row.notes,
                ]
            )
            + " |"
        )

    return "\n".join(table_lines), csv_rows


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_metric(rows: list[BenchmarkRow], metric_name: str, output_path: Path, title: str) -> None:
    labels = [f"{row.experiment}\n{row.hardware}" for row in rows]
    values = [getattr(row, metric_name) if getattr(row, metric_name) is not None else 0.0 for row in rows]
    colors = ["#1d4ed8" if row.hardware == "CPU" else "#16a34a" for row in rows]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=1.0)
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel("Segundos")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, value in zip(bars, values):
        label = f"{value:.2f}s" if value else "n/a"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), label, ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark simplificado do artigo")
    parser.add_argument("--cache-file", default=str(DEFAULT_CACHE_FILE), help="Arquivo de cache com features ou caminho inexistente para usar dados sintéticos")
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR), help="Diretório raiz de saida")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE, help="Numero maximo de amostras usadas por benchmark")
    parser.add_argument("--synthetic-samples", type=int, default=DEFAULT_SYNTHETIC_SAMPLES, help="Numero de amostras sintéticas quando o cache nao existir")
    parser.add_argument("--synthetic-features", type=int, default=DEFAULT_SYNTHETIC_FEATURES, help="Numero de features sintéticas quando o cache nao existir")
    parser.add_argument("--ae-epochs", type=int, default=DEFAULT_AE_EPOCHS)
    parser.add_argument("--ae-batch-size", type=int, default=DEFAULT_AE_BATCH_SIZE)
    parser.add_argument("--ae-latent-dim", type=int, default=DEFAULT_AE_LATENT_DIM)
    parser.add_argument("--ae-lr", type=float, default=1e-3)
    parser.add_argument("--kmeans-clusters", type=int, default=DEFAULT_KMEANS_CLUSTERS)
    parser.add_argument("--kmeans-max-iter", type=int, default=DEFAULT_KMEANS_MAX_ITER)
    parser.add_argument("--run-gpu-autoencoder", action="store_true", help="Executa a versao GPU do Autoencoder quando CUDA estiver disponivel")
    parser.add_argument("--run-gpu-kmeans", action="store_true", help="Executa a versao GPU do K-Means quando cuML estiver disponivel")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    results_dir = outdir / "results"
    figures_dir = outdir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    cache_file = Path(args.cache_file)
    X, meta = _load_dataset(cache_file if cache_file.exists() else None, args.sample_size, args.synthetic_samples, args.synthetic_features)

    print("=" * 72)
    print("Article benchmark runner")
    print(f"Source: {meta['source']}")
    print(f"Rows used: {meta['n_rows_used']}")
    print(f"Features: {meta['n_features']}")
    print("=" * 72)

    rows: list[BenchmarkRow] = []

    ae_cpu_train, ae_cpu_infer, ae_cpu_scores = _run_autoencoder(X, "cpu", args.ae_epochs, args.ae_batch_size, args.ae_latent_dim, args.ae_lr)
    ae_cpu_class = ae_cpu_train + ae_cpu_infer
    rows.append(
        BenchmarkRow(
            experiment="Autoencoder",
            hardware="CPU",
            train_s=ae_cpu_train,
            infer_s=ae_cpu_infer,
            classification_s=ae_cpu_class,
            speedup=1.0,
            status="ok",
            notes=f"threshold sample={len(ae_cpu_scores)}",
        )
    )

    gpu_available = torch is not None and torch.cuda.is_available()
    if args.run_gpu_autoencoder:
        if gpu_available:
            ae_gpu_train, ae_gpu_infer, ae_gpu_scores = _run_autoencoder(X, "cuda", args.ae_epochs, args.ae_batch_size, args.ae_latent_dim, args.ae_lr)
            ae_gpu_class = ae_gpu_train + ae_gpu_infer
            rows.append(
                BenchmarkRow(
                    experiment="Autoencoder",
                    hardware="GPU",
                    train_s=ae_gpu_train,
                    infer_s=ae_gpu_infer,
                    classification_s=ae_gpu_class,
                    speedup=(ae_cpu_class / ae_gpu_class) if ae_gpu_class > 0 else None,
                    status="ok",
                    notes=f"threshold sample={len(ae_gpu_scores)}",
                )
            )
        else:
            rows.append(
                BenchmarkRow(
                    experiment="Autoencoder",
                    hardware="GPU",
                    train_s=None,
                    infer_s=None,
                    classification_s=None,
                    speedup=None,
                    status="unavailable",
                    notes="CUDA unavailable",
                )
            )
    else:
        rows.append(
            BenchmarkRow(
                experiment="Autoencoder",
                hardware="GPU",
                train_s=None,
                infer_s=None,
                classification_s=None,
                speedup=None,
                status="skipped",
                notes="GPU flag not set",
            )
        )

    km_cpu_train, km_cpu_infer, km_cpu_labels = _run_kmeans_cpu(X, args.kmeans_clusters, args.kmeans_max_iter)
    km_cpu_class = km_cpu_train + km_cpu_infer
    rows.append(
        BenchmarkRow(
            experiment="K-Means",
            hardware="CPU",
            train_s=km_cpu_train,
            infer_s=km_cpu_infer,
            classification_s=km_cpu_class,
            speedup=1.0,
            status="ok",
            notes=f"labels={len(km_cpu_labels)}",
        )
    )

    if args.run_gpu_kmeans:
        km_gpu_train, km_gpu_infer, km_gpu_labels, km_gpu_status = _run_kmeans_gpu(X, args.kmeans_clusters, args.kmeans_max_iter)
    else:
        km_gpu_train, km_gpu_infer, km_gpu_labels, km_gpu_status = None, None, None, "GPU flag not set"

    if km_gpu_train is None or km_gpu_infer is None:
        rows.append(
            BenchmarkRow(
                experiment="K-Means",
                hardware="GPU",
                train_s=None,
                infer_s=None,
                classification_s=None,
                speedup=None,
                status="unavailable" if args.run_gpu_kmeans else "skipped",
                notes=km_gpu_status,
            )
        )
    else:
        km_gpu_class = km_gpu_train + km_gpu_infer
        rows.append(
            BenchmarkRow(
                experiment="K-Means",
                hardware="GPU",
                train_s=km_gpu_train,
                infer_s=km_gpu_infer,
                classification_s=km_gpu_class,
                speedup=(km_cpu_class / km_gpu_class) if km_gpu_class > 0 else None,
                status="ok",
                notes=f"labels={len(km_gpu_labels) if km_gpu_labels is not None else 0}",
            )
        )

    table_md, csv_rows = _build_table(rows)

    payload = {
        "source": meta,
        "arguments": vars(args),
        "results": [
            {
                "experiment": row.experiment,
                "hardware": row.hardware,
                "train_s": row.train_s,
                "infer_s": row.infer_s,
                "classification_s": row.classification_s,
                "speedup": row.speedup,
                "status": row.status,
                "notes": row.notes,
            }
            for row in rows
        ],
        "article_table_md": table_md,
    }

    results_json = results_dir / "benchmark_results.json"
    article_md = results_dir / "article_table.md"
    article_csv = results_dir / "article_table.csv"

    with open(results_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)

    article_md.write_text(table_md + "\n", encoding="utf-8")
    _write_csv(article_csv, csv_rows)

    _plot_metric(rows, "train_s", figures_dir / "train_time.png", "Tempo de treino")
    _plot_metric(rows, "infer_s", figures_dir / "inference_time.png", "Tempo de inferencia")
    _plot_metric(rows, "speedup", figures_dir / "speedup.png", "Speedup relativo")

    print(f"Results written to: {results_json}")
    print(f"Table written to: {article_md}")
    print(f"CSV written to: {article_csv}")
    print(f"Figures written to: {figures_dir}")


if __name__ == "__main__":
    main()
