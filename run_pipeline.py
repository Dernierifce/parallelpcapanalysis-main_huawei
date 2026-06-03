"""
run_pipeline.py
Pipeline completo: extração de features + benchmark com log detalhado.

Uso:
    python run_pipeline.py
    python run_pipeline.py --outdir ./data/results --sample-size 10000
    python run_pipeline.py --skip-extraction --cache-file ./data/results/features_cache.pkl

Dependências:
    pip install pyshark pandas numpy torch scikit-learn matplotlib
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    torch = nn = DataLoader = TensorDataset = None
    TORCH_AVAILABLE = False

# ── Caminhos padrão ──────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent
PCAP_DIR    = BASE_DIR / "data" / "pcaps"
DEFAULT_OUT = BASE_DIR / "data" / "results"
DEFAULT_CACHE = DEFAULT_OUT / "features_cache.pkl"

PCAP_FILES = [
    PCAP_DIR / "shard__00091_20260414170212.pcapng",
    PCAP_DIR / "shard__00092_20260414170229.pcapng",
    PCAP_DIR / "shard__00093_20260414170247.pcapng",
    PCAP_DIR / "shard__00094_20260414170303.pcapng",
]

# ── Logger ───────────────────────────────────────────────────────────────────

def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def sep(logger: logging.Logger, title: str = "") -> None:
    line = "=" * 72
    if title:
        logger.info(line)
        logger.info(f"  {title}")
    logger.info(line)


# ── Extração de features ─────────────────────────────────────────────────────

def extract_flows_logged(pcap_path: Path, logger: logging.Logger) -> pd.DataFrame:
    """Chama feature_extractor.extract_flows com log detalhado."""
    sys.path.insert(0, str(BASE_DIR))
    from feature_extractor import extract_flows

    size_mb = pcap_path.stat().st_size / (1024 ** 2)
    logger.info(f"Arquivo     : {pcap_path.name}")
    logger.info(f"Tamanho     : {size_mb:,.1f} MB")

    t0 = time.perf_counter()
    df = extract_flows(str(pcap_path))
    elapsed = time.perf_counter() - t0

    logger.info(f"Fluxos      : {len(df):,}")
    logger.info(f"Features    : {len(df.columns)}")
    logger.info(f"Tempo       : {elapsed:.2f}s  ({elapsed/60:.1f} min)")
    logger.info(f"Taxa        : {len(df)/elapsed:,.0f} fluxos/s")
    logger.debug(df.describe().to_string())
    return df


def run_extraction(pcap_files: list[Path], cache_file: Path,
                   logger: logging.Logger) -> pd.DataFrame:
    sep(logger, "ETAPA 1 — EXTRAÇÃO DE FEATURES")
    logger.info(f"Arquivos pcapng : {len(pcap_files)}")
    logger.info(f"Cache de saída  : {cache_file}")

    all_dfs: list[pd.DataFrame] = []
    total_t0 = time.perf_counter()

    for i, pcap in enumerate(pcap_files, 1):
        logger.info("")
        logger.info(f"[Shard {i}/{len(pcap_files)}] ─────────────────────────────────────")
        if not pcap.exists():
            logger.warning(f"Arquivo não encontrado, pulando: {pcap}")
            continue
        df = extract_flows_logged(pcap, logger)
        all_dfs.append(df)

    if not all_dfs:
        raise RuntimeError("Nenhum arquivo pcapng processado com sucesso.")

    combined = pd.concat(all_dfs, ignore_index=True)
    total_elapsed = time.perf_counter() - total_t0

    logger.info("")
    logger.info("── Resumo da extração ───────────────────────────────────────")
    logger.info(f"Total de shards    : {len(all_dfs)}")
    logger.info(f"Total de fluxos    : {len(combined):,}")
    logger.info(f"Features por fluxo : {len(combined.columns)}")
    logger.info(f"Tempo total        : {total_elapsed:.2f}s  ({total_elapsed/60:.1f} min)")
    logger.info(f"Volume processado  : {sum(p.stat().st_size for p in pcap_files if p.exists()) / (1024**3):.2f} GB")

    # Normaliza e salva cache
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(combined.values.astype(np.float32))
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "wb") as f:
        pickle.dump({"X_scaled": X_scaled, "X": combined.values,
                     "columns": list(combined.columns)}, f)
    logger.info(f"Cache salvo em     : {cache_file}")
    return combined


# ── Autoencoder ───────────────────────────────────────────────────────────────

if TORCH_AVAILABLE:
    class Autoencoder(nn.Module):
        def __init__(self, n_features: int, latent_dim: int):
            super().__init__()
            hidden = max(32, n_features * 2)
            self.encoder = nn.Sequential(
                nn.Linear(n_features, hidden), nn.ReLU(),
                nn.Linear(hidden, latent_dim),  nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, n_features),
            )
        def forward(self, x):
            return self.decoder(self.encoder(x))


def _sync(device: str) -> None:
    if TORCH_AVAILABLE and device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def run_autoencoder(X: np.ndarray, device: str, epochs: int, batch_size: int,
                    latent_dim: int, lr: float,
                    logger: logging.Logger) -> tuple[float, float, np.ndarray]:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch não disponível.")

    model = Autoencoder(X.shape[1], latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    tensor_X = torch.from_numpy(X.astype(np.float32))
    loader_train = DataLoader(TensorDataset(tensor_X), batch_size=batch_size,
                              shuffle=True,  pin_memory=(device=="cuda"))
    loader_infer = DataLoader(TensorDataset(tensor_X), batch_size=batch_size,
                              shuffle=False, pin_memory=(device=="cuda"))

    logger.info(f"  Épocas        : {epochs}")
    logger.info(f"  Batch size    : {batch_size}")
    logger.info(f"  Latent dim    : {latent_dim}")
    logger.info(f"  Learning rate : {lr}")
    logger.info(f"  Amostras      : {len(X):,}")
    logger.info(f"  Device        : {device.upper()}")

    _sync(device)
    t_train = time.perf_counter()
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for (batch,) in loader_train:
            batch = batch.to(device, non_blocking=(device=="cuda"))
            recon = model(batch)
            loss = loss_fn(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch)
        avg_loss = epoch_loss / len(X)
        logger.info(f"  [Autoencoder] Época {epoch:>3}/{epochs} | loss={avg_loss:.6f}")
    _sync(device)
    train_s = time.perf_counter() - t_train

    _sync(device)
    t_infer = time.perf_counter()
    model.eval()
    errors: list[np.ndarray] = []
    with torch.no_grad():
        for (batch,) in loader_infer:
            batch = batch.to(device, non_blocking=(device=="cuda"))
            recon = model(batch)
            errors.append(((recon - batch)**2).mean(dim=1).cpu().numpy())
    _sync(device)
    infer_s = time.perf_counter() - t_infer

    logger.info(f"  Treino        : {train_s:.3f}s")
    logger.info(f"  Inferência    : {infer_s:.3f}s")
    return train_s, infer_s, np.concatenate(errors)


# ── KMeans ────────────────────────────────────────────────────────────────────

def run_kmeans_cpu(X: np.ndarray, clusters: int, max_iter: int,
                   logger: logging.Logger) -> tuple[float, float, np.ndarray]:
    logger.info(f"  Clusters  : {clusters}")
    logger.info(f"  Max iter  : {max_iter}")
    logger.info(f"  Amostras  : {len(X):,}")
    logger.info(f"  Device    : CPU")

    model = KMeans(n_clusters=clusters, max_iter=max_iter, n_init=10,
                   random_state=42, verbose=0)
    t0 = time.perf_counter()
    model.fit(X)
    train_s = time.perf_counter() - t0
    logger.info(f"  Iterações reais   : {model.n_iter_}")
    logger.info(f"  Inércia final     : {model.inertia_:.4f}")
    logger.info(f"  Treino            : {train_s:.3f}s")

    t1 = time.perf_counter()
    labels = model.predict(X)
    infer_s = time.perf_counter() - t1
    unique, counts = np.unique(labels, return_counts=True)
    for c, n in zip(unique, counts):
        logger.info(f"  Cluster {c:>2}        : {n:,} amostras  ({100*n/len(X):.1f}%)")
    logger.info(f"  Inferência        : {infer_s:.3f}s")
    return train_s, infer_s, labels


def run_kmeans_gpu(X: np.ndarray, clusters: int, max_iter: int,
                   logger: logging.Logger) -> tuple[float | None, float | None, np.ndarray | None, str]:
    try:
        from cuml.cluster import KMeans as CuMLKMeans
    except ImportError as e:
        logger.warning(f"  cuML não disponível: {e}")
        return None, None, None, f"cuML unavailable: {e}"

    logger.info(f"  Clusters  : {clusters}")
    logger.info(f"  Max iter  : {max_iter}")
    logger.info(f"  Device    : GPU (cuML)")

    model = CuMLKMeans(n_clusters=clusters, max_iter=max_iter, random_state=42)
    t0 = time.perf_counter()
    model.fit(X.astype(np.float32))
    train_s = time.perf_counter() - t0
    logger.info(f"  Treino    : {train_s:.3f}s")

    t1 = time.perf_counter()
    labels = np.asarray(model.predict(X.astype(np.float32)))
    infer_s = time.perf_counter() - t1
    logger.info(f"  Inferência: {infer_s:.3f}s")
    return train_s, infer_s, labels, "ok"


# ── Benchmark ─────────────────────────────────────────────────────────────────

def run_benchmark(X: np.ndarray, args: argparse.Namespace,
                  logger: logging.Logger) -> list[dict]:
    sep(logger, "ETAPA 2 — BENCHMARK")
    logger.info(f"Amostras usadas : {len(X):,}")
    logger.info(f"Features        : {X.shape[1]}")
    gpu_available = TORCH_AVAILABLE and torch.cuda.is_available()
    logger.info(f"CUDA disponível : {gpu_available}")

    rows: list[dict] = []

    # ── Autoencoder CPU ───────────────────────────────────────────────────────
    logger.info("")
    logger.info("── Autoencoder / CPU ────────────────────────────────────────")
    ae_cpu_train, ae_cpu_infer, ae_cpu_scores = run_autoencoder(
        X, "cpu", args.ae_epochs, args.ae_batch_size,
        args.ae_latent_dim, args.ae_lr, logger)
    rows.append(dict(
        experiment="Autoencoder", hardware="CPU",
        train_s=ae_cpu_train, infer_s=ae_cpu_infer,
        classification_s=ae_cpu_train + ae_cpu_infer,
        speedup=1.0, status="ok",
        notes=f"samples={len(ae_cpu_scores)} | epochs={args.ae_epochs}",
    ))

    # ── Autoencoder GPU ───────────────────────────────────────────────────────
    logger.info("")
    logger.info("── Autoencoder / GPU ────────────────────────────────────────")
    if args.run_gpu_autoencoder:
        if gpu_available:
            ae_gpu_train, ae_gpu_infer, ae_gpu_scores = run_autoencoder(
                X, "cuda", args.ae_epochs, args.ae_batch_size,
                args.ae_latent_dim, args.ae_lr, logger)
            ae_cpu_total = ae_cpu_train + ae_cpu_infer
            ae_gpu_total = ae_gpu_train + ae_gpu_infer
            speedup = ae_cpu_total / ae_gpu_total if ae_gpu_total > 0 else None
            logger.info(f"  Speedup GPU vs CPU : {speedup:.2f}x" if speedup else "  Speedup: n/a")
            rows.append(dict(
                experiment="Autoencoder", hardware="GPU",
                train_s=ae_gpu_train, infer_s=ae_gpu_infer,
                classification_s=ae_gpu_total,
                speedup=speedup, status="ok",
                notes=f"samples={len(ae_gpu_scores)} | epochs={args.ae_epochs}",
            ))
        else:
            logger.warning("  CUDA não disponível — pulando Autoencoder GPU")
            rows.append(dict(experiment="Autoencoder", hardware="GPU",
                             train_s=None, infer_s=None, classification_s=None,
                             speedup=None, status="unavailable", notes="CUDA unavailable"))
    else:
        logger.info("  Flag --run-gpu-autoencoder não definida — pulando")
        rows.append(dict(experiment="Autoencoder", hardware="GPU",
                         train_s=None, infer_s=None, classification_s=None,
                         speedup=None, status="skipped", notes="GPU flag not set"))

    # ── KMeans CPU ────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("── K-Means / CPU ────────────────────────────────────────────")
    km_cpu_train, km_cpu_infer, km_cpu_labels = run_kmeans_cpu(
        X, args.kmeans_clusters, args.kmeans_max_iter, logger)
    rows.append(dict(
        experiment="K-Means", hardware="CPU",
        train_s=km_cpu_train, infer_s=km_cpu_infer,
        classification_s=km_cpu_train + km_cpu_infer,
        speedup=1.0, status="ok",
        notes=f"labels={len(km_cpu_labels)} | clusters={args.kmeans_clusters}",
    ))

    # ── KMeans GPU ────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("── K-Means / GPU ────────────────────────────────────────────")
    if args.run_gpu_kmeans:
        km_gpu_train, km_gpu_infer, km_gpu_labels, km_status = run_kmeans_gpu(
            X, args.kmeans_clusters, args.kmeans_max_iter, logger)
        if km_gpu_train is not None:
            km_cpu_total = km_cpu_train + km_cpu_infer
            km_gpu_total = km_gpu_train + km_gpu_infer
            speedup = km_cpu_total / km_gpu_total if km_gpu_total > 0 else None
            logger.info(f"  Speedup GPU vs CPU : {speedup:.2f}x" if speedup else "  Speedup: n/a")
            rows.append(dict(
                experiment="K-Means", hardware="GPU",
                train_s=km_gpu_train, infer_s=km_gpu_infer,
                classification_s=km_gpu_total,
                speedup=speedup, status="ok",
                notes=f"labels={len(km_gpu_labels)} | clusters={args.kmeans_clusters}",
            ))
        else:
            rows.append(dict(experiment="K-Means", hardware="GPU",
                             train_s=None, infer_s=None, classification_s=None,
                             speedup=None, status="unavailable", notes=km_status))
    else:
        logger.info("  Flag --run-gpu-kmeans não definida — pulando")
        rows.append(dict(experiment="K-Means", hardware="GPU",
                         train_s=None, infer_s=None, classification_s=None,
                         speedup=None, status="skipped", notes="GPU flag not set"))

    return rows


# ── Relatório final ───────────────────────────────────────────────────────────

def write_results(rows: list[dict], args: argparse.Namespace,
                  outdir: Path, logger: logging.Logger) -> None:
    sep(logger, "ETAPA 3 — COMPARAÇÃO DE MÉTODOS")

    headers = ["Experimento", "Hardware", "Treino(s)", "Inferência(s)",
               "Total(s)", "Speedup", "Status", "Notas"]

    col_w = [14, 10, 12, 14, 10, 10, 12, 40]

    def fmt_row(vals):
        return "  ".join(str(v).ljust(w) for v, w in zip(vals, col_w))

    logger.info("")
    logger.info(fmt_row(headers))
    logger.info("  " + "-" * (sum(col_w) + 2 * len(col_w)))

    for r in rows:
        def fs(v): return f"{v:.3f}" if isinstance(v, float) else ("n/a" if v is None else str(v))
        speedup_str = f"{r['speedup']:.2f}x" if isinstance(r.get("speedup"), float) else "n/a"
        logger.info(fmt_row([
            r["experiment"], r["hardware"],
            fs(r["train_s"]), fs(r["infer_s"]), fs(r["classification_s"]),
            speedup_str, r["status"], r["notes"],
        ]))

    # Calcula e loga speedups comparativos
    logger.info("")
    logger.info("── Speedups calculados ──────────────────────────────────────")
    by_exp: dict[str, dict] = {}
    for r in rows:
        key = r["experiment"]
        if key not in by_exp:
            by_exp[key] = {}
        by_exp[key][r["hardware"]] = r

    for exp, hw_map in by_exp.items():
        cpu = hw_map.get("CPU")
        gpu = hw_map.get("GPU")
        if cpu and gpu and cpu.get("classification_s") and gpu.get("classification_s"):
            sp = cpu["classification_s"] / gpu["classification_s"]
            logger.info(f"  {exp}: CPU={cpu['classification_s']:.3f}s  GPU={gpu['classification_s']:.3f}s  Speedup={sp:.2f}x")
        elif cpu:
            logger.info(f"  {exp}: CPU={cpu['classification_s']:.3f}s  GPU=n/a")

    # Salva arquivos
    outdir.mkdir(parents=True, exist_ok=True)

    csv_path = outdir / "benchmark_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = outdir / "benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"arguments": vars(args), "results": rows}, f,
                  indent=2, ensure_ascii=False, default=str)

    logger.info("")
    logger.info(f"CSV salvo em  : {csv_path}")
    logger.info(f"JSON salvo em : {json_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline completo: extração + benchmark")
    parser.add_argument("--outdir",           default=str(DEFAULT_OUT))
    parser.add_argument("--cache-file",       default=str(DEFAULT_CACHE))
    parser.add_argument("--skip-extraction",  action="store_true",
                        help="Pula extração e usa cache existente")
    parser.add_argument("--sample-size",      type=int,   default=5000)
    parser.add_argument("--ae-epochs",        type=int,   default=12)
    parser.add_argument("--ae-batch-size",    type=int,   default=256)
    parser.add_argument("--ae-latent-dim",    type=int,   default=8)
    parser.add_argument("--ae-lr",            type=float, default=1e-3)
    parser.add_argument("--kmeans-clusters",  type=int,   default=8)
    parser.add_argument("--kmeans-max-iter",  type=int,   default=300)
    parser.add_argument("--run-gpu-autoencoder", action="store_true")
    parser.add_argument("--run-gpu-kmeans",      action="store_true")
    args = parser.parse_args()

    outdir     = Path(args.outdir)
    cache_file = Path(args.cache_file)
    log_path   = outdir / "pipeline.log"

    logger = setup_logger(log_path)

    run_start = time.perf_counter()
    sep(logger, "PIPELINE INICIADO")
    logger.info(f"Data/hora   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Python      : {sys.version.split()[0]}")
    logger.info(f"PyTorch     : {torch.__version__ if TORCH_AVAILABLE else 'não instalado'}")
    logger.info(f"CUDA        : {torch.cuda.is_available() if TORCH_AVAILABLE else False}")
    logger.info(f"Outdir      : {outdir}")
    logger.info(f"Cache file  : {cache_file}")
    logger.info(f"Sample size : {args.sample_size:,}")

    # ── Extração ──────────────────────────────────────────────────────────────
    if not args.skip_extraction:
        run_extraction(PCAP_FILES, cache_file, logger)
    else:
        logger.info("")
        logger.info("Extração pulada — usando cache existente.")

    # ── Carrega dados ─────────────────────────────────────────────────────────
    sep(logger, "CARREGANDO DADOS DO CACHE")
    if not cache_file.exists():
        logger.error(f"Cache não encontrado: {cache_file}")
        sys.exit(1)

    with open(cache_file, "rb") as f:
        data = pickle.load(f)

    if "X_scaled" in data:
        X = np.asarray(data["X_scaled"], dtype=np.float32)
    elif "X" in data:
        X = StandardScaler().fit_transform(
            np.asarray(data["X"], dtype=np.float32))
    else:
        raise KeyError("Cache não contém X_scaled ou X")

    logger.info(f"Total no cache  : {X.shape[0]:,} amostras x {X.shape[1]} features")

    if args.sample_size > 0 and X.shape[0] > args.sample_size:
        rng = np.random.default_rng(42)
        idx = rng.choice(X.shape[0], size=args.sample_size, replace=False)
        X = X[idx]
        logger.info(f"Amostrado para  : {len(X):,} (--sample-size={args.sample_size})")

    # ── Benchmark ─────────────────────────────────────────────────────────────
    rows = run_benchmark(X, args, logger)

    # ── Resultados ────────────────────────────────────────────────────────────
    write_results(rows, args, outdir, logger)

    # ── Encerramento ──────────────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - run_start
    sep(logger, "PIPELINE CONCLUÍDO")
    logger.info(f"Tempo total  : {total_elapsed:.2f}s  ({total_elapsed/60:.1f} min)")
    logger.info(f"Log salvo em : {log_path}")
    sep(logger)


if __name__ == "__main__":
    main()
