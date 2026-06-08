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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

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
    """
    K-Means GPU implementado em PyTorch puro (sem cuML).
    Algoritmo Lloyd com operações vetorizadas na GPU via torch.
    """
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        msg = "CUDA não disponível"
        logger.warning(f"  {msg}")
        return None, None, None, msg

    logger.info(f"  Clusters  : {clusters}")
    logger.info(f"  Max iter  : {max_iter}")
    logger.info(f"  Amostras  : {len(X):,}")
    logger.info(f"  Device    : GPU (PyTorch) — {torch.cuda.get_device_name(0)}")

    device = "cuda"
    X_t = torch.from_numpy(X.astype(np.float32)).to(device)

    # ── Treino ────────────────────────────────────────────────────────────────
    torch.cuda.synchronize()
    t_train = time.perf_counter()

    # Inicialização kmeans++ na GPU
    rng = torch.Generator(device=device)
    rng.manual_seed(42)
    first_idx = torch.randint(0, X_t.shape[0], (1,), generator=rng, device=device).item()
    centroids = X_t[first_idx].unsqueeze(0)

    for _ in range(1, clusters):
        dists = torch.cdist(X_t, centroids).min(dim=1).values
        probs = dists / dists.sum()
        idx = torch.multinomial(probs, 1, generator=rng).item()
        centroids = torch.cat([centroids, X_t[idx].unsqueeze(0)], dim=0)

    inertia_history: list[float] = []
    labels_t = torch.zeros(X_t.shape[0], dtype=torch.long, device=device)

    for iteration in range(max_iter):
        # Atribuição
        dists = torch.cdist(X_t, centroids)
        new_labels = dists.argmin(dim=1)

        # Inércia
        inertia = dists.min(dim=1).values.pow(2).sum().item()
        inertia_history.append(inertia)

        # Convergência
        if iteration > 0 and torch.equal(new_labels, labels_t):
            logger.info(f"  Convergiu em {iteration + 1} iterações")
            labels_t = new_labels
            break
        labels_t = new_labels

        # Atualização de centroides
        for k in range(clusters):
            mask = labels_t == k
            if mask.any():
                centroids[k] = X_t[mask].mean(dim=0)

    torch.cuda.synchronize()
    train_s = time.perf_counter() - t_train

    logger.info(f"  Inércia final     : {inertia_history[-1]:.4f}")
    logger.info(f"  Iterações reais   : {len(inertia_history)}")
    logger.info(f"  Treino            : {train_s:.3f}s")

    # ── Inferência ────────────────────────────────────────────────────────────
    torch.cuda.synchronize()
    t_infer = time.perf_counter()
    dists_final = torch.cdist(X_t, centroids)
    labels_final = dists_final.argmin(dim=1)
    torch.cuda.synchronize()
    infer_s = time.perf_counter() - t_infer

    labels_np = labels_final.cpu().numpy()

    unique, counts = np.unique(labels_np, return_counts=True)
    for c, n in zip(unique, counts):
        logger.info(f"  Cluster {c:>2}        : {n:,} amostras  ({100*n/len(X):.1f}%)")
    logger.info(f"  Inferência        : {infer_s:.3f}s")

    return train_s, infer_s, labels_np, "ok (PyTorch GPU)"


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



# ── Geração de relatório visual ───────────────────────────────────────────────

def _parse_log(log_path: Path) -> dict:
    """Extrai dados estruturados do pipeline.log."""
    data: dict = {
        "meta": {},
        "shards": [],
        "ae_epochs": [],
        "ae_loss": [],
        "ae_cpu": {},
        "ae_gpu": {},
        "km_cpu": {"clusters": {}},
        "km_gpu": {"clusters": {}},
        "bench_rows": [],
        "total_min": None,
        "cuda": "False",
        "gpu_name": "",
    }
    if not log_path.exists():
        return data

    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    current_shard: dict | None = None
    current_sub = ""   # ae_cpu | ae_gpu | km_cpu | km_gpu

    def store(key: str, field: str, val):
        data[key][field] = val

    for line in lines:
        msg = line.split(" | ", 2)[-1].strip() if " | " in line else line.strip()

        # ── Meta ──────────────────────────────────────────────────────────────
        if msg.startswith("CUDA        :"):
            data["cuda"] = msg.split(":", 1)[1].strip()
        elif msg.startswith("PyTorch     :"):
            data["meta"]["pytorch"] = msg.split(":", 1)[1].strip()
        elif msg.startswith("Python      :"):
            data["meta"]["python"] = msg.split(":", 1)[1].strip()
        elif msg.startswith("Sample size :"):
            data["meta"]["sample_size"] = msg.split(":", 1)[1].strip()
        elif "Device    : GPU (PyTorch)" in msg and "─" in msg:
            data["gpu_name"] = msg.split("─", 1)[1].strip(" ─")

        # ── Shards ────────────────────────────────────────────────────────────
        elif msg.startswith("[Shard "):
            current_shard = {}
        elif current_shard is not None:
            if msg.startswith("Arquivo     :"):
                current_shard["name"] = msg.split(":", 1)[1].strip()
            elif msg.startswith("Tamanho     :"):
                try:
                    current_shard["size_mb"] = float(
                        msg.split(":", 1)[1].strip().replace(",", "").replace(" MB", ""))
                except Exception: pass
            elif msg.startswith("Fluxos      :"):
                try:
                    current_shard["flows"] = int(msg.split(":", 1)[1].strip().replace(",", ""))
                except Exception: pass
            elif msg.startswith("Tempo       :") and "(" in msg:
                try:
                    current_shard["time_min"] = float(msg.split("(")[1].split(" min")[0])
                    data["shards"].append(dict(current_shard))
                    current_shard = None
                except Exception: pass

        # ── Subsection markers ────────────────────────────────────────────────
        if "Autoencoder / CPU" in msg:
            current_sub = "ae_cpu"
        elif "Autoencoder / GPU" in msg:
            current_sub = "ae_gpu"
        elif "K-Means / CPU" in msg:
            current_sub = "km_cpu"
        elif "K-Means / GPU" in msg:
            current_sub = "km_gpu"

        # ── Autoencoder epochs ────────────────────────────────────────────────
        if "[Autoencoder] Época" in msg:
            try:
                parts = msg.split("|")
                epoch_part = parts[0].strip().split()[-1].split("/")[0]
                loss_part  = parts[1].strip().split("=")[1]
                data["ae_epochs"].append(int(epoch_part))
                data["ae_loss"].append(float(loss_part))
            except Exception: pass

        # ── Treino ────────────────────────────────────────────────────────────
        if "Treino" in msg and ":" in msg and current_sub:
            try:
                val_str = msg.split(":")[-1].strip().replace("s", "").strip()
                val = float(val_str)
                data[current_sub]["train_s"] = val
            except Exception: pass

        # ── Inferência ────────────────────────────────────────────────────────
        if "Inferência" in msg and ":" in msg and current_sub:
            try:
                val_str = msg.split(":")[-1].strip().replace("s", "").strip()
                val = float(val_str)
                data[current_sub]["infer_s"] = val
            except Exception: pass

        # ── Clusters ──────────────────────────────────────────────────────────
        if "Cluster" in msg and "amostras" in msg and current_sub in ("km_cpu","km_gpu"):
            try:
                parts = msg.split(":")
                cluster_id = int(parts[0].strip().split()[-1])
                count = int(parts[1].strip().split()[0].replace(",", ""))
                pct   = float(parts[1].strip().split("(")[1].replace("%)", ""))
                data[current_sub]["clusters"][cluster_id] = {"count": count, "pct": pct}
            except Exception: pass

        # ── Inércia ───────────────────────────────────────────────────────────
        if "Inércia final" in msg and current_sub in ("km_cpu","km_gpu"):
            try:
                data[current_sub]["inertia"] = float(msg.split(":")[-1].strip())
            except Exception: pass

        # ── Iterações ─────────────────────────────────────────────────────────
        if ("Iterações reais" in msg or "Convergiu em" in msg) and current_sub in ("km_cpu","km_gpu"):
            try:
                if "Convergiu em" in msg:
                    val = int(msg.split("Convergiu em")[1].split("iter")[0].strip())
                else:
                    val = int(msg.split(":")[-1].strip())
                data[current_sub]["n_iter"] = val
            except Exception: pass

        # ── Speedup ───────────────────────────────────────────────────────────
        if "Speedup GPU vs CPU" in msg and current_sub in ("ae_gpu","km_gpu"):
            try:
                data[current_sub]["speedup"] = float(msg.split(":")[-1].strip().replace("x",""))
            except Exception: pass

        # ── Tempo total ───────────────────────────────────────────────────────
        if msg.startswith("Tempo total  :") and "min" in msg:
            try:
                data["total_min"] = float(msg.split("(")[1].split(" min")[0])
            except Exception: pass

    return data


def plot_report(log_path: Path, out_path: Path, logger: logging.Logger) -> None:
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib não disponível — relatório visual não gerado.")
        return

    logger.info("Gerando relatório visual...")
    d = _parse_log(log_path)

    # ── Paleta ───────────────────────────────────────────────────────────────
    DARK   = "#0f172a"; PANEL  = "#1e293b"; GRID   = "#334155"
    TEXT   = "#f1f5f9"; MUTED  = "#94a3b8"
    BLUE   = "#3b82f6"; GREEN  = "#22c55e"; AMBER  = "#f59e0b"
    PURPLE = "#a855f7"; RED    = "#ef4444"; CYAN   = "#06b6d4"
    SHARD_COLORS = [BLUE, GREEN, AMBER, PURPLE]

    fig = plt.figure(figsize=(20, 28), facecolor=DARK)
    fig.patch.set_facecolor(DARK)
    gs  = gridspec.GridSpec(5, 2, figure=fig, hspace=0.55, wspace=0.35,
                            top=0.95, bottom=0.04, left=0.07, right=0.96)

    def style(ax, title):
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values(): sp.set_edgecolor(GRID)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.set_title(title, color=TEXT, fontsize=11, fontweight="bold", pad=10)
        ax.grid(axis="y", color=GRID, linestyle="--", linewidth=0.6, alpha=0.6)

    cuda_ok  = d["cuda"].lower() == "true"
    gpu_label = d["gpu_name"] or ("RTX 4060" if cuda_ok else "N/A")
    pytorch  = d["meta"].get("pytorch", "N/A")
    python   = d["meta"].get("python",  "N/A")
    total_m  = f"{d['total_min']:.1f} min" if d["total_min"] else "N/A"
    ts_str   = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ── 0. Header ─────────────────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, :])
    ax0.set_facecolor(PANEL)
    for sp in ax0.spines.values(): sp.set_edgecolor(BLUE)
    ax0.set_xticks([]); ax0.set_yticks([])
    ax0.text(0.5, 0.72, "Parallel PCAP Analysis — Relatório de Execução",
             transform=ax0.transAxes, ha="center", color=TEXT,
             fontsize=16, fontweight="bold")
    subtitle = (f"Gerado: {ts_str}  |  Duração total: {total_m}  |  "
                f"Python {python}  |  PyTorch {pytorch}  |  "
                f"CUDA: {d['cuda']}  |  GPU: {gpu_label}")
    ax0.text(0.5, 0.30, subtitle, transform=ax0.transAxes,
             ha="center", color=MUTED, fontsize=9.5)

    # ── 1. Tempo de extração por shard ────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1, 0])
    style(ax1, "Extração — Tempo por Shard (min)")
    if d["shards"]:
        names  = [s["name"].replace("shard__","S").split("_")[0][:10] for s in d["shards"]]
        times  = [s.get("time_min", 0) for s in d["shards"]]
        colors = SHARD_COLORS[:len(names)]
        bars = ax1.bar(names, times, color=colors, edgecolor=DARK, linewidth=1, width=0.55)
        for bar, t in zip(bars, times):
            ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.4,
                     f"{t:.1f}m", ha="center", color=TEXT, fontsize=9, fontweight="bold")
    ax1.set_ylabel("Minutos", color=MUTED, fontsize=9)
    ax1.tick_params(axis="x", colors=TEXT, labelsize=8)

    # ── 2. Fluxos por shard ───────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 1])
    style(ax2, "Extração — Fluxos Extraídos por Shard")
    if d["shards"]:
        flows  = [s.get("flows", 0) for s in d["shards"]]
        bars2  = ax2.bar(names, flows, color=colors, edgecolor=DARK, linewidth=1, width=0.55)
        for bar, f in zip(bars2, flows):
            ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+200,
                     f"{f:,}", ha="center", color=TEXT, fontsize=9, fontweight="bold")
        total_f = sum(flows)
        ax2.text(0.98, 0.95, f"Total: {total_f:,}", transform=ax2.transAxes,
                 ha="right", color=AMBER, fontsize=10, fontweight="bold")
    ax2.set_ylabel("Nº de Fluxos", color=MUTED, fontsize=9)
    ax2.tick_params(axis="x", colors=TEXT, labelsize=8)

    # ── 3. Convergência Autoencoder ───────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, 0])
    style(ax3, "Autoencoder — Convergência Loss por Época")
    if d["ae_epochs"] and d["ae_loss"]:
        ax3.plot(d["ae_epochs"], d["ae_loss"], color=BLUE, linewidth=2.5,
                 marker="o", markersize=6, markerfacecolor=TEXT,
                 markeredgecolor=BLUE, zorder=3)
        ax3.fill_between(d["ae_epochs"], d["ae_loss"], alpha=0.15, color=BLUE)
        for e, l in zip(d["ae_epochs"], d["ae_loss"]):
            ax3.annotate(f"{l:.3f}", (e, l), textcoords="offset points",
                         xytext=(0, 8), ha="center", color=MUTED, fontsize=7.5)
        cpu_t = d["ae_cpu"].get("train_s", 0)
        cpu_i = d["ae_cpu"].get("infer_s", 0)
        gpu_t = d["ae_gpu"].get("train_s")
        sp    = d["ae_gpu"].get("speedup")
        info  = f"CPU treino: {cpu_t:.3f}s\nCPU infer: {cpu_i:.3f}s"
        if gpu_t:
            info += f"\nGPU treino: {gpu_t:.3f}s"
        if sp:
            info += f"\nSpeedup: {sp:.2f}x"
        ax3.text(0.97, 0.92, info, transform=ax3.transAxes, ha="right",
                 color=GREEN, fontsize=8.5, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor=DARK,
                           edgecolor=GREEN, alpha=0.85))
    ax3.set_xlabel("Época", color=MUTED, fontsize=9)
    ax3.set_ylabel("MSE Loss", color=MUTED, fontsize=9)
    if d["ae_epochs"]:
        ax3.set_xticks(d["ae_epochs"])
    ax3.tick_params(axis="x", colors=TEXT)

    # ── 4. K-Means distribuição CPU vs GPU ────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 1])
    style(ax4, "K-Means — Distribuição de Clusters")
    cpu_cl = d["km_cpu"].get("clusters", {})
    gpu_cl = d["km_gpu"].get("clusters", {})
    cluster_ids = sorted(set(list(cpu_cl.keys()) + list(gpu_cl.keys())))
    if cluster_ids:
        x = np.arange(len(cluster_ids))
        w = 0.38 if gpu_cl else 0.6
        cpu_counts = [cpu_cl.get(c, {}).get("count", 0) for c in cluster_ids]
        bars_cpu = ax4.bar(x - (w/2 if gpu_cl else 0), cpu_counts,
                           width=w, color=BLUE, edgecolor=DARK, linewidth=1,
                           label="CPU")
        if gpu_cl:
            gpu_counts = [gpu_cl.get(c, {}).get("count", 0) for c in cluster_ids]
            ax4.bar(x + w/2, gpu_counts, width=w, color=GREEN,
                    edgecolor=DARK, linewidth=1, label="GPU")
            ax4.legend(facecolor=PANEL, edgecolor=GRID,
                       labelcolor=TEXT, fontsize=8)
        ax4.set_xticks(x)
        ax4.set_xticklabels([f"C{c}" for c in cluster_ids], color=TEXT, fontsize=8)
        inertia = d["km_cpu"].get("inertia")
        n_iter  = d["km_cpu"].get("n_iter")
        km_t    = d["km_cpu"].get("train_s", 0)
        gkm_t   = d["km_gpu"].get("train_s")
        gkm_sp  = d["km_gpu"].get("speedup")
        info = f"Inércia: {inertia:.2f}\n" if inertia else ""
        info += f"Iter: {n_iter}\n" if n_iter else ""
        info += f"CPU: {km_t:.3f}s"
        if gkm_t:  info += f"\nGPU: {gkm_t:.3f}s"
        if gkm_sp: info += f"\nSpeedup: {gkm_sp:.2f}x"
        ax4.text(0.97, 0.92, info, transform=ax4.transAxes, ha="right",
                 color=AMBER, fontsize=8.5, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor=DARK,
                           edgecolor=AMBER, alpha=0.85))
    ax4.set_ylabel("Amostras", color=MUTED, fontsize=9)

    # ── 5. Tabela comparativa ─────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[3, :])
    ax5.set_facecolor(PANEL)
    for sp in ax5.spines.values(): sp.set_edgecolor(GRID)
    ax5.set_xticks([]); ax5.set_yticks([])
    ax5.set_title("Comparativo de Métodos — CPU vs GPU", color=TEXT,
                  fontsize=11, fontweight="bold", pad=10)

    def fv(v): return f"{v:.3f}s" if isinstance(v, (int, float)) else "—"
    def sp_str(v): return f"{v:.2f}x" if isinstance(v, (int, float)) else "—"

    ae_cpu_total = (d["ae_cpu"].get("train_s",0) or 0) + (d["ae_cpu"].get("infer_s",0) or 0)
    ae_gpu_total = (d["ae_gpu"].get("train_s",0) or 0) + (d["ae_gpu"].get("infer_s",0) or 0)
    km_cpu_total = (d["km_cpu"].get("train_s",0) or 0) + (d["km_cpu"].get("infer_s",0) or 0)
    km_gpu_total = (d["km_gpu"].get("train_s",0) or 0) + (d["km_gpu"].get("infer_s",0) or 0)
    ae_sp = ae_cpu_total / ae_gpu_total if ae_gpu_total > 0 else None
    km_sp = km_cpu_total / km_gpu_total if km_gpu_total > 0 else None

    table_rows = [
        ["Autoencoder", "CPU",
         fv(d["ae_cpu"].get("train_s")), fv(d["ae_cpu"].get("infer_s")),
         fv(ae_cpu_total or None), "1.00x", "✓ ok"],
        ["Autoencoder", "GPU",
         fv(d["ae_gpu"].get("train_s")), fv(d["ae_gpu"].get("infer_s")),
         fv(ae_gpu_total or None), sp_str(ae_sp),
         "✓ ok" if d["ae_gpu"].get("train_s") else "⊘ skipped"],
        ["K-Means", "CPU",
         fv(d["km_cpu"].get("train_s")), fv(d["km_cpu"].get("infer_s")),
         fv(km_cpu_total or None), "1.00x", "✓ ok"],
        ["K-Means", "GPU",
         fv(d["km_gpu"].get("train_s")), fv(d["km_gpu"].get("infer_s")),
         fv(km_gpu_total or None), sp_str(km_sp),
         "✓ ok" if d["km_gpu"].get("train_s") else "⊘ skipped"],
    ]
    col_hdrs = ["Experimento", "Hardware", "Treino", "Inferência", "Total", "Speedup", "Status"]
    col_x    = [0.01, 0.16, 0.27, 0.38, 0.50, 0.61, 0.72]

    for cx, h in zip(col_x, col_hdrs):
        ax5.text(cx, 0.88, h, transform=ax5.transAxes,
                 color=BLUE, fontsize=9.5, fontweight="bold", va="top")
    line = plt.Line2D([0.01,0.99],[0.78,0.78], transform=ax5.transAxes,
                      color=GRID, linewidth=1)
    ax5.add_line(line)

    row_ys = [0.62, 0.44, 0.26, 0.08]
    for ri, (row, ry) in enumerate(zip(table_rows, row_ys)):
        bg = "#243044" if ri % 2 else "#1e293b"
        rect = mpatches.FancyBboxPatch((0.005, ry-0.09), 0.99, 0.18,
                                        boxstyle="round,pad=0.01",
                                        facecolor=bg, edgecolor=GRID,
                                        linewidth=0.5,
                                        transform=ax5.transAxes, clip_on=False)
        ax5.add_patch(rect)
        for ci, (val, cx) in enumerate(zip(row, col_x)):
            color = TEXT
            if ci == 1: color = GREEN if val == "CPU" else PURPLE
            elif ci == 5: color = CYAN if val not in ("—", "1.00x") else MUTED
            elif ci == 6: color = GREEN if "ok" in val else RED
            ax5.text(cx, ry, val, transform=ax5.transAxes,
                     color=color, fontsize=9, va="center")

    # ── 6. Timeline ───────────────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[4, :])
    ax6.set_facecolor(PANEL)
    for sp in ax6.spines.values(): sp.set_edgecolor(GRID)
    ax6.set_title("Timeline de Execução do Pipeline (minutos)", color=TEXT,
                  fontsize=11, fontweight="bold", pad=10)

    if d["shards"]:
        timeline: list[tuple] = []
        cur = 0.0
        colors_t = SHARD_COLORS[:len(d["shards"])]
        for s, c in zip(d["shards"], colors_t):
            dur = s.get("time_min", 0)
            name = s["name"].split("__")[1][:10] if "__" in s.get("name","") else s.get("name","")[:10]
            timeline.append((name, cur, dur, c))
            cur += dur
        # Autoencoder + KMeans (segundos → minutos)
        ae_bench = (ae_cpu_total + ae_gpu_total) / 60
        km_bench = (km_cpu_total + km_gpu_total) / 60
        if ae_bench > 0:
            timeline.append(("AE bench", cur, max(ae_bench, 0.05), CYAN))
            cur += max(ae_bench, 0.05)
        if km_bench > 0:
            timeline.append(("KM bench", cur, max(km_bench, 0.05), RED))

        total_t = sum(t[2] for t in timeline)
        for label, start, dur, col in timeline:
            ax6.barh(0, dur, left=start, color=col, edgecolor=DARK,
                     linewidth=0.8, height=0.5)
            if dur > total_t * 0.04:
                ax6.text(start+dur/2, 0, f"{label}\n{dur:.1f}m",
                         ha="center", va="center", color=DARK,
                         fontsize=8.5, fontweight="bold")
            else:
                ax6.text(start+dur+total_t*0.005, 0.32, label,
                         ha="left", color=col, fontsize=7.5)
        ax6.set_xlim(0, total_t * 1.06)
        ax6.text(0.99, 0.88, f"Total: {total_t:.1f} min",
                 transform=ax6.transAxes, ha="right",
                 color=AMBER, fontsize=10, fontweight="bold")

    ax6.set_ylim(-0.5, 0.8); ax6.set_yticks([])
    ax6.set_xlabel("Tempo acumulado (minutos)", color=MUTED, fontsize=9)
    ax6.tick_params(axis="x", colors=MUTED)
    ax6.grid(axis="x", color=GRID, linestyle="--", linewidth=0.6, alpha=0.5)

    plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=DARK)
    plt.close(fig)
    logger.info(f"Relatório visual salvo em : {out_path}")


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

    # ── Relatório visual ──────────────────────────────────────────────────────
    plot_report(log_path, outdir / "pipeline_report.png", logger)

    # ── Encerramento ──────────────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - run_start
    sep(logger, "PIPELINE CONCLUÍDO")
    logger.info(f"Tempo total  : {total_elapsed:.2f}s  ({total_elapsed/60:.1f} min)")
    logger.info(f"Log salvo em : {log_path}")
    sep(logger)


if __name__ == "__main__":
    main()
