"""
anomaly_compare.py
══════════════════════════════════════════════════════════════════════════════
Comparativo unificado de detecção de anomalias em tráfego de rede.
Executa Autoencoder e K-Means em CPU e GPU, mede qualidade e desempenho,
e gera relatorio completo (TXT detalhado + PNG resumido).

Métricas de qualidade (não supervisionadas):
  • Anomaly Rate          — % de fluxos classificados como anômalos
  • Score Distribution    — média, desvio, p95, p99 dos scores de anomalia
  • Silhouette Score      — coesão e separação dos clusters (K-Means)
  • Reconstruction Error  — distribuição do erro MSE (Autoencoder)
  • Threshold Analysis    — impacto do percentil de corte na taxa de anomalia

Métricas de desempenho:
  • train_s, infer_s, total_s, speedup

Uso:
    python anomaly_compare.py
    python anomaly_compare.py --skip-extraction --gpu
    python anomaly_compare.py --skip-extraction --gpu --sample-size 20000 --test-size 0.30
    python anomaly_compare.py --skip-extraction --gpu --ae-epochs 30 --kmeans-clusters 10

Dependências:
    pip install torch scikit-learn pandas numpy matplotlib
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    MATPLOTLIB_OK = True
except ImportError:
    MATPLOTLIB_OK = False

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

# ── Paths padrão ──────────────────────────────────────────────────────────────
BASE   = Path(__file__).resolve().parent
PCAPS  = BASE / "data" / "pcaps"
OUT    = BASE / "data" / "results"
CACHE  = OUT / "features_cache.pkl"

PCAP_FILES = sorted(PCAPS.glob("*.pcapng"))


# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════

def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("anomaly_compare")
    log.setLevel(logging.DEBUG)
    if log.handlers:
        log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(ch)
    return log


def section(log: logging.Logger, title: str) -> None:
    log.info("=" * 72)
    log.info(f"  {title}")
    log.info("=" * 72)


# ══════════════════════════════════════════════════════════════════════════════
# EXTRAÇÃO
# ══════════════════════════════════════════════════════════════════════════════

def run_extraction(cache_file: Path, log: logging.Logger) -> None:
    section(log, "ETAPA 1 — EXTRAÇÃO DE FEATURES")
    sys.path.insert(0, str(BASE))
    from feature_extractor import extract_flows

    dfs = []
    shard_stats = []
    t_total = time.perf_counter()

    for i, pcap in enumerate(PCAP_FILES, 1):
        log.info(f"[Shard {i}/{len(PCAP_FILES)}] {pcap.name}")
        size_mb = pcap.stat().st_size / 1024**2
        log.info(f"  Tamanho  : {size_mb:,.1f} MB")
        t0 = time.perf_counter()
        df = extract_flows(str(pcap))
        elapsed = time.perf_counter() - t0
        log.info(f"  Fluxos   : {len(df):,}")
        log.info(f"  Tempo    : {elapsed:.1f}s ({elapsed/60:.1f} min)")
        dfs.append(df)
        shard_stats.append({"name": pcap.name, "flows": len(df),
                             "size_mb": round(size_mb, 1), "time_s": round(elapsed, 2)})

    combined = pd.concat(dfs, ignore_index=True)
    total_elapsed = time.perf_counter() - t_total
    log.info(f"Total fluxos : {len(combined):,} em {total_elapsed/60:.1f} min")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(combined.values.astype(np.float32))
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "wb") as f:
        pickle.dump({"X_scaled": X_scaled, "X": combined.values,
                     "columns": list(combined.columns),
                     "shard_stats": shard_stats}, f)
    log.info(f"Cache salvo  : {cache_file}")


# ══════════════════════════════════════════════════════════════════════════════
# MÉTRICAS DE QUALIDADE
# ══════════════════════════════════════════════════════════════════════════════

def score_distribution(scores: np.ndarray, label: str, log: logging.Logger) -> dict:
    """Calcula e loga a distribuição completa dos scores de anomalia."""
    pcts = [50, 75, 90, 95, 99]
    percentiles = {f"p{p}": float(np.percentile(scores, p)) for p in pcts}

    log.info(f"  [{label}] Distribuição de scores:")
    log.info(f"    Média     : {scores.mean():.6f}")
    log.info(f"    Desvio    : {scores.std():.6f}")
    log.info(f"    Mín/Máx   : {scores.min():.6f} / {scores.max():.6f}")
    for p, v in percentiles.items():
        log.info(f"    {p:<8}  : {v:.6f}")
    return {"mean": float(scores.mean()), "std": float(scores.std()),
            "min": float(scores.min()), "max": float(scores.max()),
            **percentiles}


def anomaly_rate_analysis(scores: np.ndarray, label: str,
                           log: logging.Logger) -> dict:
    """Analisa como a taxa de anomalia varia com o threshold."""
    thresholds = [90, 92, 95, 97, 99]
    analysis = {}
    log.info(f"  [{label}] Análise de threshold:")
    for pct in thresholds:
        thresh = float(np.percentile(scores, pct))
        n_anom = int((scores > thresh).sum())
        rate   = n_anom / len(scores) * 100
        log.info(f"    p{pct} (thresh={thresh:.4f}) → {n_anom:,} anomalias ({rate:.1f}%)")
        analysis[f"p{pct}"] = {"threshold": thresh, "n_anomalies": n_anom,
                                "anomaly_rate_pct": round(rate, 2)}
    return analysis


def kmeans_quality(X: np.ndarray, labels: np.ndarray,
                   log: logging.Logger) -> dict:
    """Calcula métricas de qualidade de clustering."""
    n_unique = len(np.unique(labels))

    # Silhouette (amostrado se muito grande para não travar)
    n_sil = min(len(X), 5000)
    if n_unique < 2:
        sil = None
        log.warning("  Silhouette: skipped (apenas 1 cluster)")
    else:
        idx = np.random.default_rng(42).choice(len(X), n_sil, replace=False)
        sil = float(silhouette_score(X[idx], labels[idx], random_state=42))
        log.info(f"  Silhouette Score  : {sil:.4f}  (−1 ruim → +1 ótimo, >0.2 aceitável)")

    # Distribuição por cluster
    unique, counts = np.unique(labels, return_counts=True)
    cluster_dist = {}
    log.info("  Distribuição de clusters:")
    for c, n in zip(unique, counts):
        pct = n / len(labels) * 100
        flag = " ⚠ SUSPEITO" if pct < 1.0 else ""
        log.info(f"    Cluster {c:>2} : {n:>5,} amostras ({pct:5.1f}%){flag}")
        cluster_dist[int(c)] = {"count": int(n), "pct": round(pct, 2)}

    # Clusters suspeitos (< 1% das amostras)
    suspicious = [c for c, v in cluster_dist.items() if v["pct"] < 1.0]
    n_suspicious = sum(cluster_dist[c]["count"] for c in suspicious)
    suspicious_rate = n_suspicious / len(labels) * 100
    log.info(f"  Clusters suspeitos : {suspicious} → {n_suspicious:,} fluxos ({suspicious_rate:.2f}%)")

    return {"silhouette": sil, "cluster_distribution": cluster_dist,
            "suspicious_clusters": suspicious,
            "n_suspicious_flows": n_suspicious,
            "suspicious_rate_pct": round(suspicious_rate, 2)}


# ══════════════════════════════════════════════════════════════════════════════
# AUTOENCODER
# ══════════════════════════════════════════════════════════════════════════════

if TORCH_OK:
    class Autoencoder(nn.Module):
        def __init__(self, n_features: int, latent_dim: int):
            super().__init__()
            h = max(32, n_features * 2)
            self.encoder = nn.Sequential(
                nn.Linear(n_features, h), nn.ReLU(),
                nn.Linear(h, latent_dim),  nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, h), nn.ReLU(),
                nn.Linear(h, n_features),
            )
        def forward(self, x):
            return self.decoder(self.encoder(x))


def _sync(device: str) -> None:
    if TORCH_OK and device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def run_autoencoder(X_train: np.ndarray, X_test: np.ndarray, device: str,
                    epochs: int, batch_size: int, latent_dim: int, lr: float,
                    log: logging.Logger) -> dict:
    if not TORCH_OK:
        return {"status": "error", "notes": "PyTorch não disponível"}

    log.info(f"  Device      : {device.upper()}")
    log.info(f"  Épocas      : {epochs}")
    log.info(f"  Batch size  : {batch_size}")
    log.info(f"  Latent dim  : {latent_dim}")
    log.info(f"  LR          : {lr}")
    log.info(f"  Treino      : {len(X_train):,} amostras")
    log.info(f"  Teste       : {len(X_test):,} amostras")

    model   = Autoencoder(X_train.shape[1], latent_dim).to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    tensor_train = torch.from_numpy(X_train.astype(np.float32))
    tensor_test  = torch.from_numpy(X_test.astype(np.float32))
    loader_tr = DataLoader(TensorDataset(tensor_train), batch_size=batch_size,
                           shuffle=True,  pin_memory=(device=="cuda"))
    loader_te = DataLoader(TensorDataset(tensor_test),  batch_size=batch_size,
                           shuffle=False, pin_memory=(device=="cuda"))

    # ── Treino (apenas X_train) ───────────────────────────────────────────────
    _sync(device)
    t_train = time.perf_counter()
    epoch_losses = []
    model.train()
    for epoch in range(1, epochs + 1):
        ep_loss = 0.0
        for (batch,) in loader_tr:
            batch = batch.to(device, non_blocking=(device=="cuda"))
            recon = model(batch)
            loss  = loss_fn(recon, batch)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item() * len(batch)
        avg = ep_loss / len(X_train)
        epoch_losses.append(avg)
        log.info(f"  [AE-{device.upper()}] Época {epoch:>3}/{epochs} | loss={avg:.6f}")
    _sync(device)
    train_s = time.perf_counter() - t_train

    # ── Threshold definido nos erros de TREINO (sem leakage) ─────────────────
    model.eval()
    train_errors_list: list[np.ndarray] = []
    with torch.no_grad():
        for (batch,) in loader_tr:
            batch = batch.to(device, non_blocking=(device=="cuda"))
            recon = model(batch)
            train_errors_list.append(((recon - batch)**2).mean(dim=1).cpu().numpy())
    train_errors = np.concatenate(train_errors_list)
    threshold = float(np.percentile(train_errors, 95))
    log.info(f"  Threshold p95 (treino) : {threshold:.6f}")

    # ── Inferência em X_test (dados nunca vistos) ─────────────────────────────
    _sync(device)
    t_infer = time.perf_counter()
    test_errors: list[np.ndarray] = []
    with torch.no_grad():
        for (batch,) in loader_te:
            batch = batch.to(device, non_blocking=(device=="cuda"))
            recon = model(batch)
            test_errors.append(((recon - batch)**2).mean(dim=1).cpu().numpy())
    _sync(device)
    infer_s = time.perf_counter() - t_infer

    scores = np.concatenate(test_errors)
    log.info(f"  Treino      : {train_s:.3f}s")
    log.info(f"  Inferência  : {infer_s:.3f}s")

    # ── Qualidade no conjunto de TESTE ────────────────────────────────────────
    log.info("")
    log.info("  [Scores no conjunto de TESTE — dados não vistos no treino]")
    score_dist  = score_distribution(scores, f"AE-{device.upper()}", log)
    thresh_anal = anomaly_rate_analysis(scores, f"AE-{device.upper()}", log)

    labels    = (scores > threshold).astype(int)
    n_anom    = int(labels.sum())
    anom_rate = n_anom / len(scores) * 100
    log.info(f"  Anomalias detectadas (p95 do treino) : {n_anom:,} ({anom_rate:.2f}%)")

    return {
        "status":           "ok",
        "device":           device,
        "train_s":          round(train_s, 4),
        "infer_s":          round(infer_s, 4),
        "total_s":          round(train_s + infer_s, 4),
        "epoch_losses":     [round(l, 6) for l in epoch_losses],
        "final_loss":       round(epoch_losses[-1], 6),
        "train_final_loss": round(float(train_errors.mean()), 6),
        "scores":           scores,
        "threshold_p95":    threshold,
        "n_train":          len(X_train),
        "n_test":           len(X_test),
        "n_anomalies":      n_anom,
        "anomaly_rate":     round(anom_rate, 2),
        "score_dist":       score_dist,
        "threshold_analysis": thresh_anal,
        "notes":            f"epochs={epochs} latent={latent_dim} batch={batch_size} train={len(X_train)} test={len(X_test)}",
    }


# ══════════════════════════════════════════════════════════════════════════════
# K-MEANS CPU
# ══════════════════════════════════════════════════════════════════════════════

def run_kmeans_cpu(X_train: np.ndarray, X_test: np.ndarray, k: int,
                   max_iter: int, log: logging.Logger) -> dict:
    log.info(f"  Device    : CPU (scikit-learn)")
    log.info(f"  k         : {k}")
    log.info(f"  max_iter  : {max_iter}")
    log.info(f"  Treino    : {len(X_train):,} amostras")
    log.info(f"  Teste     : {len(X_test):,} amostras")

    model = KMeans(n_clusters=k, max_iter=max_iter, n_init=10, random_state=42)

    # ── Treino (apenas X_train) ───────────────────────────────────────────────
    t0 = time.perf_counter()
    model.fit(X_train)
    train_s = time.perf_counter() - t0

    # ── Threshold no X_train (sem leakage) ───────────────────────────────────
    centroids    = model.cluster_centers_
    train_labels = model.predict(X_train)
    train_dists  = np.linalg.norm(X_train - centroids[train_labels], axis=1)
    threshold    = float(np.percentile(train_dists, 95))
    log.info(f"  Iterações : {model.n_iter_} / {max_iter}")
    log.info(f"  Inércia   : {model.inertia_:.4f}")
    log.info(f"  Threshold p95 (treino) : {threshold:.6f}")
    log.info(f"  Treino    : {train_s:.3f}s")

    # ── Inferência em X_test (dados nunca vistos) ─────────────────────────────
    t1 = time.perf_counter()
    labels = model.predict(X_test)
    infer_s = time.perf_counter() - t1
    dists   = np.linalg.norm(X_test - centroids[labels], axis=1)
    log.info(f"  Inferência: {infer_s:.4f}s")

    # ── Qualidade no conjunto de TESTE ────────────────────────────────────────
    log.info("")
    log.info("  [Scores no conjunto de TESTE — dados não vistos no treino]")
    quality     = kmeans_quality(X_test, labels, log)
    score_dist  = score_distribution(dists, "KM-CPU", log)
    thresh_anal = anomaly_rate_analysis(dists, "KM-CPU", log)

    n_anom    = int((dists > threshold).sum())
    anom_rate = n_anom / len(dists) * 100
    log.info(f"  Anomalias detectadas (p95 do treino) : {n_anom:,} ({anom_rate:.2f}%)")

    return {
        "status":        "ok",
        "device":        "cpu",
        "train_s":       round(train_s, 4),
        "infer_s":       round(infer_s, 4),
        "total_s":       round(train_s + infer_s, 4),
        "inertia":       round(float(model.inertia_), 4),
        "n_iter":        int(model.n_iter_),
        "labels":        labels,
        "scores":        dists,
        "threshold_p95": threshold,
        "n_train":       len(X_train),
        "n_test":        len(X_test),
        "n_anomalies":   n_anom,
        "anomaly_rate":  round(anom_rate, 2),
        "score_dist":    score_dist,
        "threshold_analysis": thresh_anal,
        "quality":       quality,
        "notes":         f"k={k} n_iter={model.n_iter_} inertia={model.inertia_:.2f} train={len(X_train)} test={len(X_test)}",
    }


# ══════════════════════════════════════════════════════════════════════════════
# K-MEANS GPU (PyTorch puro)
# ══════════════════════════════════════════════════════════════════════════════

def run_kmeans_gpu(X_train: np.ndarray, X_test: np.ndarray, k: int,
                   max_iter: int, log: logging.Logger) -> dict:
    if not TORCH_OK or not torch.cuda.is_available():
        msg = "CUDA não disponível"
        log.warning(f"  {msg}")
        return {"status": "unavailable", "notes": msg}

    device   = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    log.info(f"  Device    : GPU — {gpu_name}")
    log.info(f"  k         : {k}")
    log.info(f"  max_iter  : {max_iter}")
    log.info(f"  Treino    : {len(X_train):,} amostras")
    log.info(f"  Teste     : {len(X_test):,} amostras")

    X_tr = torch.from_numpy(X_train.astype(np.float32)).to(device)
    X_te = torch.from_numpy(X_test.astype(np.float32)).to(device)
    rng  = torch.Generator(device=device); rng.manual_seed(42)

    # ── kmeans++ na GPU (treino apenas em X_train) ────────────────────────────
    torch.cuda.synchronize()
    t_train = time.perf_counter()

    idx0      = torch.randint(0, X_tr.shape[0], (1,), generator=rng, device=device).item()
    centroids = X_tr[idx0].unsqueeze(0)
    for _ in range(1, k):
        d    = torch.cdist(X_tr, centroids).min(dim=1).values
        prob = d / d.sum()
        idx  = torch.multinomial(prob, 1, generator=rng).item()
        centroids = torch.cat([centroids, X_tr[idx].unsqueeze(0)], dim=0)

    labels_t        = torch.zeros(X_tr.shape[0], dtype=torch.long, device=device)
    inertia_history = []

    for iteration in range(max_iter):
        dists_all  = torch.cdist(X_tr, centroids)
        new_labels = dists_all.argmin(dim=1)
        inertia    = dists_all.min(dim=1).values.pow(2).sum().item()
        inertia_history.append(inertia)
        if iteration > 0 and torch.equal(new_labels, labels_t):
            log.info(f"  Convergiu em {iteration + 1} iterações")
            labels_t = new_labels
            break
        labels_t = new_labels
        for ki in range(k):
            mask = labels_t == ki
            if mask.any():
                centroids[ki] = X_tr[mask].mean(dim=0)

    torch.cuda.synchronize()
    train_s = time.perf_counter() - t_train

    # ── Threshold calculado no X_train (sem leakage) ──────────────────────────
    train_dists = torch.cdist(X_tr, centroids).min(dim=1).values.cpu().numpy()
    threshold   = float(np.percentile(train_dists, 95))
    log.info(f"  Iterações : {len(inertia_history)} / {max_iter}")
    log.info(f"  Inércia   : {inertia_history[-1]:.4f}")
    log.info(f"  Threshold p95 (treino) : {threshold:.6f}")
    log.info(f"  Treino    : {train_s:.3f}s")

    # ── Inferência em X_test (dados nunca vistos) ─────────────────────────────
    torch.cuda.synchronize()
    t_infer      = time.perf_counter()
    dists_test   = torch.cdist(X_te, centroids)
    labels_final = dists_test.argmin(dim=1)
    dists_min    = dists_test.min(dim=1).values
    torch.cuda.synchronize()
    infer_s   = time.perf_counter() - t_infer
    labels_np = labels_final.cpu().numpy()
    scores_np = dists_min.cpu().numpy()

    log.info(f"  Inferência: {infer_s:.4f}s")

    # ── Qualidade no conjunto de TESTE ────────────────────────────────────────
    log.info("")
    log.info("  [Scores no conjunto de TESTE — dados não vistos no treino]")
    quality     = kmeans_quality(X_test, labels_np, log)
    score_dist  = score_distribution(scores_np, "KM-GPU", log)
    thresh_anal = anomaly_rate_analysis(scores_np, "KM-GPU", log)

    n_anom    = int((scores_np > threshold).sum())
    anom_rate = n_anom / len(scores_np) * 100
    inertia_final = float(inertia_history[-1])
    log.info(f"  Anomalias detectadas (p95 do treino) : {n_anom:,} ({anom_rate:.2f}%)")

    return {
        "status":        "ok",
        "device":        "cuda",
        "train_s":       round(train_s, 4),
        "infer_s":       round(infer_s, 4),
        "total_s":       round(train_s + infer_s, 4),
        "inertia":       round(inertia_final, 4),
        "n_iter":        len(inertia_history),
        "labels":        labels_np,
        "scores":        scores_np,
        "threshold_p95": threshold,
        "n_train":       len(X_train),
        "n_test":        len(X_test),
        "n_anomalies":   n_anom,
        "anomaly_rate":  round(anom_rate, 2),
        "score_dist":    score_dist,
        "threshold_analysis": thresh_anal,
        "quality":       quality,
        "notes":         f"k={k} n_iter={len(inertia_history)} inertia={inertia_final:.2f} train={len(X_train)} test={len(X_test)} (PyTorch GPU)",
    }


# ══════════════════════════════════════════════════════════════════════════════
# RELATÓRIO VISUAL
# ══════════════════════════════════════════════════════════════════════════════

def _fmt(v, digits: int = 4) -> str:
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    if isinstance(v, int):
        return f"{v:,}"
    if v is None:
        return "N/A"
    return str(v)


def write_text_report(results: dict, txt_path: Path, args: argparse.Namespace,
                      total_s: float, log: logging.Logger) -> None:
    """Append a human-readable final report to anomaly_compare.txt."""
    lines: list[str] = []
    lines.append("")
    lines.append("=" * 78)
    lines.append("RELATORIO DETALHADO DA EXECUCAO")
    lines.append("=" * 78)
    lines.append("")
    lines.append("Objetivo")
    lines.append("  Comparar Autoencoder e K-Means para deteccao nao supervisionada")
    lines.append("  de anomalias em fluxos de rede, medindo qualidade e desempenho")
    lines.append("  em CPU e, quando habilitado, GPU.")
    lines.append("")
    lines.append("Parametros essenciais")
    lines.append(f"  Diretorio de saida : {args.outdir}")
    lines.append(f"  Cache de features  : {args.cache_file}")
    lines.append(f"  Pular extracao     : {args.skip_extraction}")
    lines.append(f"  GPU habilitada     : {args.gpu}")
    lines.append(f"  Amostras usadas    : {args.sample_size if args.sample_size > 0 else 'todas'}")
    lines.append(f"  Fracao de teste    : {args.test_size:.2f}")
    lines.append(f"  Epocas AE          : {args.ae_epochs}")
    lines.append(f"  K-Means clusters   : {args.kmeans_clusters}")
    lines.append("")
    lines.append("Etapas executadas")
    lines.append("  1. Extracao/cache: quando --skip-extraction nao e usado, os PCAPs")
    lines.append("     em data/pcaps sao transformados em features numericas e salvos")
    lines.append("     em features_cache.pkl. Com --skip-extraction, o cache existente")
    lines.append("     e carregado diretamente.")
    lines.append("  2. Normalizacao e amostragem: as features escaladas sao carregadas")
    lines.append("     do cache; se --sample-size for maior que zero, uma amostra")
    lines.append("     reprodutivel com seed 42 e selecionada.")
    lines.append("  3. Divisao treino/teste: os dados sao separados com random_state=42.")
    lines.append("     O treino aprende o comportamento normal; o teste mede a deteccao")
    lines.append("     em dados nao vistos.")
    lines.append("  4. Autoencoder: treina uma rede neural para reconstruir os fluxos.")
    lines.append("     Erros de reconstrucao maiores indicam maior suspeita de anomalia.")
    lines.append("  5. K-Means: agrupa os fluxos e usa a distancia ao centroide mais")
    lines.append("     proximo como score de anomalia.")
    lines.append("  6. Threshold: o percentil 95 e calculado somente no treino para")
    lines.append("     reduzir data leakage. A contagem final de anomalias usa o teste.")
    lines.append("  7. Relatorio visual: anomaly_report.png mostra apenas parametros")
    lines.append("     essenciais, tempo total, taxa de anomalia e quantidade detectada.")
    lines.append("")
    lines.append("Resumo por metodo")
    lines.append("  Metodo     HW   Status      Treino(s) Infer(s) Total(s) Anomalias Anom%  Threshold")
    lines.append("  " + "-" * 86)

    for name in ["AE-CPU", "AE-GPU", "KM-CPU", "KM-GPU"]:
        res = results.get(name, {})
        status = res.get("status", "nao executado")
        hw = "GPU" if res.get("device") in ("cuda", "gpu") else "CPU"
        if status != "ok":
            lines.append(f"  {name:<9} {hw:<4} {status:<10} {res.get('notes', '')}")
            continue
        lines.append(
            f"  {name:<9} {hw:<4} {status:<10} "
            f"{_fmt(res.get('train_s'), 3):>8} "
            f"{_fmt(res.get('infer_s'), 3):>8} "
            f"{_fmt(res.get('total_s'), 3):>8} "
            f"{_fmt(res.get('n_anomalies')):>9} "
            f"{_fmt(res.get('anomaly_rate'), 2):>6}% "
            f"{_fmt(res.get('threshold_p95'), 6):>10}"
        )

    lines.append("")
    lines.append("Detalhes por metodo")
    for name in ["AE-CPU", "AE-GPU", "KM-CPU", "KM-GPU"]:
        res = results.get(name, {})
        lines.append("")
        lines.append(f"[{name}]")
        lines.append(f"  Status: {res.get('status', 'nao executado')}")
        if res.get("notes"):
            lines.append(f"  Observacoes: {res.get('notes')}")
        if res.get("status") != "ok":
            continue
        lines.append(f"  Treino/Teste: {res.get('n_train', 'N/A')} / {res.get('n_test', 'N/A')}")
        lines.append(f"  Tempo de treino: {_fmt(res.get('train_s'), 3)}s")
        lines.append(f"  Tempo de inferencia: {_fmt(res.get('infer_s'), 3)}s")
        lines.append(f"  Tempo total: {_fmt(res.get('total_s'), 3)}s")
        lines.append(f"  Threshold p95 do treino: {_fmt(res.get('threshold_p95'), 6)}")
        lines.append(f"  Anomalias no teste: {_fmt(res.get('n_anomalies'))} ({_fmt(res.get('anomaly_rate'), 2)}%)")
        if "final_loss" in res:
            lines.append(f"  Loss final do Autoencoder: {_fmt(res.get('final_loss'), 6)}")
        if "train_final_loss" in res:
            lines.append(f"  Erro medio no treino: {_fmt(res.get('train_final_loss'), 6)}")
        if "inertia" in res:
            lines.append(f"  Inercia K-Means: {_fmt(res.get('inertia'), 4)}")
        quality = res.get("quality", {})
        if quality:
            lines.append(f"  Silhouette: {_fmt(quality.get('silhouette'), 4)}")
            lines.append(f"  Clusters suspeitos: {quality.get('suspicious_clusters', [])}")
            lines.append(f"  Fluxos em clusters suspeitos: {_fmt(quality.get('n_suspicious_flows'))}")
        score_dist = res.get("score_dist", {})
        if score_dist:
            lines.append("  Distribuicao dos scores:")
            for key in ["mean", "std", "min", "max", "p50", "p75", "p90", "p95", "p99"]:
                if key in score_dist:
                    lines.append(f"    {key:<4}: {_fmt(score_dist.get(key), 6)}")
        threshold_analysis = res.get("threshold_analysis", {})
        if threshold_analysis:
            lines.append("  Analise por percentil:")
            for pct, data in threshold_analysis.items():
                lines.append(
                    f"    {pct}: threshold={_fmt(data.get('threshold'), 6)}, "
                    f"anomalias={_fmt(data.get('n_anomalies'))}, "
                    f"taxa={_fmt(data.get('anomaly_rate_pct'), 2)}%"
                )

    lines.append("")
    lines.append(f"Tempo total da execucao: {total_s:.2f}s ({total_s/60:.1f} min)")
    lines.append("Arquivos gerados")
    lines.append(f"  TXT detalhado : {txt_path}")
    lines.append(f"  PNG visual    : {Path(args.outdir) / 'anomaly_report.png'}")
    lines.append("")

    with open(txt_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log.info(f"Relatorio detalhado TXT salvo em : {txt_path}")


def plot_report(results: dict, out_path: Path, args: argparse.Namespace,
                log: logging.Logger) -> None:
    if not MATPLOTLIB_OK:
        log.warning("matplotlib indisponivel - relatorio visual nao gerado")
        return

    BG = "#FFFFFF"
    TEXT = "#111827"
    MUTED = "#4B5563"
    GRID = "#D1D5DB"
    BLUE = "#2563EB"
    GREEN = "#059669"
    ORANGE = "#D97706"
    RED = "#DC2626"
    COLORS = {"AE-CPU": BLUE, "AE-GPU": GREEN, "KM-CPU": ORANGE, "KM-GPU": RED}

    fig = plt.figure(figsize=(14, 9), facecolor=BG)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.28,
                           top=0.84, bottom=0.08, left=0.08, right=0.96)
    fig.suptitle("Deteccao de Anomalias - Resumo Essencial",
                 fontsize=18, fontweight="bold", color=TEXT, y=0.96)
    fig.text(
        0.5, 0.91,
        f"amostras={args.sample_size if args.sample_size > 0 else 'todas'} | "
        f"teste={args.test_size:.0%} | AE epocas={args.ae_epochs} | "
        f"K-Means k={args.kmeans_clusters} | GPU={'sim' if args.gpu else 'nao'}",
        ha="center", fontsize=11, color=MUTED,
    )

    def style(ax, title: str) -> None:
        ax.set_facecolor(BG)
        ax.set_title(title, fontsize=12, fontweight="bold", color=TEXT, pad=10)
        ax.grid(axis="y", color=GRID, linestyle="--", linewidth=0.8, alpha=0.8)
        ax.tick_params(colors=TEXT, labelsize=10)
        for sp in ax.spines.values():
            sp.set_color(GRID)

    ok_items = [(name, results.get(name, {})) for name in ["AE-CPU", "AE-GPU", "KM-CPU", "KM-GPU"]
                if results.get(name, {}).get("status") == "ok"]
    names = [name for name, _ in ok_items]

    ax1 = fig.add_subplot(gs[0, 0])
    style(ax1, "Tempo total por metodo (s)")
    totals = [res.get("total_s", 0) for _, res in ok_items]
    if totals:
        bars = ax1.bar(names, totals, color=[COLORS[n] for n in names], width=0.55)
        ax1.set_ylabel("segundos", color=MUTED)
        for bar, value in zip(bars, totals):
            ax1.text(bar.get_x() + bar.get_width()/2, value, f"{value:.2f}s",
                     ha="center", va="bottom", fontsize=10, color=TEXT)
    else:
        ax1.text(0.5, 0.5, "Nenhum metodo executado", ha="center", va="center",
                 transform=ax1.transAxes, color=MUTED)

    ax2 = fig.add_subplot(gs[0, 1])
    style(ax2, "Taxa de anomalia no teste (%)")
    rates = [res.get("anomaly_rate", 0) for _, res in ok_items]
    if rates:
        bars = ax2.bar(names, rates, color=[COLORS[n] for n in names], width=0.55)
        ax2.axhline(5.0, color=GRID, linewidth=1.2, linestyle="--")
        ax2.set_ylabel("%", color=MUTED)
        for bar, value in zip(bars, rates):
            ax2.text(bar.get_x() + bar.get_width()/2, value, f"{value:.1f}%",
                     ha="center", va="bottom", fontsize=10, color=TEXT)

    ax3 = fig.add_subplot(gs[1, 0])
    style(ax3, "Anomalias detectadas")
    counts = [res.get("n_anomalies", 0) for _, res in ok_items]
    if counts:
        bars = ax3.bar(names, counts, color=[COLORS[n] for n in names], width=0.55)
        ax3.set_ylabel("fluxos", color=MUTED)
        for bar, value in zip(bars, counts):
            ax3.text(bar.get_x() + bar.get_width()/2, value, f"{value:,}",
                     ha="center", va="bottom", fontsize=10, color=TEXT)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    ax4.set_title("Leitura rapida", fontsize=12, fontweight="bold", color=TEXT, pad=10)
    summary_lines = [
        "Threshold: percentil 95 calculado no treino",
        "Avaliacao: anomalias contadas somente no teste",
        "TXT: contem etapas, parametros e detalhes por metodo",
        "PNG: mostra apenas os indicadores principais",
    ]
    for idx, line in enumerate(summary_lines):
        ax4.text(0.02, 0.82 - idx * 0.16, line, transform=ax4.transAxes,
                 fontsize=11, color=TEXT, va="center")

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    log.info(f"Relatorio visual salvo em : {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comparativo unificado Autoencoder + K-Means — CPU vs GPU")
    parser.add_argument("--outdir",           default=str(OUT))
    parser.add_argument("--cache-file",       default=str(CACHE))
    parser.add_argument("--skip-extraction",  action="store_true")
    parser.add_argument("--gpu",              action="store_true",
                        help="Habilita GPU para Autoencoder e K-Means")
    parser.add_argument("--sample-size",      type=int,   default=5000)
    parser.add_argument("--ae-epochs",        type=int,   default=12)
    parser.add_argument("--ae-batch-size",    type=int,   default=256)
    parser.add_argument("--ae-latent-dim",    type=int,   default=8)
    parser.add_argument("--ae-lr",            type=float, default=1e-3)
    parser.add_argument("--kmeans-clusters",  type=int,   default=8)
    parser.add_argument("--kmeans-max-iter",  type=int,   default=300)
    parser.add_argument("--test-size",        type=float, default=0.30,
                        help="Fração dos dados reservada para teste (padrão: 0.30)")
    args = parser.parse_args()

    outdir     = Path(args.outdir)
    cache_file = Path(args.cache_file)
    outdir.mkdir(parents=True, exist_ok=True)
    for old_name in ("anomaly_compare.log", "anomaly_compare.json", "anomaly_compare.csv"):
        old_path = outdir / old_name
        if old_path.exists():
            old_path.unlink()
    log_path   = outdir / "anomaly_compare.txt"
    log        = setup_logger(log_path)

    t_global = time.perf_counter()
    section(log, "ANOMALY COMPARE — INICIADO")
    log.info(f"Data/hora   : {datetime.now():%Y-%m-%d %H:%M:%S}")
    log.info(f"Python      : {sys.version.split()[0]}")
    log.info(f"PyTorch     : {torch.__version__ if TORCH_OK else 'não instalado'}")
    gpu_ok = TORCH_OK and torch.cuda.is_available()
    log.info(f"CUDA        : {gpu_ok}")
    if gpu_ok:
        log.info(f"GPU         : {torch.cuda.get_device_name(0)}")
    log.info(f"GPU ativo   : {args.gpu}")
    log.info(f"Sample size : {args.sample_size:,}")

    # ── Extração ──────────────────────────────────────────────────────────────
    if not args.skip_extraction:
        run_extraction(cache_file, log)

    # ── Carrega dados ─────────────────────────────────────────────────────────
    section(log, "CARREGANDO CACHE")
    if not cache_file.exists():
        log.error(f"Cache não encontrado: {cache_file}")
        sys.exit(1)

    with open(cache_file, "rb") as f:
        data = pickle.load(f)

    if "X_scaled" in data:
        X = np.asarray(data["X_scaled"], dtype=np.float32)
    else:
        X = StandardScaler().fit_transform(
            np.asarray(data["X"], dtype=np.float32))

    log.info(f"Total no cache  : {X.shape[0]:,} × {X.shape[1]} features")
    if args.sample_size > 0 and X.shape[0] > args.sample_size:
        idx = np.random.default_rng(42).choice(X.shape[0], args.sample_size,
                                               replace=False)
        X = X[idx]
        log.info(f"Amostrado para  : {len(X):,}")

    # ── Divisão treino / teste ────────────────────────────────────────────────
    section(log, "DIVISÃO TREINO / TESTE")
    X_train, X_test = train_test_split(
        X, test_size=args.test_size, random_state=42, shuffle=True)
    log.info(f"Total de amostras : {len(X):,}")
    log.info(f"Treino            : {len(X_train):,} ({1-args.test_size:.0%})")
    log.info(f"Teste             : {len(X_test):,}  ({args.test_size:.0%})")
    log.info(f"Seed              : 42 (reproduzível)")
    log.info("")
    log.info("  O threshold será calculado APENAS no conjunto de treino.")
    log.info("  As anomalias serão detectadas APENAS no conjunto de teste.")
    log.info("  Isso evita data leakage e garante avaliação realista.")

    results: dict = {}

    # ── Autoencoder CPU ───────────────────────────────────────────────────────
    section(log, "AUTOENCODER — CPU")
    results["AE-CPU"] = run_autoencoder(
        X_train, X_test, "cpu", args.ae_epochs, args.ae_batch_size,
        args.ae_latent_dim, args.ae_lr, log)

    # ── Autoencoder GPU ───────────────────────────────────────────────────────
    section(log, "AUTOENCODER — GPU")
    if args.gpu and gpu_ok:
        results["AE-GPU"] = run_autoencoder(
            X_train, X_test, "cuda", args.ae_epochs, args.ae_batch_size,
            args.ae_latent_dim, args.ae_lr, log)
        ae_sp = results["AE-CPU"]["total_s"] / results["AE-GPU"]["total_s"]
        log.info(f"  Speedup GPU vs CPU : {ae_sp:.2f}x")
    else:
        msg = "GPU não disponível" if not gpu_ok else "flag --gpu não definida"
        log.info(f"  Pulando: {msg}")
        results["AE-GPU"] = {"status": "skipped", "notes": msg}

    # ── K-Means CPU ───────────────────────────────────────────────────────────
    section(log, "K-MEANS — CPU")
    results["KM-CPU"] = run_kmeans_cpu(
        X_train, X_test, args.kmeans_clusters, args.kmeans_max_iter, log)

    # ── K-Means GPU ───────────────────────────────────────────────────────────
    section(log, "K-MEANS — GPU")
    if args.gpu and gpu_ok:
        results["KM-GPU"] = run_kmeans_gpu(
            X_train, X_test, args.kmeans_clusters, args.kmeans_max_iter, log)
        km_sp = results["KM-CPU"]["total_s"] / results["KM-GPU"]["total_s"]
        log.info(f"  Speedup GPU vs CPU : {km_sp:.2f}x")
    else:
        msg = "GPU não disponível" if not gpu_ok else "flag --gpu não definida"
        log.info(f"  Pulando: {msg}")
        results["KM-GPU"] = {"status": "skipped", "notes": msg}

    # ── Comparativo final ─────────────────────────────────────────────────────
    section(log, "COMPARATIVO FINAL")
    header = f"  {'Método':<10} {'HW':<5} {'Treino':>9} {'Infer':>9} {'Total':>9} {'Speedup':>9} {'Anomalias':>10} {'Anom%':>7} {'SilhouetteScore':>16}"
    log.info(header)
    log.info("  " + "─" * 90)

    ae_ct = results["AE-CPU"].get("total_s")
    km_ct = results["KM-CPU"].get("total_s")

    for name, res in results.items():
        if res.get("status") != "ok":
            log.info(f"  {name:<10} {'—':<5} — — — — — — ({res.get('status')})")
            continue
        hw   = "GPU" if res["device"] in ("cuda","gpu") else "CPU"
        ref  = ae_ct if "AE" in name else km_ct
        gt   = res["total_s"]
        sp   = f"{ref/gt:.2f}x" if ref and gt and gt != ref else "ref"
        sil  = res.get("quality", {}).get("silhouette")
        sil_s = f"{sil:.4f}" if sil is not None else "  N/A  "
        log.info(
            f"  {name:<10} {hw:<5} "
            f"{res['train_s']:>9.3f}s "
            f"{res['infer_s']:>9.4f}s "
            f"{res['total_s']:>9.3f}s "
            f"{sp:>9} "
            f"{res['n_anomalies']:>10,} "
            f"{res['anomaly_rate']:>6.1f}% "
            f"{sil_s:>16}"
        )

    # ── Relatório visual ──────────────────────────────────────────────────────
    plot_report(results, outdir / "anomaly_report.png", args, log)

    # ── Relatório detalhado em TXT ────────────────────────────────────────────
    total_s = time.perf_counter() - t_global
    write_text_report(results, log_path, args, total_s, log)

    # ── Encerramento ──────────────────────────────────────────────────────────
    section(log, "CONCLUÍDO")
    log.info(f"Tempo total  : {total_s:.2f}s ({total_s/60:.1f} min)")
    log.info(f"TXT detalhado : {log_path}")
    log.info(f"Relatório    : {outdir / 'anomaly_report.png'}")
    log.info("=" * 72)


if __name__ == "__main__":
    main()
