"""
anomaly_compare.py
══════════════════════════════════════════════════════════════════════════════
Comparativo unificado de detecção de anomalias em tráfego de rede.
Executa Autoencoder e K-Means em CPU e GPU, mede qualidade e desempenho,
e gera relatório completo (log + PNG + JSON + CSV).

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
    python anomaly_compare.py --skip-extraction --gpu --sample-size 20000
    python anomaly_compare.py --skip-extraction --gpu --ae-epochs 30 --kmeans-clusters 10

Dependências:
    pip install torch scikit-learn pandas numpy matplotlib
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
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.patches as mpatches
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
    fh = logging.FileHandler(log_path, encoding="utf-8")
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


def run_autoencoder(X: np.ndarray, device: str, epochs: int,
                    batch_size: int, latent_dim: int, lr: float,
                    log: logging.Logger) -> dict:
    if not TORCH_OK:
        return {"status": "error", "notes": "PyTorch não disponível"}

    log.info(f"  Device      : {device.upper()}")
    log.info(f"  Épocas      : {epochs}")
    log.info(f"  Batch size  : {batch_size}")
    log.info(f"  Latent dim  : {latent_dim}")
    log.info(f"  LR          : {lr}")
    log.info(f"  Amostras    : {len(X):,}")

    model = Autoencoder(X.shape[1], latent_dim).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    tensor_X = torch.from_numpy(X.astype(np.float32))
    loader_tr = DataLoader(TensorDataset(tensor_X), batch_size=batch_size,
                           shuffle=True,  pin_memory=(device=="cuda"))
    loader_in = DataLoader(TensorDataset(tensor_X), batch_size=batch_size,
                           shuffle=False, pin_memory=(device=="cuda"))

    # ── Treino ────────────────────────────────────────────────────────────────
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
        avg = ep_loss / len(X)
        epoch_losses.append(avg)
        log.info(f"  [AE-{device.upper()}] Época {epoch:>3}/{epochs} | loss={avg:.6f}")
    _sync(device)
    train_s = time.perf_counter() - t_train

    # ── Inferência ────────────────────────────────────────────────────────────
    _sync(device)
    t_infer = time.perf_counter()
    model.eval()
    errors: list[np.ndarray] = []
    with torch.no_grad():
        for (batch,) in loader_in:
            batch = batch.to(device, non_blocking=(device=="cuda"))
            recon = model(batch)
            errors.append(((recon - batch)**2).mean(dim=1).cpu().numpy())
    _sync(device)
    infer_s = time.perf_counter() - t_infer

    scores = np.concatenate(errors)
    log.info(f"  Treino      : {train_s:.3f}s")
    log.info(f"  Inferência  : {infer_s:.3f}s")

    # ── Qualidade ─────────────────────────────────────────────────────────────
    log.info("")
    score_dist  = score_distribution(scores, f"AE-{device.upper()}", log)
    thresh_anal = anomaly_rate_analysis(scores, f"AE-{device.upper()}", log)

    # Threshold padrão p95
    threshold = float(np.percentile(scores, 95))
    labels    = (scores > threshold).astype(int)
    n_anom    = int(labels.sum())
    anom_rate = n_anom / len(scores) * 100

    log.info(f"  Anomalias detectadas (p95) : {n_anom:,} ({anom_rate:.2f}%)")

    return {
        "status":        "ok",
        "device":        device,
        "train_s":       round(train_s, 4),
        "infer_s":       round(infer_s, 4),
        "total_s":       round(train_s + infer_s, 4),
        "epoch_losses":  [round(l, 6) for l in epoch_losses],
        "final_loss":    round(epoch_losses[-1], 6),
        "scores":        scores,
        "threshold_p95": threshold,
        "n_anomalies":   n_anom,
        "anomaly_rate":  round(anom_rate, 2),
        "score_dist":    score_dist,
        "threshold_analysis": thresh_anal,
        "notes":         f"epochs={epochs} latent={latent_dim} batch={batch_size}",
    }


# ══════════════════════════════════════════════════════════════════════════════
# K-MEANS CPU
# ══════════════════════════════════════════════════════════════════════════════

def run_kmeans_cpu(X: np.ndarray, k: int, max_iter: int,
                   log: logging.Logger) -> dict:
    log.info(f"  Device    : CPU (scikit-learn)")
    log.info(f"  k         : {k}")
    log.info(f"  max_iter  : {max_iter}")
    log.info(f"  Amostras  : {len(X):,}")

    model = KMeans(n_clusters=k, max_iter=max_iter, n_init=10, random_state=42)

    t0 = time.perf_counter()
    model.fit(X)
    train_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    labels = model.predict(X)
    infer_s = time.perf_counter() - t1

    # Score de anomalia: distância ao centroide mais próximo
    centroids = model.cluster_centers_
    dists = np.linalg.norm(X - centroids[labels], axis=1)

    log.info(f"  Iterações : {model.n_iter_} / {max_iter}")
    log.info(f"  Inércia   : {model.inertia_:.4f}")
    log.info(f"  Treino    : {train_s:.3f}s")
    log.info(f"  Inferência: {infer_s:.4f}s")

    log.info("")
    quality     = kmeans_quality(X, labels, log)
    score_dist  = score_distribution(dists, "KM-CPU", log)
    thresh_anal = anomaly_rate_analysis(dists, "KM-CPU", log)

    threshold = float(np.percentile(dists, 95))
    n_anom    = int((dists > threshold).sum())
    anom_rate = n_anom / len(dists) * 100
    log.info(f"  Anomalias detectadas (p95) : {n_anom:,} ({anom_rate:.2f}%)")

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
        "n_anomalies":   n_anom,
        "anomaly_rate":  round(anom_rate, 2),
        "score_dist":    score_dist,
        "threshold_analysis": thresh_anal,
        "quality":       quality,
        "notes":         f"k={k} n_iter={model.n_iter_} inertia={model.inertia_:.2f}",
    }


# ══════════════════════════════════════════════════════════════════════════════
# K-MEANS GPU (PyTorch puro)
# ══════════════════════════════════════════════════════════════════════════════

def run_kmeans_gpu(X: np.ndarray, k: int, max_iter: int,
                   log: logging.Logger) -> dict:
    if not TORCH_OK or not torch.cuda.is_available():
        msg = "CUDA não disponível"
        log.warning(f"  {msg}")
        return {"status": "unavailable", "notes": msg}

    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    log.info(f"  Device    : GPU — {gpu_name}")
    log.info(f"  k         : {k}")
    log.info(f"  max_iter  : {max_iter}")
    log.info(f"  Amostras  : {len(X):,}")

    X_t = torch.from_numpy(X.astype(np.float32)).to(device)
    rng = torch.Generator(device=device); rng.manual_seed(42)

    # ── kmeans++ na GPU ───────────────────────────────────────────────────────
    torch.cuda.synchronize()
    t_train = time.perf_counter()

    idx0 = torch.randint(0, X_t.shape[0], (1,), generator=rng, device=device).item()
    centroids = X_t[idx0].unsqueeze(0)
    for _ in range(1, k):
        d = torch.cdist(X_t, centroids).min(dim=1).values
        prob = d / d.sum()
        idx = torch.multinomial(prob, 1, generator=rng).item()
        centroids = torch.cat([centroids, X_t[idx].unsqueeze(0)], dim=0)

    labels_t = torch.zeros(X_t.shape[0], dtype=torch.long, device=device)
    inertia_history = []

    for iteration in range(max_iter):
        dists_all  = torch.cdist(X_t, centroids)
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
                centroids[ki] = X_t[mask].mean(dim=0)

    torch.cuda.synchronize()
    train_s = time.perf_counter() - t_train

    # ── Inferência ────────────────────────────────────────────────────────────
    torch.cuda.synchronize()
    t_infer = time.perf_counter()
    dists_final  = torch.cdist(X_t, centroids)
    labels_final = dists_final.argmin(dim=1)
    dists_min    = dists_final.min(dim=1).values
    torch.cuda.synchronize()
    infer_s = time.perf_counter() - t_infer

    labels_np = labels_final.cpu().numpy()
    scores_np = dists_min.cpu().numpy()
    inertia_final = float(inertia_history[-1])

    log.info(f"  Iterações : {len(inertia_history)} / {max_iter}")
    log.info(f"  Inércia   : {inertia_final:.4f}")
    log.info(f"  Treino    : {train_s:.3f}s")
    log.info(f"  Inferência: {infer_s:.4f}s")

    log.info("")
    quality     = kmeans_quality(X, labels_np, log)
    score_dist  = score_distribution(scores_np, "KM-GPU", log)
    thresh_anal = anomaly_rate_analysis(scores_np, "KM-GPU", log)

    threshold = float(np.percentile(scores_np, 95))
    n_anom    = int((scores_np > threshold).sum())
    anom_rate = n_anom / len(scores_np) * 100
    log.info(f"  Anomalias detectadas (p95) : {n_anom:,} ({anom_rate:.2f}%)")

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
        "n_anomalies":   n_anom,
        "anomaly_rate":  round(anom_rate, 2),
        "score_dist":    score_dist,
        "threshold_analysis": thresh_anal,
        "quality":       quality,
        "notes":         f"k={k} n_iter={len(inertia_history)} inertia={inertia_final:.2f} (PyTorch GPU)",
    }


# ══════════════════════════════════════════════════════════════════════════════
# RELATÓRIO VISUAL
# ══════════════════════════════════════════════════════════════════════════════

def plot_report(results: dict, out_path: Path, args: argparse.Namespace,
                log: logging.Logger) -> None:
    if not MATPLOTLIB_OK:
        log.warning("matplotlib indisponível — relatório visual não gerado")
        return

    DARK  = "#0D1B2A"; PANEL = "#1B2A3B"; GRID  = "#263445"
    TEXT  = "#F0F4F8"; MUTED = "#7B92A8"
    BLUE  = "#1C7293"; TEAL  = "#028090"; MINT  = "#02C39A"
    AMBER = "#F59E0B"; RED   = "#EF4444"; PURPLE= "#7C3AED"

    fig = plt.figure(figsize=(22, 26), facecolor=DARK)
    gs  = gridspec.GridSpec(5, 2, figure=fig, hspace=0.52, wspace=0.32,
                            top=0.95, bottom=0.04, left=0.07, right=0.96)

    def style(ax, title):
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values(): sp.set_edgecolor(GRID)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.set_title(title, color=TEXT, fontsize=11, fontweight="bold", pad=10)
        ax.grid(axis="y", color=GRID, linestyle="--", linewidth=0.6, alpha=0.7)

    gpu_ok  = TORCH_OK and torch.cuda.is_available()
    gpu_lbl = torch.cuda.get_device_name(0) if gpu_ok else "N/A"

    # ── 0. Header ─────────────────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, :])
    ax0.set_facecolor(PANEL)
    for sp in ax0.spines.values(): sp.set_edgecolor(TEAL)
    ax0.set_xticks([]); ax0.set_yticks([])
    ax0.text(0.5, 0.74, "Parallel PCAP — Detecção de Anomalias CPU vs GPU",
             transform=ax0.transAxes, ha="center", color=TEXT,
             fontsize=17, fontweight="bold")
    ax0.text(0.5, 0.30,
             f"Gerado: {datetime.now():%Y-%m-%d %H:%M}  |  "
             f"Amostras: {args.sample_size:,}  |  "
             f"AE épocas: {args.ae_epochs}  |  "
             f"K-Means k={args.kmeans_clusters}  |  "
             f"GPU: {gpu_lbl}",
             transform=ax0.transAxes, ha="center", color=MUTED, fontsize=10)

    # ── 1. Anomaly Rate — todos os métodos ────────────────────────────────────
    ax1 = fig.add_subplot(gs[1, 0])
    style(ax1, "Taxa de Anomalia por Método e Threshold")
    labels_x = ["p90", "p92", "p95", "p97", "p99"]
    method_colors = {
        "AE-CPU":  BLUE,  "AE-GPU":  TEAL,
        "KM-CPU":  AMBER, "KM-GPU":  MINT,
    }
    x = np.arange(len(labels_x))
    width = 0.2
    offsets = {"AE-CPU": -1.5, "AE-GPU": -0.5, "KM-CPU": 0.5, "KM-GPU": 1.5}
    plotted = False
    for name, res in results.items():
        if res.get("status") != "ok" or "threshold_analysis" not in res:
            continue
        rates = [res["threshold_analysis"].get(p, {}).get("anomaly_rate_pct", 0)
                 for p in labels_x]
        off = offsets.get(name, 0)
        ax1.bar(x + off * width, rates, width=width,
                color=method_colors.get(name, MUTED),
                edgecolor=DARK, linewidth=0.8, label=name)
        plotted = True
    if plotted:
        ax1.set_xticks(x); ax1.set_xticklabels(labels_x, color=TEXT)
        ax1.set_ylabel("% anomalias", color=MUTED, fontsize=9)
        ax1.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=9)

    # ── 2. Score Distribution — box-style ────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 1])
    style(ax2, "Distribuição dos Scores de Anomalia (p50/p95/p99)")
    box_data = []
    box_labels = []
    box_colors = []
    for name, res in results.items():
        if res.get("status") != "ok" or "scores" not in res:
            continue
        scores = res["scores"]
        box_data.append([np.percentile(scores, p) for p in [25, 50, 75, 95, 99]])
        box_labels.append(name)
        box_colors.append(method_colors.get(name, MUTED))

    if box_data:
        positions = np.arange(len(box_data))
        for i, (bd, col) in enumerate(zip(box_data, box_colors)):
            ax2.bar(i, bd[4], color=col, alpha=0.25, edgecolor=DARK, width=0.6)
            ax2.bar(i, bd[2], color=col, alpha=0.7,  edgecolor=DARK, width=0.6)
            ax2.plot([i - 0.3, i + 0.3], [bd[1], bd[1]],
                     color=TEXT, linewidth=2.5, zorder=3)
            ax2.plot([i - 0.15, i + 0.15], [bd[3], bd[3]],
                     color=AMBER, linewidth=2, linestyle="--", zorder=3)
        ax2.set_xticks(positions)
        ax2.set_xticklabels(box_labels, color=TEXT, fontsize=9)
        ax2.set_ylabel("Score", color=MUTED, fontsize=9)
        # Legenda manual
        legend_els = [
            mpatches.Patch(color=TEXT, alpha=0.9, label="mediana (p50)"),
            mpatches.Patch(color=AMBER, alpha=0.9, label="p95 (threshold)"),
        ]
        ax2.legend(handles=legend_els, facecolor=PANEL,
                   edgecolor=GRID, labelcolor=TEXT, fontsize=8)

    # ── 3. Loss Autoencoder por época ─────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, 0])
    style(ax3, "Autoencoder — Loss por Época")
    for name, col in [("AE-CPU", BLUE), ("AE-GPU", TEAL)]:
        res = results.get(name, {})
        if res.get("status") == "ok" and res.get("epoch_losses"):
            epochs = range(1, len(res["epoch_losses"]) + 1)
            ax3.plot(epochs, res["epoch_losses"], color=col,
                     linewidth=2.5, marker="o", markersize=5,
                     markerfacecolor=TEXT, markeredgecolor=col,
                     label=name, zorder=3)
            ax3.fill_between(epochs, res["epoch_losses"], alpha=0.1, color=col)
    ax3.set_xlabel("Época", color=MUTED, fontsize=9)
    ax3.set_ylabel("MSE Loss", color=MUTED, fontsize=9)
    ax3.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=9)
    ax3.tick_params(axis="x", colors=TEXT)

    # ── 4. Silhouette e Inércia K-Means ───────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 1])
    style(ax4, "K-Means — Silhouette Score e Inércia")
    km_names, sil_vals, inertia_vals = [], [], []
    for name in ["KM-CPU", "KM-GPU"]:
        res = results.get(name, {})
        if res.get("status") == "ok":
            km_names.append(name)
            sil_vals.append(res.get("quality", {}).get("silhouette") or 0)
            inertia_vals.append(res.get("inertia", 0))
    if km_names:
        x_km = np.arange(len(km_names))
        ax4_twin = ax4.twinx()
        ax4_twin.set_facecolor(PANEL)
        ax4_twin.tick_params(colors=MUTED, labelsize=9)
        bars_sil = ax4.bar(x_km - 0.2, sil_vals, width=0.35,
                           color=[AMBER, MINT][:len(km_names)],
                           edgecolor=DARK, label="Silhouette")
        bars_in  = ax4_twin.bar(x_km + 0.2, inertia_vals, width=0.35,
                                color=[BLUE, TEAL][:len(km_names)],
                                alpha=0.6, edgecolor=DARK, label="Inércia")
        ax4.set_xticks(x_km)
        ax4.set_xticklabels(km_names, color=TEXT)
        ax4.set_ylabel("Silhouette (−1 → +1)", color=AMBER, fontsize=9)
        ax4_twin.set_ylabel("Inércia", color=BLUE, fontsize=9)
        for bar, v in zip(bars_sil, sil_vals):
            ax4.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.005,
                     f"{v:.4f}", ha="center", color=TEXT, fontsize=9)
        for bar, v in zip(bars_in, inertia_vals):
            ax4_twin.text(bar.get_x() + bar.get_width()/2,
                          bar.get_height() + max(inertia_vals)*0.01,
                          f"{v:,.0f}", ha="center", color=TEXT, fontsize=9)
        ax4.grid(axis="y", color=GRID, linestyle="--", linewidth=0.6, alpha=0.5)

    # ── 5. Tabela comparativa ─────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[3, :])
    ax5.set_facecolor(PANEL)
    for sp in ax5.spines.values(): sp.set_edgecolor(GRID)
    ax5.set_xticks([]); ax5.set_yticks([])
    ax5.set_title("Comparativo Completo — Qualidade e Desempenho",
                  color=TEXT, fontsize=11, fontweight="bold", pad=10)

    hdrs = ["Método", "HW", "Treino(s)", "Infer(s)", "Total(s)",
            "Speedup", "Anomalias", "Anom%", "Score p95", "Status"]
    col_x = [0.01, 0.10, 0.19, 0.28, 0.37, 0.46, 0.55, 0.64, 0.73, 0.85]
    for cx, h in zip(col_x, hdrs):
        ax5.text(cx, 0.88, h, transform=ax5.transAxes,
                 color=TEAL, fontsize=9, fontweight="bold", va="top")
    line = plt.Line2D([0.01, 0.99], [0.78, 0.78],
                      transform=ax5.transAxes, color=GRID, linewidth=1)
    ax5.add_line(line)

    # Calcula speedups
    ae_cpu_t  = results.get("AE-CPU", {}).get("total_s")
    ae_gpu_t  = results.get("AE-GPU", {}).get("total_s")
    km_cpu_t  = results.get("KM-CPU", {}).get("total_s")
    km_gpu_t  = results.get("KM-GPU", {}).get("total_s")
    ae_sp = f"{ae_cpu_t/ae_gpu_t:.2f}x" if ae_cpu_t and ae_gpu_t else "—"
    km_sp = f"{km_cpu_t/km_gpu_t:.2f}x" if km_cpu_t and km_gpu_t else "—"

    def fs(v): return f"{v:.3f}" if isinstance(v, float) else "—"
    def fi(v): return f"{v:,}" if isinstance(v, int) else "—"

    table_rows = [
        ("AE-CPU",  "CPU", results.get("AE-CPU",{}), "1.00x"),
        ("AE-GPU",  "GPU", results.get("AE-GPU",{}), ae_sp),
        ("KM-CPU",  "CPU", results.get("KM-CPU",{}), "1.00x"),
        ("KM-GPU",  "GPU", results.get("KM-GPU",{}), km_sp),
    ]
    row_ys = [0.62, 0.44, 0.26, 0.08]

    for ri, (name, hw, res, sp) in enumerate(table_rows):
        bg = "#162030" if ri % 2 else "#1B2A3B"
        rect = mpatches.FancyBboxPatch(
            (0.005, row_ys[ri]-0.10), 0.99, 0.18,
            boxstyle="round,pad=0.01", facecolor=bg,
            edgecolor=GRID, linewidth=0.5,
            transform=ax5.transAxes, clip_on=False)
        ax5.add_patch(rect)

        status = res.get("status", "—")
        vals = [
            name,
            hw,
            fs(res.get("train_s")),
            fs(res.get("infer_s")),
            fs(res.get("total_s")),
            sp,
            fi(res.get("n_anomalies")),
            f"{res.get('anomaly_rate', '—'):.1f}%" if isinstance(res.get('anomaly_rate'), float) else "—",
            f"{res.get('threshold_p95', 0):.4f}" if res.get("status")=="ok" else "—",
            "✓ ok" if status=="ok" else f"⊘ {status}",
        ]
        for ci, (val, cx) in enumerate(zip(vals, col_x)):
            color = TEXT
            if ci == 1:   color = BLUE if hw=="CPU" else MINT
            elif ci == 5: color = AMBER if sp not in ("1.00x","—") else MUTED
            elif ci == 9: color = MINT if "ok" in val else RED
            ax5.text(cx, row_ys[ri], val, transform=ax5.transAxes,
                     color=color, fontsize=8.5, va="center")

    # ── 6. Speedup + Anomaly Rate side-by-side ────────────────────────────────
    ax6 = fig.add_subplot(gs[4, 0])
    style(ax6, "Speedup GPU vs CPU")
    sp_names, sp_vals, sp_cols = [], [], []
    for algo, cpu_k, gpu_k, col in [("Autoencoder","AE-CPU","AE-GPU",BLUE),
                                     ("K-Means","KM-CPU","KM-GPU",AMBER)]:
        ct = results.get(cpu_k, {}).get("total_s")
        gt = results.get(gpu_k, {}).get("total_s")
        if ct and gt:
            sp_names.append(algo)
            sp_vals.append(round(ct/gt, 2))
            sp_cols.append(col)
    if sp_names:
        bars = ax6.bar(sp_names, sp_vals, color=sp_cols,
                       edgecolor=DARK, linewidth=1, width=0.45)
        ax6.axhline(1.0, color=MUTED, linewidth=1.5, linestyle="--", alpha=0.7)
        ax6.text(len(sp_names)-0.5, 1.05, "CPU baseline",
                 color=MUTED, fontsize=8)
        for bar, v in zip(bars, sp_vals):
            col_label = MINT if v > 1 else RED
            ax6.text(bar.get_x()+bar.get_width()/2,
                     bar.get_height()+0.1,
                     f"{v:.2f}x", ha="center",
                     color=col_label, fontsize=12, fontweight="bold")
        ax6.set_ylabel("Speedup (maior = melhor GPU)", color=MUTED, fontsize=9)
        ax6.tick_params(axis="x", colors=TEXT)

    ax7 = fig.add_subplot(gs[4, 1])
    style(ax7, "Taxa de Anomalia @ p95 por Método")
    ar_names, ar_vals, ar_cols = [], [], []
    for name, col in method_colors.items():
        res = results.get(name, {})
        if res.get("status") == "ok":
            ar_names.append(name)
            ar_vals.append(res.get("anomaly_rate", 0))
            ar_cols.append(col)
    if ar_names:
        bars = ax7.bar(ar_names, ar_vals, color=ar_cols,
                       edgecolor=DARK, linewidth=1, width=0.45)
        ax7.axhline(5.0, color=AMBER, linewidth=1.5, linestyle="--", alpha=0.8)
        ax7.text(len(ar_names)-1, 5.2, "p95 esperado = 5%",
                 color=AMBER, fontsize=8)
        for bar, v in zip(bars, ar_vals):
            ax7.text(bar.get_x()+bar.get_width()/2,
                     bar.get_height()+0.1,
                     f"{v:.1f}%", ha="center", color=TEXT,
                     fontsize=10, fontweight="bold")
        ax7.set_ylabel("% de fluxos anômalos", color=MUTED, fontsize=9)
        ax7.tick_params(axis="x", colors=TEXT)

    fig.patch.set_facecolor(DARK)
    plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=DARK)
    plt.close(fig)
    log.info(f"Relatório visual salvo em : {out_path}")


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
    args = parser.parse_args()

    outdir     = Path(args.outdir)
    cache_file = Path(args.cache_file)
    outdir.mkdir(parents=True, exist_ok=True)
    log_path   = outdir / "anomaly_compare.log"
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

    results: dict = {}

    # ── Autoencoder CPU ───────────────────────────────────────────────────────
    section(log, "AUTOENCODER — CPU")
    results["AE-CPU"] = run_autoencoder(
        X, "cpu", args.ae_epochs, args.ae_batch_size,
        args.ae_latent_dim, args.ae_lr, log)

    # ── Autoencoder GPU ───────────────────────────────────────────────────────
    section(log, "AUTOENCODER — GPU")
    if args.gpu and gpu_ok:
        results["AE-GPU"] = run_autoencoder(
            X, "cuda", args.ae_epochs, args.ae_batch_size,
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
        X, args.kmeans_clusters, args.kmeans_max_iter, log)

    # ── K-Means GPU ───────────────────────────────────────────────────────────
    section(log, "K-MEANS — GPU")
    if args.gpu and gpu_ok:
        results["KM-GPU"] = run_kmeans_gpu(
            X, args.kmeans_clusters, args.kmeans_max_iter, log)
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

    # ── Salva JSON + CSV ──────────────────────────────────────────────────────
    def safe(v):
        if isinstance(v, np.ndarray): return v.tolist()
        if isinstance(v, (np.integer,)): return int(v)
        if isinstance(v, (np.floating,)): return float(v)
        return v

    payload = {"args": vars(args), "timestamp": datetime.now().isoformat(),
               "results": {k: {f: safe(v) for f, v in res.items()
                               if f not in ("scores","labels")}
                           for k, res in results.items()}}

    json_path = outdir / "anomaly_compare.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)

    csv_path = outdir / "anomaly_compare.csv"
    csv_rows = []
    for name, res in results.items():
        if res.get("status") != "ok":
            continue
        csv_rows.append({
            "method": name,
            "hardware": "GPU" if res["device"]=="cuda" else "CPU",
            "train_s": res.get("train_s"),
            "infer_s": res.get("infer_s"),
            "total_s": res.get("total_s"),
            "n_anomalies": res.get("n_anomalies"),
            "anomaly_rate_pct": res.get("anomaly_rate"),
            "threshold_p95": res.get("threshold_p95"),
            "final_loss": res.get("final_loss"),
            "silhouette": res.get("quality", {}).get("silhouette"),
            "inertia": res.get("inertia"),
            "n_iter": res.get("n_iter"),
            "notes": res.get("notes"),
        })
    if csv_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

    log.info(f"JSON : {json_path}")
    log.info(f"CSV  : {csv_path}")

    # ── Relatório visual ──────────────────────────────────────────────────────
    plot_report(results, outdir / "anomaly_report.png", args, log)

    # ── Encerramento ──────────────────────────────────────────────────────────
    total_s = time.perf_counter() - t_global
    section(log, "CONCLUÍDO")
    log.info(f"Tempo total  : {total_s:.2f}s ({total_s/60:.1f} min)")
    log.info(f"Log          : {log_path}")
    log.info(f"Relatório    : {outdir / 'anomaly_report.png'}")
    log.info("=" * 72)


if __name__ == "__main__":
    main()
