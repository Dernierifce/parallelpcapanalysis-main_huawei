"""
plots_cpu_gpu_compare.py
Gera gráficos de comparação CPU federado vs GPU benchmark.

Foco: Tempo de CLASSIFICAÇÃO (train + infer), não incluindo extração de features.
      Se arquivos foram gerados com --cache-file, isolam completamente classificação.
"""

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Comparação CPU federado vs GPU (foco em CLASSIFICAÇÃO)"
    )
    parser.add_argument("--cpu-results", required=True, help="Saída de federated_train.py")
    parser.add_argument("--gpu-results", required=True, help="Saída de gpu_train.py")
    parser.add_argument("--outdir", default="/data/results")
    parser.add_argument("--basename", default="cpu_gpu_comparison")
    args = parser.parse_args()

    cpu = _load_pickle(args.cpu_results)
    gpu = _load_pickle(args.gpu_results)

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    # ── EXTRAIR MÉTRICAS CPU ──────────────────────────────────────────
    cpu_mode = cpu.get("mode", "unknown")
    cpu_total_time = float(cpu.get("total_time_s", 0.0))
    cpu_flows = int(sum(r.get("n_flows", 0) for r in cpu.get("final_round", [])))
    cpu_anom = int(sum(r.get("n_anomalies", 0) for r in cpu.get("final_round", [])))
    cpu_rate = float(cpu_anom / cpu_flows) if cpu_flows else 0.0

    # Tempo de classificação CPU (soma de todos os rounds)
    cpu_classification_time = 0.0
    for round_info in cpu.get("history", []):
        cpu_classification_time += round_info.get("serial_eq_s", 0.0)

    # ── EXTRAIR MÉTRICAS GPU ──────────────────────────────────────────
    gpu_mode = gpu.get("mode", "unknown")
    gpu_times = gpu.get("times", {})
    gpu_total_time = float(gpu_times.get("total_s", 0.0))
    gpu_extract = float(gpu_times.get("extract_s", 0.0))
    gpu_train = float(gpu_times.get("train_s", 0.0))
    gpu_infer = float(gpu_times.get("infer_s", 0.0))
    gpu_classification_time = float(gpu_times.get("classification_s", 0.0))
    gpu_flows = int(gpu.get("n_flows", 0))
    gpu_anom = int(gpu.get("n_anomalies", 0))
    gpu_rate = float(gpu.get("anom_rate", 0.0))

    # Se não temos classification_s explícito, usar train + infer
    if gpu_classification_time == 0.0:
        gpu_classification_time = gpu_train + gpu_infer

    # ── CALCULAR SPEEDUP ──────────────────────────────────────────────
    # Priorizar comparação por classification_time se ambos estão em modo cache
    if cpu_mode == "cache" and gpu_mode == "cache":
        speedup = cpu_classification_time / gpu_classification_time if gpu_classification_time > 0 else 0.0
        comparison_mode = "CLASSIFICAÇÃO (cache)"
        cpu_time_used = cpu_classification_time
        gpu_time_used = gpu_classification_time
    else:
        # Fallback: comparar por total_time
        speedup = cpu_total_time / gpu_total_time if gpu_total_time > 0 else 0.0
        comparison_mode = "TOTAL (inclui extração)"
        cpu_time_used = cpu_total_time
        gpu_time_used = gpu_total_time

    # ── CRIAR FIGURAS ─────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("#F8F9FA")

    # Gráfico 1: Tempo de CLASSIFICAÇÃO (métrica principal)
    ax = axes[0, 0]
    labels = ["CPU federado", "GPU benchmark"]
    times = [cpu_time_used, gpu_time_used]
    colors = ["#2563EB", "#16A34A"]
    bars = ax.bar(labels, times, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5)
    for bar, val in zip(bars, times):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val,
            f"{val:.1f}s",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    ax.set_title("Tempo de CLASSIFICAÇÃO (train + infer)", fontweight="bold", fontsize=12)
    ax.set_ylabel("Segundos", fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.text(0.5, 0.95, f"Speedup: {speedup:.2f}×", transform=ax.transAxes,
            ha="center", va="top", fontsize=14, fontweight="bold",
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "#FFEB3B", "alpha": 0.7})

    # Gráfico 2: Quebra de tempos GPU (se houver extração)
    ax = axes[0, 1]
    if gpu_extract > 0:
        parts_labels = ["Extração\n(não medido)", "Treino", "Inferência"]
        gpu_parts = [gpu_extract, gpu_train, gpu_infer]
        colors_parts = ["#94A3B8", "#16A34A", "#F59E0B"]
    else:
        parts_labels = ["Treino", "Inferência"]
        gpu_parts = [gpu_train, gpu_infer]
        colors_parts = ["#16A34A", "#F59E0B"]

    part_bars = ax.bar(parts_labels, gpu_parts, color=colors_parts, alpha=0.7, edgecolor="black", linewidth=1.5)
    for bar, val in zip(part_bars, gpu_parts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val,
            f"{val:.2f}s",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    ax.set_title("Componentes do tempo GPU", fontweight="bold", fontsize=12)
    ax.set_ylabel("Segundos", fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Gráfico 3: Taxa de anomalias
    ax = axes[1, 0]
    rates = [cpu_rate * 100, gpu_rate * 100]
    rate_bars = ax.bar(labels, rates, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5)
    for bar, val in zip(rate_bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val,
            f"{val:.2f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    ax.set_title("Taxa de anomalias detectadas", fontweight="bold", fontsize=12)
    ax.set_ylabel("% de fluxos", fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Gráfico 4: Resumo (texto)
    ax = axes[1, 1]
    ax.axis("off")

    summary_text = (
        f"MODO DE COMPARAÇÃO: {comparison_mode}\n"
        f"CPU: {cpu_mode.upper()} | GPU: {gpu_mode.upper()}\n"
        f"\n"
        f"CLASSIFICAÇÃO:\n"
        f"  CPU: {cpu_time_used:.1f}s\n"
        f"  GPU: {gpu_time_used:.1f}s\n"
        f"  Speedup: {speedup:.2f}×\n"
        f"\n"
        f"FLUXOS E ANOMALIAS:\n"
        f"  CPU: {cpu_flows:,} fluxos, {cpu_anom:,} anomalias\n"
        f"  GPU: {gpu_flows:,} fluxos, {gpu_anom:,} anomalias\n"
        f"\n"
        f"BACKEND GPU: {gpu.get('backend', 'n/a')}\n"
    )

    if gpu_extract > 0:
        summary_text += (
            f"\nEXTRAÇÃO (não medida em modo cache):\n"
            f"  GPU: {gpu_extract:.1f}s\n"
        )

    ax.text(
        0.05,
        0.95,
        summary_text,
        ha="left",
        va="top",
        fontsize=11,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.8", "facecolor": "#E3F2FD", "edgecolor": "#1976D2", "linewidth": 2},
    )

    fig.suptitle(
        "Comparação de Desempenho: CPU Federado vs GPU Benchmark\n"
        "(Foco em Classificação de Anomalias)",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()

    png_path = Path(args.outdir) / f"{args.basename}.png"
    pdf_path = Path(args.outdir) / f"{args.basename}.pdf"

    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    print("\n" + "=" * 70)
    print(f"  Gráficos gerados com sucesso!")
    print(f"  PNG: {png_path}")
    print(f"  PDF: {pdf_path}")
    print("=" * 70)
    print(f"\n  Resumo da Comparação:")
    print(f"  ─────────────────────")
    print(f"  Modo: {comparison_mode}")
    print(f"  CPU ({cpu_mode}): {cpu_time_used:.1f}s")
    print(f"  GPU ({gpu_mode}): {gpu_time_used:.1f}s")
    print(f"  Speedup: {speedup:.2f}×")
    print("=" * 70)


if __name__ == "__main__":
    main()
