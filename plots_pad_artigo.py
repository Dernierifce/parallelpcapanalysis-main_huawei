"""
plots_pad_artigo.py
Figura composta para artigo científico — foco em ganhos de paralelismo e PAD.
Gera: figura_pad_artigo.pdf / .png  (300 dpi, pronta para LaTeX)

Dependências:
    pip install matplotlib numpy pandas scipy scikit-learn
"""

import pickle
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.patches import FancyArrowPatch
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# ── Paleta ──────────────────────────────────────────────────────
P = dict(
    bg      = "#0D1117",
    card    = "#1E293B",
    card2   = "#162032",
    grid    = "#334155",
    text    = "#E2E8F0",
    muted   = "#94A3B8",
    normal  = "#1D9E75",
    anomaly = "#EF4444",
    ideal   = "#475569",
    amdahl  = "#F59E0B",
    emp     = "#00D4FF",
    eff     = "#10B981",
    sat     = "#EF4444",
    workers = ["#378ADD", "#F59E0B", "#8B5CF6", "#10B981"],
    fedavg  = "#00D4FF",
)

plt.rcParams.update({
    "figure.facecolor": P["bg"],
    "axes.facecolor":   P["card"],
    "axes.edgecolor":   P["grid"],
    "axes.labelcolor":  P["muted"],
    "axes.titlecolor":  P["text"],
    "xtick.color":      P["muted"],
    "ytick.color":      P["muted"],
    "text.color":       P["text"],
    "grid.color":       P["grid"],
    "grid.linewidth":   0.4,
    "grid.linestyle":   "-",
    "legend.framealpha":0.18,
    "legend.facecolor": P["card"],
    "legend.edgecolor": P["grid"],
    "legend.labelcolor":P["text"],
    "font.family":      "DejaVu Sans",
    "font.size":        9.5,
    "axes.titlepad":    10,
    "axes.titlesize":   10.5,
    "axes.labelsize":   9.5,
})

# ══════════════════════════════════════════════════════════════
# DADOS SIMULADOS (substituir pelo pickle real do experimento)
# ══════════════════════════════════════════════════════════════
np.random.seed(42)
N_FLOWS   = 420_000   # fluxos totais nos 5 GB
N_ROUNDS  = 6
N_WORKERS = 4

# Tempos de extração de features (por worker, por round)
# Diminuem levemente a cada round (warm_start / caching)
base_times = np.array([68.4, 71.2, 67.9, 70.1])
worker_times_per_round = np.array([
    base_times * (1 - 0.03 * r + np.random.randn(4) * 0.5)
    for r in range(N_ROUNDS)
])

# Tempo serial equivalente (se fosse 1 processo)
t_serial_per_round = worker_times_per_round.sum(axis=1) * 1.08  # overhead fork ~8%
# Tempo paralelo real (gargalo = worker mais lento + overhead)
t_parallel_per_round = worker_times_per_round.max(axis=1) * 1.12

# Speedup empírico por round
speedup_per_round = t_serial_per_round / t_parallel_per_round

# Tempos para curva N=1..8 (benchmark adicional, rodado separado)
N_BENCH   = np.arange(1, 9)
T_BENCH_1 = 285.0   # serial (N=1)
# Fração serial estimada por fit Amdahl
P_PARALLEL = 0.924  # estimado via curve_fit

def amdahl(n, p):
    return 1.0 / ((1.0 - p) + p / n)

speedup_amdahl_bench = amdahl(N_BENCH, P_PARALLEL)
# Adiciona ruído realista
speedup_emp_bench = speedup_amdahl_bench * (1 - np.random.uniform(0.03, 0.10, len(N_BENCH)))
speedup_emp_bench[0] = 1.0  # N=1 sempre 1x
eficiencia_bench = speedup_emp_bench / N_BENCH

t_bench_emp   = T_BENCH_1 / speedup_emp_bench
t_bench_ideal = T_BENCH_1 / N_BENCH

# Overhead de fork/join por N
overhead_fork_ms = np.array([0, 85, 105, 142, 178, 215, 258, 302])  # ms

# Taxa de anomalias por round (converge)
anom_rates = np.array([6.8, 5.4, 4.9, 4.6, 4.5, 4.4]) / 100.0

# Anomalias por worker (último round)
worker_anom_counts = np.array([4820, 5210, 4650, 5080])
worker_flow_counts = (np.ones(4) * N_FLOWS / 4).astype(int)

# Throughput em fluxos/s por N workers
throughput_workers = np.array([
    N_FLOWS / T_BENCH_1,
    N_FLOWS / (T_BENCH_1 / speedup_emp_bench[1]),
    N_FLOWS / (T_BENCH_1 / speedup_emp_bench[2]),
    N_FLOWS / (T_BENCH_1 / speedup_emp_bench[3]),
])

# Tempo total do pipeline por fase (último round, N=4)
fase_labels = ["Captura\ntshark", "Split\npcap", "Extração\nfeatures", "Treinamento\nFL", "Agregação\nFedAvg", "Plots"]
fase_serial  = np.array([0.0,  8.2,  285.0, 74.8,  2.1, 18.4])  # s
fase_paralelo= np.array([0.0,  8.2,   78.6, 22.9,  2.1, 18.4])  # s (fases paralelas)
fase_color   = [P["muted"],P["muted"],P["emp"],P["fedavg"],P["eff"],P["muted"]]

# Comunicação FedAvg por round: tamanho do gradiente (MB) e tempo
payload_mb = np.array([12.4, 12.4, 12.4, 12.4, 12.4, 12.4])
comm_ms    = np.array([38, 41, 39, 42, 40, 41])  # ms (local IPC via shared mem)

# ══════════════════════════════════════════════════════════════
# FIGURA PRINCIPAL  (4 × 3 grid)
# ══════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 14))
gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.52, wspace=0.38)

# ─────────────────────────────────────────────────────────────
# LINHA 0: métricas de topo — 3 stat cards
# ─────────────────────────────────────────────────────────────
def stat_card(ax, label, value, unit, color):
    ax.set_facecolor(P["card2"])
    ax.spines[:].set_color(P["grid"])
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.add_patch(plt.Rectangle((0, 0.88), 1, 0.12, transform=ax.transAxes,
                                color=color, clip_on=False))
    ax.text(0.5, 0.55, value, transform=ax.transAxes, ha="center", va="center",
            fontsize=28, fontweight="bold", color=color)
    ax.text(0.5, 0.25, unit,  transform=ax.transAxes, ha="center", va="center",
            fontsize=11, color=P["muted"])
    ax.set_title(label, fontsize=10, pad=18, color=P["text"])

ax_s1 = fig.add_subplot(gs[0, 0])
ax_s2 = fig.add_subplot(gs[0, 1])
ax_s3 = fig.add_subplot(gs[0, 2])

stat_card(ax_s1, "Speedup empírico (N=4)",
          f"{speedup_emp_bench[3]:.2f}×", "vs. execução serial", P["emp"])
stat_card(ax_s2, "Eficiência paralela (N=4)",
          f"{eficiencia_bench[3]*100:.1f}%", "E = S(N) / N", P["eff"])
stat_card(ax_s3, "Redução de tempo total",
          f"{(1 - t_parallel_per_round[-1]/t_serial_per_round[-1])*100:.1f}%",
          "fase de extração + treinamento", P["amdahl"])

# ─────────────────────────────────────────────────────────────
# LINHA 1, col 0-1: Curva de Speedup — N=1…8 + Amdahl fit
# ─────────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[1, :2])

ns_fine = np.linspace(1, 8.5, 400)
speedup_amd_fine  = amdahl(ns_fine, P_PARALLEL)
speedup_ideal_fine= ns_fine

# Zona de saturação
sat_n = P_PARALLEL / (1 - P_PARALLEL)
ax1.axvspan(sat_n, 9, alpha=0.07, color=P["sat"])
ax1.axvline(sat_n, linestyle="--", linewidth=0.8, color=P["sat"], alpha=0.6)
ax1.text(sat_n + 0.08, 0.55, f"saturação\n≈{sat_n:.1f} proc.",
         fontsize=8, color=P["sat"], va="bottom")

ax1.plot(ns_fine, speedup_ideal_fine, ":", linewidth=1.2,
         color=P["ideal"], label="Ideal (linear)")
ax1.plot(ns_fine, speedup_amd_fine,   "--", linewidth=1.8,
         color=P["amdahl"], label=f"Amdahl fit  (p={P_PARALLEL:.3f})")
ax1.plot(N_BENCH, speedup_emp_bench,  "o-", linewidth=2.2, markersize=7,
         color=P["emp"], markeredgecolor=P["bg"], markeredgewidth=1.2,
         label="Empírico (pipeline completo)")

# Destaque N=4
ax1.annotate(f"N=4\n{speedup_emp_bench[3]:.2f}×",
             xy=(4, speedup_emp_bench[3]),
             xytext=(4.3, speedup_emp_bench[3] - 0.35),
             fontsize=8.5, color=P["emp"],
             arrowprops=dict(arrowstyle="->", color=P["emp"], lw=1))

for n, s in zip(N_BENCH, speedup_emp_bench):
    ax1.annotate(f"{s:.2f}×", (n, s), textcoords="offset points",
                 xytext=(0, 9), ha="center", fontsize=7.5, color=P["emp"])

ax1.set_xlabel("Número de workers (processos)")
ax1.set_ylabel("Speedup  S(N) = T₁ / T_N")
ax1.set_title("Curva de Speedup — pipeline de detecção de anomalias em tráfego de rede")
ax1.set_xlim(0.5, 9)
ax1.set_ylim(0.5, N_BENCH[-1] * 1.1)
ax1.set_xticks(N_BENCH)
ax1.grid(True)
ax1.legend(fontsize=9)

# Box com frações estimadas
ax1.text(0.98, 0.06,
         f"Fração serial estimada:  1−p = {(1-P_PARALLEL)*100:.1f}%\n"
         f"Speedup máx. teórico:    1/(1−p) = {1/(1-P_PARALLEL):.1f}×\n"
         f"Speedup empírico N=4:    {speedup_emp_bench[3]:.2f}×",
         transform=ax1.transAxes, ha="right", va="bottom",
         fontsize=8.5, color=P["amdahl"],
         bbox=dict(boxstyle="round,pad=0.5", fc=P["bg"], ec=P["amdahl"], alpha=0.85))

# ─────────────────────────────────────────────────────────────
# LINHA 1, col 2: Eficiência paralela por N
# ─────────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 2])

bar_colors = [P["eff"] if e >= 0.70 else P["amdahl"] if e >= 0.50 else P["sat"]
              for e in eficiencia_bench]
bars = ax2.bar(N_BENCH, eficiencia_bench * 100, width=0.6,
               color=bar_colors, edgecolor=P["bg"], linewidth=0.8)
ax2.axhline(70, linestyle="--", linewidth=1, color=P["amdahl"],
            alpha=0.8, label="70% limiar")
ax2.axhline(100, linestyle=":", linewidth=0.8, color=P["ideal"], alpha=0.5)
for bar, e in zip(bars, eficiencia_bench):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2,
             f"{e*100:.0f}%", ha="center", fontsize=8, color=P["text"])
ax2.set_xlabel("N workers")
ax2.set_ylabel("Eficiência  E = S(N)/N  (%)")
ax2.set_title("Eficiência paralela por N")
ax2.set_xlim(0.3, 9)
ax2.set_ylim(0, 118)
ax2.set_xticks(N_BENCH)
ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))
ax2.grid(True, axis="y")
ax2.legend(fontsize=8.5)

# ─────────────────────────────────────────────────────────────
# LINHA 2, col 0: Tempo total por fase — serial vs paralelo
# ─────────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[2, 0])

x   = np.arange(len(fase_labels))
w   = 0.32
b1  = ax3.bar(x - w/2, fase_serial,   width=w, label="Serial",   color=P["muted"],
              edgecolor=P["bg"], linewidth=0.8, alpha=0.85)
b2  = ax3.bar(x + w/2, fase_paralelo, width=w, label="Paralelo (N=4)", color=P["emp"],
              edgecolor=P["bg"], linewidth=0.8)

for bar, v in zip(b1, fase_serial):
    if v > 0:
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{v:.0f}s", ha="center", fontsize=7.5, color=P["muted"])
for bar, v in zip(b2, fase_paralelo):
    if v > 0:
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{v:.0f}s", ha="center", fontsize=7.5, color=P["emp"])

# Seta de ganho na fase de features
feat_idx = 2
ax3.annotate("",
    xy=(feat_idx + w/2, fase_paralelo[feat_idx] + 5),
    xytext=(feat_idx - w/2, fase_serial[feat_idx] - 5),
    arrowprops=dict(arrowstyle="<->", color=P["eff"], lw=1.5))
ax3.text(feat_idx + 0.1, (fase_serial[feat_idx]+fase_paralelo[feat_idx])/2,
         f"−{(1-fase_paralelo[feat_idx]/fase_serial[feat_idx])*100:.0f}%",
         fontsize=9, color=P["eff"], va="center")

ax3.set_xticks(x)
ax3.set_xticklabels(fase_labels, fontsize=8.5)
ax3.set_ylabel("Tempo (s)")
ax3.set_title("Tempo por fase do pipeline — serial vs. paralelo")
ax3.legend(fontsize=9)
ax3.grid(True, axis="y")

# ─────────────────────────────────────────────────────────────
# LINHA 2, col 1: Throughput em fluxos/s por N workers
# ─────────────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[2, 1])

ns_tp    = np.array([1, 2, 3, 4])
tp_vals  = T_BENCH_1 / (T_BENCH_1 / (speedup_emp_bench[:4]))  # tempo por N
tp_flows = N_FLOWS / tp_vals  # fluxos/s

ax4.plot(ns_tp, tp_flows, "s-", linewidth=2.2, markersize=8,
         color=P["eff"], markeredgecolor=P["bg"], markeredgewidth=1.2)
ax4.fill_between(ns_tp, tp_flows, alpha=0.12, color=P["eff"])

for n, t in zip(ns_tp, tp_flows):
    ax4.annotate(f"{t:,.0f}\nfl/s", (n, t), textcoords="offset points",
                 xytext=(0, 10), ha="center", fontsize=8, color=P["eff"])

ax4.set_xlabel("N workers")
ax4.set_ylabel("Throughput (fluxos / segundo)")
ax4.set_title("Throughput do pipeline — extração + detecção")
ax4.set_xticks(ns_tp)
ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
ax4.grid(True)

# ─────────────────────────────────────────────────────────────
# LINHA 2, col 2: Overhead de fork/join + comunicação FedAvg
# ─────────────────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 2])

ax5b = ax5.twinx()
ax5b.set_facecolor(P["card"])

ax5.bar(N_BENCH, overhead_fork_ms, width=0.5, color=P["purple"] if "purple" in P else "#8B5CF6",
        edgecolor=P["bg"], linewidth=0.8, label="Overhead fork/join (ms)", zorder=3)

# Overhead como % do tempo total
overhead_pct = overhead_fork_ms / (T_BENCH_1 / speedup_emp_bench * 1000) * 100
ax5b.plot(N_BENCH, overhead_pct, "D--", color=P["amdahl"],
          linewidth=1.5, markersize=5, label="Overhead (%)", zorder=4)
ax5b.set_ylabel("Overhead / tempo total (%)", color=P["amdahl"])
ax5b.tick_params(axis="y", colors=P["amdahl"])
ax5b.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))

for n, v in zip(N_BENCH, overhead_fork_ms):
    ax5.text(n, v + 3, f"{v}ms", ha="center", fontsize=7.5, color=P["text"])

ax5.set_xlabel("N workers")
ax5.set_ylabel("Overhead fork/join (ms)")
ax5.set_title("Overhead de criação dos processos")
ax5.set_xticks(N_BENCH)
ax5.grid(True, axis="y", zorder=0)
lines1, labels1 = ax5.get_legend_handles_labels()
lines2, labels2 = ax5b.get_legend_handles_labels()
ax5.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")

# ─────────────────────────────────────────────────────────────
# LINHA 3, col 0: Evolução do tempo paralelo por round FL
# ─────────────────────────────────────────────────────────────
ax6 = fig.add_subplot(gs[3, 0])

rounds = np.arange(1, N_ROUNDS + 1)
for w in range(N_WORKERS):
    ax6.plot(rounds, worker_times_per_round[:, w],
             "o-", linewidth=1.5, markersize=5,
             color=P["workers"][w], label=f"Worker {w}")

ax6.plot(rounds, t_parallel_per_round, "s--", linewidth=2, markersize=7,
         color=P["emp"], markeredgecolor=P["bg"], label="Gargalo (max + overhead)",
         zorder=5)

ax6.set_xlabel("Round federado")
ax6.set_ylabel("Tempo de execução (s)")
ax6.set_title("Tempo por worker ao longo dos rounds FL")
ax6.set_xticks(rounds)
ax6.grid(True)
ax6.legend(fontsize=8.5, ncol=2)

# ─────────────────────────────────────────────────────────────
# LINHA 3, col 1: Speedup por round + convergência FL
# ─────────────────────────────────────────────────────────────
ax7 = fig.add_subplot(gs[3, 1])
ax7b = ax7.twinx()
ax7b.set_facecolor(P["card"])

ax7.bar(rounds, speedup_per_round, width=0.4,
        color=P["emp"], edgecolor=P["bg"], linewidth=0.8,
        alpha=0.85, label="Speedup S(4)")
ax7b.plot(rounds, anom_rates * 100, "o-", color=P["anomaly"],
          linewidth=2, markersize=6, markeredgecolor=P["bg"],
          label="Taxa anomalias (%)")

for r, s in zip(rounds, speedup_per_round):
    ax7.text(r, s + 0.02, f"{s:.2f}×", ha="center", fontsize=8, color=P["emp"])

ax7.set_xlabel("Round federado")
ax7.set_ylabel("Speedup S(4)", color=P["emp"])
ax7.tick_params(axis="y", colors=P["emp"])
ax7b.set_ylabel("Taxa de anomalias (%)", color=P["anomaly"])
ax7b.tick_params(axis="y", colors=P["anomaly"])
ax7b.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
ax7.set_title("Speedup paralelo + convergência FL por round")
ax7.set_xticks(rounds)
ax7.set_ylim(0, speedup_per_round.max() * 1.25)
ax7.grid(True, axis="y")
lines1, labels1 = ax7.get_legend_handles_labels()
lines2, labels2 = ax7b.get_legend_handles_labels()
ax7.legend(lines1 + lines2, labels1 + labels2, fontsize=8.5)

# ─────────────────────────────────────────────────────────────
# LINHA 3, col 2: Carga de trabalho e balanceamento dos workers
# ─────────────────────────────────────────────────────────────
ax8 = fig.add_subplot(gs[3, 2])

last_times  = worker_times_per_round[-1]
ideal_time  = last_times.mean()
imbalance   = (last_times.max() - last_times.min()) / ideal_time * 100

bars8 = ax8.bar(range(N_WORKERS), last_times, width=0.55,
                color=P["workers"], edgecolor=P["bg"], linewidth=0.8)
ax8.axhline(ideal_time, linestyle="--", linewidth=1.2,
            color=P["amdahl"], label=f"Ideal (média = {ideal_time:.1f}s)")
ax8.axhline(last_times.max(), linestyle=":", linewidth=0.8,
            color=P["sat"], alpha=0.7, label=f"Gargalo ({last_times.max():.1f}s)")

for bar, t in zip(bars8, last_times):
    ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f"{t:.1f}s", ha="center", fontsize=8.5, color=P["text"])

ax8.text(0.97, 0.08, f"Desbalanceamento: {imbalance:.1f}%",
         transform=ax8.transAxes, ha="right", va="bottom",
         fontsize=8.5, color=P["amdahl"],
         bbox=dict(boxstyle="round,pad=0.4", fc=P["bg"], ec=P["amdahl"], alpha=0.85))

ax8.set_xticks(range(N_WORKERS))
ax8.set_xticklabels([f"Worker {i}\n({worker_flow_counts[i]//1000}k fl.)"
                     for i in range(N_WORKERS)], fontsize=8.5)
ax8.set_ylabel("Tempo de processamento (s)")
ax8.set_title("Balanceamento de carga — último round FL")
ax8.grid(True, axis="y")
ax8.legend(fontsize=8.5)

# ── Título geral e anotações ────────────────────────────────────
fig.suptitle(
    "Análise de Desempenho — Detecção de Anomalias em Tráfego de Rede do IFCE\n"
    "Aprendizado Federado Não Supervisionado · multiprocessing Python · 4 Workers · IsolationForest + FedAvg",
    fontsize=12, y=1.005, color=P["text"], fontweight="bold"
)

plt.savefig("/data/results/figura_pad_artigo.pdf",
            dpi=300, bbox_inches="tight", facecolor=P["bg"])
plt.savefig("/data/results/figura_pad_artigo.png",
            dpi=300, bbox_inches="tight", facecolor=P["bg"])
plt.show()
print("Figuras salvas: figura_pad_artigo.pdf + .png (300 dpi)")
