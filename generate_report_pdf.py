"""Gera um relatório PDF multipágina com gráficos comparativos e resumo.

1) Garante que `plots_cpu_gpu_compare.py` gere o PNG do comparativo (se necessário).
2) Monta um PDF com a figura comparativa e uma página adicional com o resumo de métricas.

Uso:
    python generate_report_pdf.py --cpu ./data/results/cpu_results.pkl --gpu ./data/results/gpu_results.pkl --outdir ./data/results --basename cpu_gpu_report

"""

from pathlib import Path
import argparse
import subprocess
import sys
import pickle

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from log_utils import setup_run_logging, emit_report


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def ensure_plots(cpu_path: Path, gpu_path: Path, outdir: Path, basename: str):
    png_path = outdir / f"{basename}.png"
    if png_path.exists():
        return png_path

    # Call plots_cpu_gpu_compare.py to generate png/pdf
    cmd = [sys.executable, "plots_cpu_gpu_compare.py", "--cpu-results", str(cpu_path), "--gpu-results", str(gpu_path), "--outdir", str(outdir), "--basename", basename]
    subprocess.run(cmd, check=True)
    return png_path


def make_summary_page(pdf: PdfPages, cpu: dict, gpu: dict, title: str = "Resumo Consolidado"):
    fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111)
    ax.axis("off")

    lines = [title, "", "CPU:"]
    cpu_times = cpu.get("times", {})
    lines += [f"  Method: {cpu.get('method', 'n/a')}", f"  Backend: {cpu.get('backend', 'n/a')}", f"  Fluxos: {cpu.get('n_flows', 0):,}", f"  Anomalias: {cpu.get('n_anomalies', 0):,}", f"  Classification (s): {cpu_times.get('classification_s', 0.0):.3f}", f"  Total (s): {cpu_times.get('total_s', 0.0):.3f}"]

    lines += ["", "GPU:"]
    gpu_times = gpu.get("times", {})
    lines += [f"  Method: {gpu.get('method', 'n/a')}", f"  Backend: {gpu.get('backend', 'n/a')}", f"  Fluxos: {gpu.get('n_flows', 0):,}", f"  Anomalias: {gpu.get('n_anomalies', 0):,}", f"  Classification (s): {gpu_times.get('classification_s', 0.0):.3f}", f"  Total (s): {gpu_times.get('total_s', 0.0):.3f}"]

    # Add comparisons
    cpu_time = cpu_times.get('classification_s') or cpu_times.get('total_s') or float('inf')
    gpu_time = gpu_times.get('classification_s') or gpu_times.get('total_s') or float('inf')
    if cpu_time == float('inf') or gpu_time == float('inf'):
        comp = "Dados incompletos para comparação objetiva"
    else:
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0.0
        comp = f"Speedup (CPU / GPU) by chosen metric: {speedup:.2f}x"

    lines += ["", "Comparação:", f"  {comp}"]

    text = "\n".join(lines)
    ax.text(0.02, 0.98, text, va='top', ha='left', family='monospace', fontsize=10)

    pdf.savefig(fig)
    plt.close(fig)


def assemble_report(cpu_path: Path, gpu_path: Path, outdir: Path, basename: str, pdf_file: Path):
    cpu = load_pickle(cpu_path)
    gpu = load_pickle(gpu_path)

    png = ensure_plots(cpu_path, gpu_path, outdir, basename)

    with PdfPages(pdf_file) as pdf:
        # First page: the generated comparison figure (png)
        fig_img = plt.figure(figsize=(11.69, 8.27))  # landscape
        ax_img = fig_img.add_subplot(111)
        ax_img.axis('off')
        img = plt.imread(png)
        ax_img.imshow(img)
        pdf.savefig(fig_img)
        plt.close(fig_img)

        # Second page: textual summary
        make_summary_page(pdf, cpu, gpu, title=f"Relatório: {basename}")


def main():
    parser = argparse.ArgumentParser(description="Gerar PDF com gráficos comparativos e resumo")
    parser.add_argument("--cpu", required=True, help="Arquivo .pkl CPU")
    parser.add_argument("--gpu", required=True, help="Arquivo .pkl GPU")
    parser.add_argument("--outdir", default=str(Path(__file__).resolve().parent / 'data' / 'results'))
    parser.add_argument("--basename", default="cpu_gpu_comparison")
    parser.add_argument("--pdf-file", default=None, help="Nome do PDF de saída (opcional)")
    parser.add_argument("--log-file", default=None, help="Arquivo de log opcional")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    log_path = setup_run_logging(outdir, "generate_report_pdf", args.log_file)

    cpu_path = Path(args.cpu)
    gpu_path = Path(args.gpu)

    pdf_file = Path(args.pdf_file) if args.pdf_file else outdir / f"{args.basename}_report.pdf"

    assemble_report(cpu_path, gpu_path, outdir, args.basename, pdf_file)

    emit_report("Relatório PDF gerado", {"pdf": str(pdf_file), "png_used": str(outdir / f"{args.basename}.png")})


if __name__ == '__main__':
    main()
