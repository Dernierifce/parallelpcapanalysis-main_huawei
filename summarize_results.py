"""Gera um resumo consolidado de múltiplos arquivos de resultado.

Uso:
    python summarize_results.py --results cpu.pkl gpu.pkl fed.pkl --names CPU GPU FED --outdir ./data/results

O script escreve no mesmo `log.txt` padrão (via `setup_run_logging`) e usa `emit_report` para
gravar uma seção consolidada com tempos, volumes, rounds/épocas e comparação de métodos.
"""

from pathlib import Path
import argparse
import pickle
from typing import List

from log_utils import setup_run_logging, emit_report


def load_result(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def collect_metrics(data: dict) -> dict:
    times = data.get("times", {})
    return {
        "method": data.get("method", "n/a"),
        "backend": data.get("backend", data.get("backend", "n/a")),
        "n_flows": int(data.get("n_flows", 0)),
        "n_features": int(data.get("n_features", 0)) if data.get("n_features") is not None else None,
        "n_anomalies": int(data.get("n_anomalies", 0)),
        "anom_rate": float(data.get("anom_rate", 0.0)),
        "extract_s": float(times.get("extract_s", 0.0)),
        "train_s": float(times.get("train_s", 0.0)),
        "infer_s": float(times.get("infer_s", 0.0)),
        "classification_s": float(times.get("classification_s", 0.0)),
        "total_s": float(times.get("total_s", 0.0)),
        "n_rounds": int(data.get("n_rounds", data.get("args", {}).get("rounds", 0))) if data.get("n_rounds") or data.get("args") else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Resumo consolidado de resultados de execução")
    parser.add_argument("--results", nargs="+", required=True, help="Arquivos .pkl de resultados")
    parser.add_argument("--names", nargs="*", help="Nomes curtos para os resultados (opcional)")
    parser.add_argument("--outdir", default=str(Path(__file__).resolve().parent / "data" / "results"))
    parser.add_argument("--log-file", default=None, help="Arquivo de log opcional")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = setup_run_logging(outdir, "summarize_results", args.log_file)

    paths: List[Path] = [Path(p) for p in args.results]
    names = args.names or [p.stem for p in paths]

    entries = {}
    for name, path in zip(names, paths):
        if not path.exists():
            print(f"Aviso: arquivo não encontrado: {path}")
            continue
        try:
            data = load_result(path)
            entries[name] = collect_metrics(data)
        except Exception as e:
            print(f"Erro ao carregar {path}: {e}")

    if not entries:
        print("Nenhum resultado válido para resumir.")
        return

    # Calcular comparações simples
    # Ordenar por classification_s se disponível, senão total_s
    def time_key(m):
        return m.get("classification_s") or m.get("total_s") or float("inf")

    sorted_items = sorted(entries.items(), key=lambda kv: time_key(kv[1]))

    summary_sections = {
        "Resumo Consolidado": {
            name: {
                "method": metrics["method"],
                "backend": metrics["backend"],
                "fluxos": metrics["n_flows"],
                "anomalias": metrics["n_anomalies"],
                "anom_rate_pct": f"{metrics['anom_rate'] * 100:.2f}%",
                "classification_s": round(metrics["classification_s"], 3),
                "total_s": round(metrics["total_s"], 3),
                "n_rounds": metrics.get("n_rounds", 0),
            }
            for name, metrics in entries.items()
        }
    }

    # Speedups relativos ao primeiro (mais rápido)
    fastest = time_key(sorted_items[0][1])
    comparisons = []
    for name, metrics in sorted_items:
        t = time_key(metrics)
        speed = (fastest / t) if t > 0 else 0.0
        comparisons.append(f"{name}: tempo={t:.3f}s | speedup_relativo={speed:.2f}x")

    summary_sections["Comparações"] = comparisons

    emit_report("Resumo Final Consolidado", summary_sections)


if __name__ == "__main__":
    main()
