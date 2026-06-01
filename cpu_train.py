"""
cpu_train.py
Benchmark em CPU usando Isolation Forest.

Uso recomendado:
    python cpu_train.py --cache-file ./data/results/features_cache.pkl --outdir ./data/results --outfile cpu_results.pkl

Modo legado:
    python cpu_train.py --shards ./data/pcaps/*.pcapng --outdir ./data/results --outfile cpu_results.pkl
"""

import argparse
import glob
import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from feature_extractor import FEATURE_COLS, extract_flows
from log_utils import emit_report, setup_run_logging


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SHARDS = sorted(glob.glob(str(BASE_DIR / "data" / "pcaps" / "*.pcapng")))
DEFAULT_OUTDIR = str(BASE_DIR / "data" / "results")
DEFAULT_OUTFILE = "cpu_results.pkl"


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


def main():
    parser = argparse.ArgumentParser(description="Benchmark CPU com Isolation Forest")
    parser.add_argument("--shards", nargs="+", default=DEFAULT_SHARDS)
    parser.add_argument("--cache-file", default=None, help="Cache gerado por preprocess_features.py")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument("--outfile", default=DEFAULT_OUTFILE)
    parser.add_argument("--log-file", default=None, help="Arquivo de log opcional")
    parser.add_argument("--estimators", type=int, default=200)
    parser.add_argument("--contamination", type=float, default=0.05)
    args = parser.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    log_path = setup_run_logging(args.outdir, "cpu_train", args.log_file)

    print("=" * 70)
    print("  Benchmark CPU — Isolation Forest")
    print(f"  Mode: {'CACHE' if args.cache_file else 'FULL'}")
    print(f"  N estimators: {args.estimators}")
    print(f"  Log: {log_path}")
    print("=" * 70)

    X_scaled, extraction_time, shard_stats, mode = _load_features(args.cache_file, args.shards)

    classification_start = time.perf_counter()
    model = IsolationForest(
        n_estimators=args.estimators,
        contamination=args.contamination,
        random_state=42,
        n_jobs=-1,
    )

    t_train_start = time.perf_counter()
    model.fit(X_scaled)
    train_time = time.perf_counter() - t_train_start

    t_infer_start = time.perf_counter()
    scores = model.decision_function(X_scaled)
    labels = model.predict(X_scaled)
    infer_time = time.perf_counter() - t_infer_start
    classification_time = time.perf_counter() - classification_start
    total_time = extraction_time + classification_time

    n_anom = int((labels == -1).sum())
    out = {
        "method": "isolation_forest",
        "backend": "cpu-sklearn",
        "mode": mode,
        "args": vars(args),
        "n_flows": int(X_scaled.shape[0]),
        "n_features": int(X_scaled.shape[1]),
        "n_anomalies": n_anom,
        "anom_rate": float(n_anom / X_scaled.shape[0]),
        "times": {
            "extract_s": extraction_time,
            "train_s": train_time,
            "infer_s": infer_time,
            "classification_s": classification_time,
            "total_s": total_time,
        },
        "shards": shard_stats,
        "scores": scores,
        "labels": labels,
    }

    out_path = Path(args.outdir) / args.outfile
    with open(out_path, "wb") as f:
        pickle.dump(out, f)

    emit_report(
        "Relatório detalhado — cpu_train",
        {
            "Configuração": {
                "metodo": out["method"],
                "backend": out["backend"],
                "mode": mode,
                "estimators": args.estimators,
                "contamination": args.contamination,
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
                "arquivo_saida": out_path,
                "classificacao_metrica_principal": f"{classification_time:.1f}s (train + infer)",
            },
        },
    )


if __name__ == "__main__":
    main()