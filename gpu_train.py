"""
gpu_train.py
Benchmark de anomalias com aceleração por GPU (cuML) para comparação
com o pipeline federado em CPU.

Modos:
    1. Com CACHE (recomendado para focar em classificação):
       python gpu_train.py \
           --cache-file /data/results/features_cache.pkl \
           --outdir /data/results \
           --outfile gpu_results.pkl
    
    2. Sem cache (modo legado — inclui extração):
       python gpu_train.py \
           --shards /data/pcaps/*.pcapng \
           --outdir /data/results \
           --outfile gpu_results.pkl

Nota: modo com --cache-file isola medição APENAS de classificação (train + infer)
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

DEFAULT_SHARDS = sorted(glob.glob("/data/pcaps/*.pcapng"))
DEFAULT_OUTDIR = "/data/results"
DEFAULT_OUTFILE = "gpu_results.pkl"


def _train_gpu_cuml(X_scaled: np.ndarray, n_estimators: int, contamination: float):
    import cupy as cp
    from cuml.ensemble import IsolationForest as CuIsolationForest

    X_gpu = cp.asarray(X_scaled.astype(np.float32, copy=False))
    model = CuIsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=42,
    )

    t0 = time.perf_counter()
    model.fit(X_gpu)
    train_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    scores = cp.asnumpy(model.decision_function(X_gpu))
    labels = cp.asnumpy(model.predict(X_gpu))
    infer_time = time.perf_counter() - t1

    return scores, labels, train_time, infer_time


def _train_cpu_fallback(X_scaled: np.ndarray, n_estimators: int, contamination: float):
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )

    t0 = time.perf_counter()
    model.fit(X_scaled)
    train_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    scores = model.decision_function(X_scaled)
    labels = model.predict(X_scaled)
    infer_time = time.perf_counter() - t1

    return scores, labels, train_time, infer_time


def main():
    parser = argparse.ArgumentParser(description="Benchmark GPU para comparação com CPU federada")
    parser.add_argument("--shards", nargs="+", default=DEFAULT_SHARDS)
    parser.add_argument("--cache-file", default=None,
                        help="Arquivo .pkl com features pré-extraídas (modo recomendado)")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument("--outfile", default=DEFAULT_OUTFILE)
    parser.add_argument("--estimators", type=int, default=400)
    parser.add_argument("--contamination", type=float, default=0.05)
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    args = parser.parse_args()

    if not args.shards and not args.cache_file:
        raise ValueError("Forneça --cache-file (recomendado) ou --shards (modo legado).")

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    # ── CARREGAR FEATURES: Do cache OU extrair ──────────────────────────
    if args.cache_file:
        print("=" * 70)
        print("  Benchmark GPU (cuML) — Modo CACHE")
        print(f"  Carregando cache: {args.cache_file}")
        print(f"  N estimadores: {args.estimators}")
        print("=" * 70)

        with open(args.cache_file, "rb") as f:
            cache = pickle.load(f)

        X_scaled = cache["X_scaled"]
        scaler = cache["scaler"]
        shard_stats = cache.get("shard_stats", [])
        extraction_time = 0.0  # Não contar extração no modo cache!
        n_flows_total = cache["n_flows"]
        mode = "cache"
    else:
        print("=" * 70)
        print("  Benchmark GPU (cuML) — Modo COMPLETO (com extração)")
        print(f"  Shards: {len(args.shards)}")
        print(f"  N estimadores: {args.estimators}")
        print("=" * 70)

        extraction_start = time.perf_counter()
        all_dfs = []
        shard_stats = []

        for shard in sorted(args.shards):
            t0 = time.perf_counter()
            df = extract_flows(shard, anonymize=True)
            elapsed = time.perf_counter() - t0

            all_dfs.append(df)
            shard_stats.append(
                {
                    "shard_path": str(shard),
                    "n_flows": int(len(df)),
                    "extract_s": elapsed,
                }
            )
            print(f"  [EXTRACT] {shard} | flows={len(df):,} | t={elapsed:.1f}s")

        extraction_time = time.perf_counter() - extraction_start
        merged = np.vstack([df[FEATURE_COLS].fillna(0).values for df in all_dfs])

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(merged)
        n_flows_total = X_scaled.shape[0]
        mode = "full"

    print(f"\n  Modo: {mode.upper()}")
    print(f"  Fluxos carregados: {X_scaled.shape[0]:,}")
    print(f"  Features: {X_scaled.shape[1]}")

    print("  ─" * 35)
    print("  MEDINDO CLASSIFICAÇÃO...")
    print("  ─" * 35 + "\n")

    # ── CLASSIFICAÇÃO: Treino + Inferência ──────────────────────────────
    classification_start = time.perf_counter()

    backend = "gpu-cuml"
    try:
        scores, labels, train_time, infer_time = _train_gpu_cuml(
            X_scaled,
            n_estimators=args.estimators,
            contamination=args.contamination,
        )
    except Exception as exc:
        if not args.allow_cpu_fallback:
            raise RuntimeError(
                "Falha ao usar cuML no container. Instale dependências de GPU ou rode com --allow-cpu-fallback."
            ) from exc
        backend = "cpu-fallback"
        print(f"  [WARN] GPU indisponível, usando fallback CPU: {exc}")
        scores, labels, train_time, infer_time = _train_cpu_fallback(
            X_scaled,
            n_estimators=args.estimators,
            contamination=args.contamination,
        )

    classification_time = time.perf_counter() - classification_start
    total_time = extraction_time + classification_time
    n_anom = int((labels == -1).sum())

    out_path = Path(args.outdir) / args.outfile
    with open(out_path, "wb") as f:
        pickle.dump(
            {
                "backend": backend,
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
            },
            f,
        )

    print("\n" + "=" * 70)
    print(f"  Backend: {backend}")
    print(f"  Modo: {mode.upper()}")
    print(f"  Fluxos: {X_scaled.shape[0]:,}")
    print(f"  Anomalias: {n_anom:,} ({n_anom / X_scaled.shape[0] * 100:.2f}%)")
    print("  " + "─" * 68)
    if extraction_time > 0:
        print(f"  Extração: {extraction_time:.1f}s (não medido em modo cache)")
    print(f"  Treino: {train_time:.1f}s")
    print(f"  Inferência: {infer_time:.1f}s")
    print(f"  ► Classificação (train+infer): {classification_time:.1f}s ◄ [MÉTRICA]")
    print(f"  Total (incl. extração): {total_time:.1f}s")
    print(f"  Resultado: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
