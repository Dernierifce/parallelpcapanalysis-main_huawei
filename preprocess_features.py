"""
preprocess_features.py
Extração de features com salvamento em cache (pickle).
Executado UMA VEZ antes dos benchmarks de classificação.

Uso:
    python preprocess_features.py \
        --shards ./data/pcaps/*.pcapng \
        --outdir ./data/results \
        --cache-file features_cache.pkl

Benefício:
    - Extração de features é operação I/O (pyshark) custosa
    - Executar UMA VEZ e reutilizar para múltiplos benchmarks
    - Mensurações focam APENAS na classificação (train + infer)
"""

import argparse
import glob
import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from feature_extractor import FEATURE_COLS, extract_flows

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SHARDS = sorted(glob.glob(str(BASE_DIR / "data" / "pcaps" / "*.pcapng")))
DEFAULT_OUTDIR = str(BASE_DIR / "data" / "results")
DEFAULT_CACHE_FILE = "features_cache.pkl"


def main():
    parser = argparse.ArgumentParser(
        description="Pré-processar e cachear features de fluxo"
    )
    parser.add_argument("--shards", nargs="+", default=DEFAULT_SHARDS)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument("--cache-file", default=DEFAULT_CACHE_FILE)
    args = parser.parse_args()

    if not args.shards:
        raise ValueError("Nenhum shard informado/encontrado. Verifique --shards.")

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  PRÉ-PROCESSAMENTO: Extração e Cache de Features")
    print(f"  Shards: {len(args.shards)}")
    print("=" * 70)

    t_total_start = time.perf_counter()
    all_dfs = []
    shard_stats = []

    # ── Extração de features por shard ────────────────────────────────
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
        print(
            f"  [EXTRACT] {Path(shard).name:30s} | "
            f"flows={len(df):,} | t={elapsed:.1f}s"
        )

    extraction_time = time.perf_counter() - t_total_start

    # ── Consolidação e normalização ───────────────────────────────────
    merged = np.vstack([df[FEATURE_COLS].fillna(0).values for df in all_dfs])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(merged)

    # ── Salvamento em cache ───────────────────────────────────────────
    cache_path = Path(args.outdir) / args.cache_file
    cache_data = {
        "X_scaled": X_scaled,
        "scaler": scaler,
        "shard_stats": shard_stats,
        "n_flows": int(X_scaled.shape[0]),
        "n_features": int(X_scaled.shape[1]),
        "total_extract_s": extraction_time,
    }

    with open(cache_path, "wb") as f:
        pickle.dump(cache_data, f)

    print("\n" + "=" * 70)
    print(f"  ✓ Cache salvo: {cache_path}")
    print(f"  Fluxos totais: {X_scaled.shape[0]:,}")
    print(f"  Features: {X_scaled.shape[1]}")
    print(f"  Tempo extração total: {extraction_time:.1f}s")
    print("=" * 70)
    print("\nPróximo passo: usar --cache-file em gpu_train.py e federated_train.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
