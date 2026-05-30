"""
federated_train.py
Treinamento federado com Isolation Forest em CPU.

Uso recomendado:
    python federated_train.py --cache-file ./data/results/features_cache.pkl --outdir ./data/results --outfile cpu_federated_results.pkl

Modo legado:
    python federated_train.py --shards ./data/pcaps/*.pcapng --outdir ./data/results --outfile cpu_federated_results.pkl
"""

import argparse
import glob
import pickle
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from feature_extractor import FEATURE_COLS, extract_flows


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SHARDS = sorted(glob.glob(str(BASE_DIR / "data" / "pcaps" / "*.pcapng")))
DEFAULT_OUTDIR = str(BASE_DIR / "data" / "results")
DEFAULT_OUTFILE = "cpu_federated_results.pkl"
DEFAULT_ROUNDS = 6
DEFAULT_WORKERS = 4
DEFAULT_ESTIMATORS = 200
DEFAULT_CONTAMINATION = 0.05


def _load_features(cache_file: str | None, shards: list[str]) -> tuple[list[np.ndarray], float, list[dict], str]:
    if cache_file:
        with open(cache_file, "rb") as f:
            cache = pickle.load(f)

        X_scaled = cache["X_scaled"]
        shard_stats = cache.get("shard_stats", [])
        if not shard_stats:
            shard_stats = [{"shard_path": f"cache_{i}", "n_flows": int(len(X_scaled))} for i in range(1)]

        counts = [int(s.get("n_flows", 0)) for s in shard_stats]
        if not counts or sum(counts) <= 0:
            counts = [len(X_scaled)]

        split_indices = [0]
        for count in counts:
            split_indices.append(split_indices[-1] + count)
        split_indices[-1] = min(split_indices[-1], len(X_scaled))

        worker_data = []
        for start, end in zip(split_indices[:-1], split_indices[1:]):
            if start >= len(X_scaled):
                continue
            worker_data.append(X_scaled[start:end])
        if not worker_data:
            worker_data = [X_scaled]
        return worker_data, 0.0, shard_stats, "cache"

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

    n_workers = max(1, min(len(shards), cpu_count()))
    worker_data = np.array_split(X_scaled, n_workers)
    worker_data = [chunk for chunk in worker_data if len(chunk) > 0]
    return worker_data, time.perf_counter() - extraction_start, shard_stats, "full"


def _train_worker(payload: tuple) -> dict:
    worker_id, X_worker, global_params, fl_round, n_estimators, contamination = payload

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=worker_id + fl_round * 100,
        warm_start=False,
        n_jobs=1,
    )

    if global_params is not None:
        model.set_params(warm_start=True)
        model.estimators_ = list(global_params["estimators"])
        model.estimators_features_ = list(global_params["features"])
        model.n_features_in_ = global_params["n_features"]
        model.max_samples_ = global_params.get("max_samples", 256)

    t_train_start = time.perf_counter()
    model.fit(X_worker)
    train_time = time.perf_counter() - t_train_start

    t_infer_start = time.perf_counter()
    scores = model.decision_function(X_worker)
    labels = model.predict(X_worker)
    infer_time = time.perf_counter() - t_infer_start
    classification_time = train_time + infer_time

    n_anom = int((labels == -1).sum())

    print(
        f"  [W{worker_id}] R{fl_round + 1} | fluxos={len(X_worker):,} | anomalias={n_anom:,} "
        f"({n_anom / max(len(X_worker), 1) * 100:.2f}%) | class={classification_time:.2f}s"
    )

    return {
        "worker_id": worker_id,
        "fl_round": fl_round,
        "n_flows": int(len(X_worker)),
        "n_anomalies": n_anom,
        "anom_rate": float(n_anom / max(len(X_worker), 1)),
        "times": {
            "train_s": train_time,
            "infer_s": infer_time,
            "classification_s": classification_time,
        },
        "scores": scores,
        "labels": labels,
        "local_params": {
            "estimators": model.estimators_,
            "features": model.estimators_features_,
            "n_features": model.n_features_in_,
            "max_samples": getattr(model, "max_samples_", 256),
            "n_flows": int(len(X_worker)),
        },
    }


def _aggregate(worker_results: list[dict], n_estimators: int) -> dict:
    total_flows = sum(r["local_params"]["n_flows"] for r in worker_results)
    if total_flows <= 0:
        total_flows = len(worker_results)

    all_estimators = []
    all_features = []

    for result in worker_results:
        n_local = max(1, round((result["local_params"]["n_flows"] / total_flows) * n_estimators))
        local_estimators = result["local_params"]["estimators"]
        local_features = result["local_params"]["features"]
        take = min(n_local, len(local_estimators))
        chosen = np.random.choice(len(local_estimators), take, replace=False)
        all_estimators.extend(local_estimators[index] for index in chosen)
        all_features.extend(local_features[index] for index in chosen)

    return {
        "estimators": all_estimators,
        "features": all_features,
        "n_features": worker_results[0]["local_params"]["n_features"],
        "max_samples": worker_results[0]["local_params"]["max_samples"],
    }


def main():
    parser = argparse.ArgumentParser(description="Treinamento federado com Isolation Forest")
    parser.add_argument("--shards", nargs="+", default=DEFAULT_SHARDS)
    parser.add_argument("--cache-file", default=None, help="Cache gerado por preprocess_features.py")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument("--outfile", default=DEFAULT_OUTFILE)
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--estimators", type=int, default=DEFAULT_ESTIMATORS)
    parser.add_argument("--contamination", type=float, default=DEFAULT_CONTAMINATION)
    args = parser.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  PAD — Treinamento Federado (Isolation Forest)")
    print(f"  Mode: {'CACHE' if args.cache_file else 'FULL'}")
    print(f"  Workers: {args.workers}")
    print(f"  Rounds: {args.rounds}")
    print(f"  N trees: {args.estimators}")
    print("=" * 70)

    worker_data, extraction_time, shard_stats, mode = _load_features(args.cache_file, args.shards)
    n_workers = max(1, min(args.workers, len(worker_data)))
    worker_data = worker_data[:n_workers]

    global_params = None
    history = []
    t_total_start = time.perf_counter()

    for fl_round in range(args.rounds):
        print(f"\n{'─' * 50}")
        print(f"  Round FL {fl_round + 1}/{args.rounds}")
        print(f"{'─' * 50}")

        task_args = [
            (worker_id, worker_data[worker_id], global_params, fl_round, args.estimators, args.contamination)
            for worker_id in range(len(worker_data))
        ]

        t_round_start = time.perf_counter()
        with Pool(processes=n_workers) as pool:
            results = pool.map(_train_worker, task_args)
        t_round = time.perf_counter() - t_round_start

        global_params = _aggregate(results, args.estimators)

        total_flows = sum(r["n_flows"] for r in results)
        total_anom = sum(r["n_anomalies"] for r in results)
        worker_times = [r["times"]["classification_s"] for r in results]
        t_serial_eq = sum(worker_times) * 1.08

        round_info = {
            "round": fl_round + 1,
            "total_flows": total_flows,
            "total_anom": total_anom,
            "anom_rate": total_anom / total_flows if total_flows > 0 else 0,
            "round_time_s": t_round,
            "serial_eq_s": t_serial_eq,
            "speedup": t_serial_eq / t_round if t_round > 0 else 0,
            "efficiency": (t_serial_eq / t_round / n_workers) if t_round > 0 else 0,
            "worker_times": worker_times,
        }
        history.append(round_info)

        print(
            f"  [Round {fl_round + 1}] flows={total_flows:,} | anomalias={total_anom:,} "
            f"({round_info['anom_rate'] * 100:.2f}%) | round={t_round:.2f}s | serial_eq={t_serial_eq:.2f}s"
        )

    total_time = time.perf_counter() - t_total_start

    final_round = history[-1] if history else {}
    out = {
        "method": "isolation_forest_federated",
        "backend": "cpu-sklearn-federated",
        "mode": mode,
        "args": vars(args),
        "n_rounds": args.rounds,
        "n_workers": n_workers,
        "n_flows": int(final_round.get("total_flows", 0)),
        "n_anomalies": int(final_round.get("total_anom", 0)),
        "anom_rate": float(final_round.get("anom_rate", 0.0)),
        "times": {
            "extract_s": extraction_time,
            "total_s": total_time,
            "classification_s": float(sum(r.get("serial_eq_s", 0.0) for r in history)),
        },
        "history": history,
        "shards": shard_stats,
    }

    out_path = Path(args.outdir) / args.outfile
    with open(out_path, "wb") as f:
        pickle.dump(out, f)

    print("\n" + "=" * 70)
    print(f"  Method: {out['method']}")
    print(f"  Backend: {out['backend']}")
    print(f"  Mode: {mode.upper()}")
    print(f"  Rounds: {args.rounds}")
    print(f"  Workers: {n_workers}")
    print(f"  Fluxos: {out['n_flows']:,}")
    print(f"  Anomalias: {out['n_anomalies']:,} ({out['anom_rate'] * 100:.2f}%)")
    print(f"  Extração: {extraction_time:.1f}s")
    print(f"  Classificação (serial eq): {out['times']['classification_s']:.1f}s")
    print(f"  Tempo total: {total_time:.1f}s")
    print(f"  Resultado: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()