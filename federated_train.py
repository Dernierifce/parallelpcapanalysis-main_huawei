"""
federated_train.py
Pipeline federado não supervisionado com IsolationForest + FedAvg.

Modos:
    1. Com CACHE (recomendado para focar em classificação):
       python federated_train.py \
           --cache-file ./data/results/features_cache.pkl \
           --outdir ./data/results \
           --rounds 6 \
           --workers 4 \
           --outfile cpu_federated_results.pkl
    
    2. Sem cache (modo legado — inclui extração):
       python federated_train.py \
           --shards ./data/pcaps/*.pcapng \
           --outdir ./data/results \
           --rounds 6 \
           --workers 4

Nota: modo com --cache-file isola medição APENAS de classificação (train + infer)
"""

import argparse
import glob
import pickle
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from feature_extractor import extract_flows, FEATURE_COLS

# ── Defaults ──────────────────────────────────────────────────
BASE_DIR           = Path(__file__).resolve().parent
DEFAULT_SHARDS     = sorted(glob.glob(str(BASE_DIR / "data" / "pcaps" / "*.pcapng")))
DEFAULT_OUTDIR     = str(BASE_DIR / "data" / "results")
DEFAULT_OUTFILE    = "final_results.pkl"
N_ESTIMATORS       = 200
CONTAMINATION      = 0.05
N_FL_ROUNDS        = 6
N_WORKERS          = 4


# ══════════════════════════════════════════════════════════════
# Worker: extrai features e treina modelo local
# ══════════════════════════════════════════════════════════════
def worker_train(args: tuple) -> dict:
    """
    Executado em processo separado (multiprocessing.Pool).
    
    Args:
        args: tuple (worker_id, features_data, global_params, fl_round, 
                     n_estimators, contamination, use_cache)
              - Se use_cache: features_data é um dicionário com X_scaled
              - Se não: features_data é shard_path string
    
    Retorna: dict com métricas incluindo classification_time (sem extração)
    """
    if len(args) == 8:
        worker_id, features_data, global_params, fl_round, n_estimators, contamination, use_cache, shard_path = args
    else:
        # Compatibilidade com modo legado
        worker_id, shard_path, global_params, fl_round, n_estimators, contamination = args
        use_cache = False
        features_data = None

    classification_start = time.perf_counter()

    # ── Carregar features: do cache OU extrair ────────────────────────
    if use_cache and features_data:
        # Modo cache: não medir extração
        X_sc = features_data["X_scaled"]
        df_n_flows = features_data["n_flows"]
        extraction_time = 0.0
    else:
        # Modo legado: extrair (inclui no tempo total)
        t_extract_start = time.perf_counter()
        df = extract_flows(shard_path, anonymize=True)
        X = df[FEATURE_COLS].fillna(0).values
        df_n_flows = len(df)
        extraction_time = time.perf_counter() - t_extract_start

        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)

    # ── Treino local ──────────────────────────────────────────
    t_train_start = time.perf_counter()
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=worker_id + fl_round * 100,
        warm_start=(fl_round > 0 and global_params is not None),
        n_jobs=1,   # cada worker usa 1 thread (paralelismo já é pelo Pool)
    )

    if fl_round > 0 and global_params is not None:
        model.estimators_ = list(global_params["estimators"])
        model.estimators_features_ = list(global_params["features"])
        model.n_features_in_ = global_params["n_features"]
        model.max_samples_ = global_params.get("max_samples", 256)

    model.fit(X_sc)
    t_train = time.perf_counter() - t_train_start

    # ── Inferência ────────────────────────────────────────────
    t_infer_start = time.perf_counter()
    scores = model.decision_function(X_sc)
    labels = model.predict(X_sc)  # -1 = anomalia, 1 = normal
    t_infer = time.perf_counter() - t_infer_start

    n_anom = int((labels == -1).sum())
    classification_time = t_train + t_infer
    total_elapsed = extraction_time + classification_time

    print(
        f"  [W{worker_id}] R{fl_round + 1} | "
        f"fluxos={df_n_flows:,} | anomalias={n_anom:,} "
        f"({n_anom / df_n_flows * 100:.2f}%) | "
        f"class={classification_time:.2f}s"
    )

    return {
        "worker_id": worker_id,
        "fl_round": fl_round,
        "shard_path": str(shard_path),
        "n_flows": df_n_flows,
        "n_anomalies": n_anom,
        "anom_rate": n_anom / df_n_flows,
        "times": {
            "extract_s": extraction_time,
            "train_s": t_train,
            "infer_s": t_infer,
            "classification_s": classification_time,
            "total_s": total_elapsed,
        },
        "scores": scores,
        "labels": labels,
        "X_scaled": X_sc,
        "local_params": {
            "estimators": model.estimators_,
            "features": model.estimators_features_,
            "n_features": model.n_features_in_,
            "max_samples": getattr(model, "max_samples_", 256),
            "n_flows": df_n_flows,
        },
    }


# ══════════════════════════════════════════════════════════════
# Agregador FedAvg
# ══════════════════════════════════════════════════════════════
def fedavg_aggregate(worker_results: list, n_estimators: int) -> dict:
    """
    FedAvg para IsolationForest: seleciona subconjunto de árvores
    de cada worker ponderado proporcionalmente ao nº de fluxos.
    Simula troca de parâmetros sem transferir dados brutos.
    """
    total_flows = sum(r["local_params"]["n_flows"] for r in worker_results)
    weights     = [r["local_params"]["n_flows"] / total_flows
                   for r in worker_results]

    all_estimators:    list = []
    all_feat_subsets:  list = []

    for r, w in zip(worker_results, weights):
        n_trees = max(1, round(w * n_estimators))
        local_est  = r["local_params"]["estimators"]
        local_feat = r["local_params"]["features"]
        idx = np.random.choice(len(local_est), min(n_trees, len(local_est)),
                               replace=False)
        all_estimators   += [local_est[i]  for i in idx]
        all_feat_subsets += [local_feat[i] for i in idx]

    return {
        "estimators":  all_estimators,
        "features":    all_feat_subsets,
        "n_features":  worker_results[0]["local_params"]["n_features"],
        "max_samples": worker_results[0]["local_params"]["max_samples"],
    }


# ══════════════════════════════════════════════════════════════
# Loop principal
# ══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Treinamento federado não supervisionado — PAD/IFCE"
    )
    parser.add_argument(
        "--shards", nargs="+", default=DEFAULT_SHARDS, help="Arquivos pcapng dos shards"
    )
    parser.add_argument(
        "--cache-file",
        default=None,
        help="Arquivo .pkl com features pré-extraídas (modo recomendado)",
    )
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument("--rounds", type=int, default=N_FL_ROUNDS)
    parser.add_argument(
        "--workers", type=int, default=min(N_WORKERS, cpu_count())
    )
    parser.add_argument("--estimators", type=int, default=N_ESTIMATORS)
    parser.add_argument("--contamination", type=float, default=CONTAMINATION)
    parser.add_argument("--outfile", default=DEFAULT_OUTFILE, help="Nome do arquivo .pkl de saída")
    args = parser.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    # ── CARREGAR FEATURES: Do cache OU preparar shards ──────────────────
    if args.cache_file:
        print("=" * 70)
        print("  PAD — Treinamento Federado (Modo CACHE)")
        print(f"  Carregando cache: {args.cache_file}")
        print(f"  Workers: {args.workers}")
        print(f"  Rounds: {args.rounds}")
        print(f"  N trees: {args.estimators}")
        print("=" * 70)

        with open(args.cache_file, "rb") as f:
            cache = pickle.load(f)

        X_scaled = cache["X_scaled"]
        shard_stats = cache.get("shard_stats", [])
        total_flows_cache = cache["n_flows"]
        use_cache = True
        mode = "cache"

        # Distribuir features entre workers proporcionalmente
        n_flows_per_shard = [s["n_flows"] for s in shard_stats]
        n_workers = min(args.workers, len(shard_stats))

        # Dividir X_scaled entre workers
        split_indices = [0]
        for n in n_flows_per_shard:
            split_indices.append(split_indices[-1] + n)

        cache_per_worker = []
        for i in range(n_workers):
            shard_idx = i % len(shard_stats)
            start_idx = split_indices[shard_idx]
            end_idx = split_indices[shard_idx + 1]
            cache_per_worker.append(
                {
                    "X_scaled": X_scaled[start_idx:end_idx],
                    "n_flows": end_idx - start_idx,
                }
            )

        shards_or_cache = cache_per_worker
    else:
        print("=" * 70)
        print("  PAD — Treinamento Federado (Modo COMPLETO com extração)")
        print(f"  Shards: {len(args.shards)}")
        print(f"  Workers: {args.workers}")
        print(f"  Rounds: {args.rounds}")
        print(f"  N trees: {args.estimators}")
        print("=" * 70)

        shards = sorted(args.shards)
        if not shards:
            raise ValueError("Nenhum shard informado/encontrado. Verifique --shards.")
        use_cache = False
        mode = "full"
        shard_stats = [{"shard_path": str(s), "n_flows": 0} for s in shards]
        shards_or_cache = shards

    print(f"  Modo: {mode.upper()}\n")

    global_params = None
    history = []
    t_total_start = time.perf_counter()

    for fl_round in range(args.rounds):
        print(f"\n{'─' * 50}")
        print(f"  Round FL {fl_round + 1}/{args.rounds}")
        print(f"{'─' * 50}")

        # ── Preparar argumentos para workers ──────────────────────────
        if use_cache:
            n_workers_actual = min(args.workers, len(shards_or_cache))
            task_args = [
                (
                    i,
                    shards_or_cache[i],
                    global_params,
                    fl_round,
                    args.estimators,
                    args.contamination,
                    use_cache,
                    f"cache_{i}",
                )
                for i in range(n_workers_actual)
            ]
        else:
            shards = shards_or_cache
            task_args = [
                (i, shards[i], global_params, fl_round, args.estimators, args.contamination)
                for i in range(len(shards))
            ]

        t_round_start = time.perf_counter()
        with Pool(processes=args.workers) as pool:
            results = pool.map(worker_train, task_args)
        t_round = time.perf_counter() - t_round_start

        # FedAvg
        global_params = fedavg_aggregate(results, args.estimators)

        # Métricas consolidadas
        total_flows = sum(r["n_flows"] for r in results)
        total_anom = sum(r["n_anomalies"] for r in results)

        # Tempos: usar classification_time se em modo cache
        if use_cache:
            worker_times = [r["times"]["classification_s"] for r in results]
        else:
            worker_times = [r["times"]["total_s"] for r in results]

        t_serial_eq = sum(worker_times) * 1.08  # overhead estimado

        round_info = {
            "round": fl_round + 1,
            "total_flows": total_flows,
            "total_anom": total_anom,
            "anom_rate": total_anom / total_flows if total_flows > 0 else 0,
            "round_time_s": t_round,
            "serial_eq_s": t_serial_eq,
            "speedup": t_serial_eq / t_round if t_round > 0 else 0,
            "efficiency": (t_serial_eq / t_round / args.workers) if t_round > 0 else 0,
            "worker_times": worker_times,
            "imbalance_pct": (
                (max(worker_times) - min(worker_times)) / np.mean(worker_times) * 100
                if worker_times
                else 0
            ),
            "mode": mode,
        }
        history.append(round_info)

        print(f"\n  Fluxos totais : {total_flows:,}")
        print(f"  Anomalias     : {total_anom:,}  ({total_anom / total_flows * 100:.2f}%)")
        print(f"  Tempo paralelo: {t_round:.1f}s")
        print(f"  Speedup S(N)  : {round_info['speedup']:.2f}×")
        print(f"  Eficiência    : {round_info['efficiency'] * 100:.1f}%")
        print(f"  Desbalance    : {round_info['imbalance_pct']:.1f}%")

    t_total = time.perf_counter() - t_total_start

    # Salva resultados
    out_file = Path(args.outdir) / args.outfile
    with open(out_file, "wb") as f:
        pickle.dump(
            {
                "mode": mode,
                "history": history,
                "final_round": results,
                "args": vars(args),
                "total_time_s": t_total,
            },
            f,
        )

    print("\n" + "=" * 70)
    print(f"  Treinamento concluído em {t_total:.1f}s")
    print(f"  Modo: {mode.upper()}")
    if mode == "cache":
        print(f"  ► Métrica final (classificação): "
              f"{sum(h['serial_eq_s'] for h in history):.1f}s")
    print(f"  Resultados salvos em: {out_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
