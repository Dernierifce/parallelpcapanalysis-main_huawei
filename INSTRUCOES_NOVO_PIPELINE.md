# INSTRUÇÕES DE USO: Novo Pipeline com Foco em Classificação

## Problema Resolvido

Anteriormente, o tempo total incluía **extração de features** (I/O com pyshark), que **não é** classificação de anomalias.

Agora:
- **Extração**: executada UMA VEZ e salva em cache
- **Classificação**: medida isoladamente (treino + inferência)

---

## Fluxo Recomendado

### 1️⃣ Pré-processar Features (executar UMA VEZ)

```bash
python preprocess_features.py \
    --shards /data/pcaps/*.pcapng \
    --outdir /data/results \
    --cache-file features_cache.pkl
```

**Saída:**
- `/data/results/features_cache.pkl` — features normalizadas prontas para benchmarks

**Tempo reportado:** Extração (informativo, não medido posteriormente)

---

### 2️⃣ Benchmark GPU (apenas classificação)

```bash
python gpu_train.py \
    --cache-file /data/results/features_cache.pkl \
    --outdir /data/results \
    --outfile gpu_results.pkl \
    --estimators 400 \
    --contamination 0.05
```

**Saída:**
```
  Backend: gpu-cuml
  Modo: CACHE
  Fluxos: 420,000
  Anomalias: 21,000 (5.00%)
  ─────────────────────────────
  Treino: 12.3s
  Inferência: 0.8s
  ► Classificação (train+infer): 13.1s ◄ [MÉTRICA]
  Total (incl. extração): 13.1s
```

---

### 3️⃣ Benchmark CPU Federado (apenas classificação)

```bash
python federated_train.py \
    --cache-file /data/results/features_cache.pkl \
    --outdir /data/results \
    --rounds 6 \
    --workers 4 \
    --outfile cpu_federated_results.pkl \
    --estimators 200 \
    --contamination 0.05
```

**Saída:**
```
  PAD — Treinamento Federado (Modo CACHE)
  Modo: CACHE
  
  [...]
  
  Treinamento concluído em 47.2s
  Modo: CACHE
  ► Métrica final (classificação): 179.3s
```

---

### 4️⃣ Gerar Gráficos Comparativos

```bash
python plots_cpu_gpu_compare.py \
    --cpu-results /data/results/cpu_federated_results.pkl \
    --gpu-results /data/results/gpu_results.pkl \
    --outdir /data/results \
    --basename cpu_gpu_comparison
```

**Gráficos gerados:**
- `cpu_gpu_comparison.png` 
- `cpu_gpu_comparison.pdf`

**Destaque:**
- Gráfico 1: **CLASSIFICAÇÃO** (train + infer) — métrica principal
- Speedup claramente indicado

---

## Novo Relatório vs Antigo

### ANTES (Total incluía extração):
```
Extração: 71.2s
Treino: 200.1s
Inferência: 14.0s
─────────────
Total: 285.3s  ← PROBLEMA: inclui I/O de pyshark!
```

### DEPOIS (Cache — apenas classificação):
```
Extração (pré-processamento): 71.2s [NÃO MEDIDO]
─────────────────────────────────────────────────
Treino: 200.1s
Inferência: 14.0s
─────────────────────────────────────────────────
► Classificação: 214.1s ◄ [MÉTRICA REAL]
```

---

## Modo Legado (sem cache — ainda suportado)

Se precisar usar o modo antigo (com extração inclusa):

```bash
# GPU
python gpu_train.py \
    --shards /data/pcaps/*.pcapng \
    --outdir /data/results \
    --outfile gpu_results.pkl

# CPU
python federated_train.py \
    --shards /data/pcaps/*.pcapng \
    --outdir /data/results \
    --rounds 6 \
    --workers 4
```

⚠️ **Nota:** Sem cache, extração e classificação não são separadas.

---

## Detalhes Técnicos

### `preprocess_features.py`
- Lê PCAPNGs
- Extrai features com `pyshark` + `feature_extractor.py`
- Normaliza com `StandardScaler`
- Salva em pickle: `X_scaled`, `scaler`, `shard_stats`

### `gpu_train.py` (modificações)
- Novo param: `--cache-file`
- Campo novo no pickle: `"classification_s"` (train + infer)
- Campo novo no pickle: `"mode"` ("cache" ou "full")

### `federated_train.py` (modificações)
- Novo param: `--cache-file`
- `worker_train()`: separa `extraction_time` de `classification_time`
- Campo novo no pickle: `"mode"` ("cache" ou "full")
- Histórico agora inclui `"mode"` em cada round

### `plots_cpu_gpu_compare.py` (modificações)
- Extrai `classification_s` se disponível
- Prioriza comparação por "classificação" se ambos em modo "cache"
- Fallback para "total" se modos diferentes
- Relatório mais claro com modos destacados

---

## Docker (Docker Compose)

O dashboard ainda funciona com os scripts antigos e novos:

```bash
docker compose up -d
```

Você pode usar a interface para:
1. Upload de PCAPs
2. Executar `preprocess_features.py` (novo)
3. Benchmark GPU com cache (novo)
4. Benchmark CPU com cache (novo)
5. Gerar gráficos (atualizado)

---

## Checklist para Validação

- [ ] `preprocess_features.py` cria `features_cache.pkl` sem erros
- [ ] `gpu_train.py --cache-file` roda e reporta `classification_s`
- [ ] `federated_train.py --cache-file` roda e reporta `classification_s` por worker
- [ ] `plots_cpu_gpu_compare.py` gera gráficos com "CLASSIFICAÇÃO" em destaque
- [ ] Modo legado (sem cache) ainda funciona
- [ ] Gráficos mostram speedup correto (CPU classification / GPU classification)

---

## Exemplos de Resultados Esperados

### GPU Benchmark (Cache Mode)
```
► Classificação (train+infer): 13.1s ◄ [MÉTRICA]
```

### CPU Federado (Cache Mode - 6 rounds)
```
► Métrica final (classificação): 179.3s
```

### Speedup Final
```
Speedup: 13.7× (179.3 / 13.1)
```

Este é o **verdadeiro ganho do GPU** em classificação, sem ruído de extração!

---

## Troubleshooting

### Erro: "Arquivo .pkl não encontrado"
- Certifique-se de que `preprocess_features.py` foi executado primeiro
- Verifique caminho em `--cache-file`

### "Modo: unknown"
- Resultado foi gerado com versão antiga
- Regenere com scripts atualizados

### Speedup = 1.0x
- Ambos backends levam o mesmo tempo
- Verifique se GPU está realmente sendo usada (`nvidia-smi` durante execução)

---

Pronto! Agora você mede **apenas** o esforço computacional de classificação. 🎯
