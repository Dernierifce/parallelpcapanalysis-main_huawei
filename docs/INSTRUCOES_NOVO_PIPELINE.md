# INSTRUÇÕES DE USO: Novo Pipeline com Foco em Classificação

## Problema Resolvido

Anteriormente, o tempo total incluía **extração de features** (I/O com pyshark), que **não é** classificação de anomalias.

Agora:

- **Extração**: executada uma vez e salva em cache
- **Classificação**: medida isoladamente (treino + inferência)

## Fluxo Recomendado

### 1️⃣ Pré-processar Features (executar uma vez)

```bash
python preprocess_features.py \
    --shards ./data/pcaps/*.pcapng \
    --outdir ./data/results \
    --cache-file features_cache.pkl
```

**Saída:**

- `./data/results/features_cache.pkl` - features normalizadas prontas para benchmarks

**Tempo reportado:** extração (informativo, não medido posteriormente)

### 2️⃣ Benchmark GPU (apenas classificação)

```bash
python gpu_train.py \
    --cache-file ./data/results/features_cache.pkl \
    --outdir ./data/results \
    --outfile gpu_results.pkl \
    --estimators 400 \
    --contamination 0.05
```

**Saída:**

```text
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

### 3️⃣ Benchmark CPU Federado (apenas classificação)

```bash
python federated_train.py \
    --cache-file ./data/results/features_cache.pkl \
    --outdir ./data/results \
    --rounds 6 \
    --workers 4 \
    --outfile cpu_federated_results.pkl \
    --estimators 200 \
    --contamination 0.05
```

**Saída:**

```text
PAD - Treinamento Federado (Modo CACHE)
Modo: CACHE

[...]

Treinamento concluído em 47.2s
Modo: CACHE
► Métrica final (classificação): 179.3s
```

### 4️⃣ Gerar Gráficos Comparativos

```bash
python plots_cpu_gpu_compare.py \
    --cpu-results ./data/results/cpu_federated_results.pkl \
    --gpu-results ./data/results/gpu_results.pkl \
    --outdir ./data/results \
    --basename cpu_gpu_comparison
```

**Gráficos gerados:**

- `cpu_gpu_comparison.png`
- `cpu_gpu_comparison.pdf`

**Destaque:**

- Gráfico 1: **CLASSIFICAÇÃO** (train + infer) - métrica principal
- Speedup claramente indicado

## Novo Relatório vs Antigo

### ANTES (Total incluía extração)

```text
Extração: 71.2s
Treino: 200.1s
Inferência: 14.0s
─────────────
Total: 285.3s  ← PROBLEMA: inclui I/O de pyshark!
```

### DEPOIS (Cache - apenas classificação)

```text
Extração (pré-processamento): 71.2s [NÃO MEDIDO]
─────────────────────────────────────────────────
Treino: 200.1s
Inferência: 14.0s
─────────────────────────────────────────────────
► Classificação: 214.1s ◄ [MÉTRICA REAL]
```

## Modo Legado (sem cache - ainda suportado)

Se precisar usar o modo antigo (com extração inclusa):

```bash
# GPU
python gpu_train.py \
    --shards ./data/pcaps/*.pcapng \
    --outdir ./data/results \
    --outfile gpu_results.pkl

# CPU
python federated_train.py \
    --shards ./data/pcaps/*.pcapng \
    --outdir ./data/results \
    --rounds 6 \
    --workers 4
```

⚠️ **Nota:** sem cache, extração e classificação não são separadas.

## Detalhes Técnicos

### `preprocess_features.py`

- Lê PCAPNGs
- Extrai features com `pyshark` + `feature_extractor.py`
- Normaliza com `StandardScaler`
- Salva em pickle: `X_scaled`, `scaler`, `shard_stats`

### `gpu_train.py` (modificações)

- Novo param: `--cache-file`
- Campo novo no pickle: `classification_s` (train + infer)
- Campo novo no pickle: `mode` (`cache` ou `full`)

### `federated_train.py` (modificações)

- Novo param: `--cache-file`
- `worker_train()`: separa `extraction_time` de `classification_time`
- Campo novo no pickle: `mode` (`cache` ou `full`)
- Histórico agora inclui `mode` em cada round

### `plots_cpu_gpu_compare.py` (modificações)

- Extrai `classification_s` se disponível
- Prioriza comparação por classificação se ambos estiverem em modo `cache`
- Fallback para `total` se os modos forem diferentes
- Relatório mais claro com modos destacados

## Execução Local no Servidor

O projeto agora foi ajustado para rodar diretamente na pasta do servidor, sem Docker. Os caminhos padrão passam a usar `./data/pcaps` e `./data/results`.

```bash
python preprocess_features.py --cache-file features_cache.pkl
python gpu_train.py --cache-file ./data/results/features_cache.pkl --outfile gpu_results.pkl
python federated_train.py --cache-file ./data/results/features_cache.pkl --outfile cpu_federated_results.pkl
python plots_cpu_gpu_compare.py --cpu-results ./data/results/cpu_federated_results.pkl --gpu-results ./data/results/gpu_results.pkl
streamlit run app_gpu_dashboard.py
```

Se precisar sobrescrever os diretórios padrão, use as variáveis de ambiente `PCAP_DIR` e `RESULTS_DIR`.

## Checklist para Validação

- [ ] `preprocess_features.py` cria `features_cache.pkl` sem erros
- [ ] `gpu_train.py --cache-file` roda e reporta `classification_s`
- [ ] `federated_train.py --cache-file` roda e reporta `classification_s` por worker
- [ ] `plots_cpu_gpu_compare.py` gera gráficos com `CLASSIFICAÇÃO` em destaque
- [ ] Modo legado (sem cache) ainda funciona
- [ ] Gráficos mostram speedup correto (CPU classification / GPU classification)

## Exemplos de Resultados Esperados

### GPU Benchmark (Cache Mode)

```text
► Classificação (train+infer): 13.1s ◄ [MÉTRICA]
```

### CPU Federado (Cache Mode - 6 rounds)

```text
► Métrica final (classificação): 179.3s
```

### Speedup Final

```text
Speedup: 13.7× (179.3 / 13.1)
```

Este é o **verdadeiro ganho do GPU** em classificação, sem ruído de extração!

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

Pronto! Agora você mede **apenas** o esforço computacional de classificação.
