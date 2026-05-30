# INSTRUÇÕES DE USO: Novo Pipeline com Foco em Classificação

## Problema Resolvido

O projeto agora trabalha com três fluxos principais:

- **CPU**: Isolation Forest
- **CPU federado**: Isolation Forest com agregação entre shards
- **GPU**: Autoencoder em PyTorch

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

### 2️⃣ Benchmark GPU (Autoencoder)

```bash
python gpu_train.py \
    --cache-file ./data/results/features_cache.pkl \
    --outdir ./data/results \
    --outfile gpu_results.pkl \
    --epochs 20 \
    --batch-size 1024 \
    --latent-dim 16 \
    --anom-percentile 95.0
```

**Saída:**

```text
Backend: cuda
Modo: CACHE
Fluxos: 420,000
Anomalias: 21,000 (5.00%)
─────────────────────────────
Treino: 12.3s
Inferência: 0.8s
► Classificação (train+infer): 13.1s ◄ [MÉTRICA]
Total (incl. extração): 13.1s
```

### 3️⃣ Benchmark CPU (Isolation Forest)

```bash
python cpu_train.py \
    --cache-file ./data/results/features_cache.pkl \
    --outdir ./data/results \
    --outfile cpu_results.pkl \
    --estimators 200 \
    --contamination 0.05
```

**Saída:**

```text
Benchmark CPU — Isolation Forest
Modo: CACHE

[...]

► Métrica final (classificação): 179.3s
```

### 4️⃣ Benchmark CPU Federado (Isolation Forest)

```bash
python federated_train.py \
    --cache-file ./data/results/features_cache.pkl \
    --outdir ./data/results \
    --outfile cpu_federated_results.pkl \
    --rounds 6 \
    --workers 4 \
    --estimators 200 \
    --contamination 0.05
```

**Saída:**

```text
PAD — Treinamento Federado (Isolation Forest)
Mode: CACHE

[...]

Classificação (serial eq): 179.3s
```

### 5️⃣ Gerar Gráficos Comparativos

```bash
python plots_cpu_gpu_compare.py \
    --cpu-results ./data/results/cpu_results.pkl \
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

### `cpu_train.py`

- Benchmark direto de Isolation Forest em CPU
- Usa cache quando disponível e também aceita shards legados
- Salva `classification_s` em `times`

### `federated_train.py`

- Treinamento federado com Isolation Forest em CPU
- Usa cache quando disponível e também aceita shards legados
- Salva `history` com tempos por round e `classification_s`
- Campo novo no pickle: `mode` (`cache` ou `full`)

### `plots_cpu_gpu_compare.py` (modificações)

- Extrai `classification_s` se disponível
- Relatório mais claro com modos destacados

## Execução Local no Servidor

O projeto agora foi ajustado para rodar diretamente na pasta do servidor, sem Docker. Os caminhos padrão passam a usar `./data/pcaps` e `./data/results`.

```bash
python preprocess_features.py --cache-file features_cache.pkl
python cpu_train.py --cache-file ./data/results/features_cache.pkl --outfile cpu_results.pkl
python federated_train.py --cache-file ./data/results/features_cache.pkl --outfile cpu_federated_results.pkl
python gpu_train.py --cache-file ./data/results/features_cache.pkl --outfile gpu_results.pkl
streamlit run app_gpu_dashboard.py
```

Se precisar sobrescrever os diretórios padrão, use as variáveis de ambiente `PCAP_DIR` e `RESULTS_DIR`.

## Checklist para Validação

- [ ] `preprocess_features.py` cria `features_cache.pkl` sem erros
- [ ] `gpu_train.py --cache-file` roda e reporta `classification_s`
- [ ] `cpu_train.py --cache-file` roda e reporta `classification_s`
- [ ] `federated_train.py --cache-file` roda e reporta `classification_s` por round
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
