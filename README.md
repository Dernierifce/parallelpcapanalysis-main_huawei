# Parallel PCAP Analysis

> Pipeline completo de análise de tráfego de rede com foco em **qualidade de detecção de anomalias** e comparativo de desempenho **CPU vs GPU**. Extrai 22 features estatísticas de fluxo a partir de capturas `.pcapng`, normaliza os dados e executa benchmarks comparativos entre Autoencoder (PyTorch) e K-Means, gerando relatório visual e log detalhado automaticamente.

---

## Índice

1. [Visão Geral](#visão-geral)
2. [Resultados — Execução de Referência](#resultados--execução-de-referência)
3. [Requisitos de Sistema](#requisitos-de-sistema)
4. [Instalação](#instalação)
5. [Estrutura do Projeto](#estrutura-do-projeto)
6. [Uso — Comandos Principais](#uso--comandos-principais)
7. [Parâmetros de Execução](#parâmetros-de-execução)
8. [Pipeline Detalhado](#pipeline-detalhado)
9. [Features Extraídas](#features-extraídas)
10. [Algoritmos e Arquiteturas](#algoritmos-e-arquiteturas)
11. [Métricas de Qualidade](#métricas-de-qualidade)
12. [Saídas Geradas](#saídas-geradas)
13. [Histórico de Execuções](#histórico-de-execuções)
14. [Configurações Avançadas](#configurações-avançadas)
15. [Solução de Problemas](#solução-de-problemas)
16. [Ambiente de Referência](#ambiente-de-referência)

---

## Visão Geral

O projeto implementa um pipeline em quatro etapas para análise de tráfego de rede capturado em produção:

```
Arquivos .pcapng  (4 shards × ~1,2 GB)
        │
        ▼
feature_extractor.py ──► 22 features por fluxo (5-tupla) ──► features_cache.pkl
        │
        ▼
anomaly_compare.py (script unificado)
        ├── Autoencoder CPU  (PyTorch)
        ├── Autoencoder GPU  (PyTorch + CUDA)
        ├── K-Means CPU      (scikit-learn)
        └── K-Means GPU      (PyTorch puro — sem cuML)
                │
                ▼
    anomaly_compare.log  +  anomaly_report.png
    anomaly_compare.json +  anomaly_compare.csv
```

**Dois modos de execução:**

| Modo | Comando | Quando usar |
|---|---|---|
| Completo | `python anomaly_compare.py --gpu` | Primeira execução ou novos pcapng |
| Cache | `python anomaly_compare.py --skip-extraction --gpu` | Re-execuções (evita ~4,5h de extração) |

---

## Resultados — Execução de Referência

Duas execuções documentadas: **Teste 1** com 5k amostras/12 épocas e **Teste 2** com 20k amostras/20 épocas.

---

### Extração de Features (comum às duas execuções)

| Shard | Tamanho | Fluxos | Tempo | Taxa |
|---|---|---|---|---|
| shard\_00091\_20260414170212 | 1.192 MB | 40.287 | 77,4 min | ~9 fluxos/s |
| shard\_00092\_20260414170229 | 1.192 MB | 38.286 | 69,4 min | ~9 fluxos/s |
| shard\_00093\_20260414170247 | 1.192 MB | 36.294 | 67,3 min | ~9 fluxos/s |
| shard\_00094\_20260414170303 | 1.004 MB | 31.832 | 59,4 min | ~9 fluxos/s |
| **TOTAL** | **4.580 MB** | **146.699** | **273,5 min** | — |

Cache gerado: `features_cache.pkl` (~37 MB) com 146.699 × 22 features.

---

### Teste 1 — 5.000 amostras, 12 épocas, k=8

**Execução:** 09/06/2026 19:46 | Duração total: **7,54s** (benchmark apenas)

#### Desempenho

| Método | Hardware | Treino | Inferência | Total | Speedup |
|---|---|---|---|---|---|
| Autoencoder | CPU | 0,516s | 0,025s | 0,541s | referência |
| Autoencoder | GPU (RTX 4060) | 0,891s | 0,067s | 0,958s | **0,56x** ⚠ |
| K-Means | CPU (sklearn) | 2,785s | 0,001s | 2,786s | referência |
| K-Means | GPU (PyTorch) | 0,221s | 0,000s | 0,221s | **12,59x** ✓ |

#### Qualidade — Autoencoder (5k, 12 épocas)

| Métrica | CPU | GPU |
|---|---|---|
| Loss final | 0,1868 | 0,1802 |
| Score médio | 0,1782 | 0,1709 |
| Score p95 (threshold) | 0,3666 | 0,4147 |
| Anomalias @ p95 | 250 (5,0%) | 250 (5,0%) |
| Score máximo | 95,91 | 95,11 |

#### Qualidade — K-Means (5k, k=8)

| Métrica | CPU | GPU |
|---|---|---|
| Silhouette Score | **0,3529** | 0,3342 |
| Inércia | 39.423 | 39.909 |
| Iterações reais | 12 / 300 | 16 / 300 |
| Anomalias @ p95 | 250 (5,0%) | 250 (5,0%) |
| Clusters suspeitos (<1%) | C4 (5 fluxos), C7 (23) | C2 (6 fluxos) |
| Fluxos suspeitos totais | 28 (0,56%) | 6 (0,12%) |

---

### Teste 2 — 20.000 amostras, 20 épocas, k=10

**Execução:** 09/06/2026 19:49 | GPU: RTX 4060

#### Desempenho

| Método | Hardware | Treino | Inferência | Total | Speedup |
|---|---|---|---|---|---|
| Autoencoder | CPU | 3,396s | 0,098s | 3,494s | referência |
| Autoencoder | GPU (RTX 4060) | 4,003s | 0,109s | 4,112s | **0,85x** ⚠ |
| K-Means | CPU (sklearn) | 2,176s | 0,001s | 2,177s | referência |
| K-Means | GPU (PyTorch) | 0,170s | 0,000s | 0,171s | **12,77x** ✓ |

#### Qualidade — Autoencoder (20k, 20 épocas)

| Métrica | CPU | GPU |
|---|---|---|
| Loss final | **0,04574** | **0,02526** |
| Score médio | 0,04462 | 0,02611 |
| Score p95 (threshold) | 0,1681 | 0,1086 |
| Anomalias @ p95 | 1.000 (5,0%) | 1.000 (5,0%) |
| Score máximo | 31,01 | 12,04 |

> GPU atingiu loss **44% menor** (0,025 vs 0,046) com 20 épocas — maior capacidade de aprendizado em batches maiores.

#### Qualidade — K-Means (20k, k=10)

| Métrica | CPU | GPU |
|---|---|---|
| Silhouette Score | **0,3337** | 0,3140 |
| Inércia | 144.600 | 181.021 |
| Iterações reais | 13 / 300 | 22 / 300 |
| Anomalias @ p95 | 1.000 (5,0%) | 1.000 (5,0%) |
| Clusters suspeitos (<1%) | C4 (4), C7 (39), C9 (7) | C3 (180), C4 (24) |
| Fluxos suspeitos totais | 50 (0,25%) | 204 (1,02%) |

---

### Análise de Threshold — Impacto na Taxa de Anomalia

Todos os métodos respondem de forma consistente à variação do percentil de corte:

| Threshold | Taxa esperada | AE-CPU | AE-GPU | KM-CPU | KM-GPU |
|---|---|---|---|---|---|
| p90 | 10% | 10,0% | 10,0% | 10,0% | 10,0% |
| p95 | 5% | 5,0% | 5,0% | 5,0% | 5,0% |
| p97 | 3% | 3,0% | 3,0% | 3,0% | 3,0% |
| p99 | 1% | 1,0% | 1,0% | 1,0% | 1,0% |

> Comportamento ideal — todos os métodos são calibráveis pelo percentil de corte.

---

### Interpretação dos Resultados

**Autoencoder GPU ainda mais lento que CPU (0,56x → 0,85x):**
com 5k–20k amostras e modelo pequeno (22→44→8), o overhead de inicialização CUDA e transferência `pin_memory` supera o ganho de paralelismo. A tendência é de aceleração a partir de ~50k amostras. A GPU compensou em **qualidade**: loss final 44% menor com 20 épocas.

**K-Means GPU consistentemente 12–13x mais rápido:**
algoritmo Lloyd é naturalmente vetorizável — `torch.cdist` executa o cálculo de distâncias em paralelo massivo na GPU. Benefício imediato mesmo com amostras pequenas.

**Silhouette Score (CPU > GPU em ambos os testes):**
K-Means CPU usa inicialização kmeans++ do scikit-learn com múltiplos re-starts (`n_init=10`), garantindo solução mais próxima do ótimo global. A implementação GPU usa kmeans++ com 1 re-start — troca qualidade por velocidade.

**Anomaly Rate idêntica entre CPU e GPU:**
ambos detectam exatamente os mesmos percentuais por construção (threshold = percentil dos próprios scores). A diferença está nos **fluxos específicos** detectados, não na quantidade.

---

## Requisitos de Sistema

### Hardware

| Componente | Mínimo | Recomendado (testado) |
|---|---|---|
| CPU | 4 cores | Intel/AMD moderno |
| RAM | 8 GB | 16 GB+ |
| Armazenamento | 10 GB livres | SSD (extração longa) |
| GPU | — | NVIDIA RTX 4060 (8 GB VRAM) |
| Driver NVIDIA | — | 591.86+ |

### Software

| Componente | Versão mínima | Versão testada |
|---|---|---|
| Python | 3.10+ | 3.13.12 |
| Windows | 10 | 11 |
| Wireshark / TShark | 4.0 | 4.x |
| CUDA Toolkit | 12.1 | 13.1 (cu128) |
| PyTorch | 2.0 | 2.11.0+cu128 |

### Dependências Python — `requirements.txt`

```
pyshark       # leitura de pcapng via TShark
pandas        # manipulação de DataFrames
numpy         # operações numéricas
scikit-learn  # StandardScaler, KMeans, IsolationForest, silhouette_score
scipy         # utilitários científicos
tqdm          # barras de progresso
matplotlib    # geração de gráficos e relatório PNG
streamlit     # dashboard interativo (opcional)
psutil        # monitoramento de recursos
torch         # PyTorch — Autoencoder e K-Means GPU
```

### Dependências GPU adicionais — `requirements-gpu.txt`

```
cupy-cuda12x  # operações NumPy aceleradas em GPU (opcional)
```

> **Nota:** cuML (RAPIDS) **não é necessário** no Windows. O K-Means GPU usa implementação PyTorch pura incluída em `anomaly_compare.py`.

---

## Instalação

### 1. Clone o repositório

```powershell
git clone <url-do-repositorio>
cd parallelpcapanalysis-main_huawei
```

### 2. Instale as dependências Python

```powershell
pip install -r requirements.txt
```

### 3. Instale o PyTorch com suporte CUDA (GPU NVIDIA)

```powershell
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Verifique:

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
# Esperado: CUDA: True | GPU: NVIDIA GeForce RTX 4060
```

> **Atenção:** Para Python 3.13, use obrigatoriamente `cu128`. O índice `cu121` não tem pacotes para esta versão.

### 4. Instale o Wireshark / TShark

Baixe em https://www.wireshark.org/download.html marcando:
- ✅ **TShark**
- ✅ **Add Wireshark to the system PATH**

Adicione ao PATH da sessão atual (se não adicionado globalmente):

```powershell
$env:PATH += ";C:\Program Files\Wireshark"
tshark -v
```

Para tornar permanente sem permissão de administrador:

```powershell
$p = [System.Environment]::GetEnvironmentVariable("PATH", "User")
[System.Environment]::SetEnvironmentVariable("PATH", $p + ";C:\Program Files\Wireshark", "User")
# Feche e reabra o PowerShell
```

---

## Estrutura do Projeto

```
parallelpcapanalysis-main_huawei/
│
├── anomaly_compare.py              # ★ Script unificado principal
│                                   #   Extração + AE + KMeans + log + relatório PNG
├── feature_extractor.py            # Extração de features de fluxo via pyshark/TShark
├── cpu_train.py                    # Benchmark Isolation Forest — CPU (legado)
├── gpu_train.py                    # Benchmark Autoencoder — CPU/GPU (legado)
├── benchmark_article.py            # Benchmark AE + KMeans para artigo (legado)
├── run_pipeline.py                 # Pipeline anterior (substituído pelo anomaly_compare.py)
├── log_utils.py                    # Utilitários de logging (setup_run_logging, emit_report)
├── plots_pad_artigo.py             # Geração de plots para publicação do artigo
├── run_article.ps1                 # Wrapper PowerShell para benchmark_article.py
├── requirements.txt                # Dependências Python (CPU)
├── requirements-gpu.txt            # Dependências adicionais GPU (cupy)
│
├── data/
│   ├── pcaps/
│   │   ├── shard__00091_20260414170212.pcapng   # 1,19 GB — tráfego real
│   │   ├── shard__00092_20260414170229.pcapng   # 1,19 GB
│   │   ├── shard__00093_20260414170247.pcapng   # 1,19 GB
│   │   └── shard__00094_20260414170303.pcapng   # 1,00 GB
│   └── results/
│       ├── features_cache.pkl       # Cache de features normalizadas (~37 MB)
│       ├── anomaly_compare.log      # ★ Log detalhado da execução atual
│       ├── anomaly_report.png       # ★ Relatório visual com 7 painéis
│       ├── anomaly_compare.json     # ★ Resultados completos em JSON
│       ├── anomaly_compare.csv      # ★ Resultados em CSV
│       ├── pipeline.log             # Log execuções anteriores (run_pipeline.py)
│       ├── pipeline_report.png      # Relatório execuções anteriores
│       ├── teste 1/                 # 5k amostras, k=8, 12 épocas — CPU+GPU
│       └── teste 2/                 # 20k amostras, k=10, 20 épocas — CPU+GPU
│
├── scripts/
│   └── benchmark_article.py        # Cópia para uso via run_article.ps1
│
├── figures/                        # Gráficos gerados pelo run_article.ps1
│   ├── train_time.png
│   ├── inference_time.png
│   └── speedup.png
│
├── results/                        # Saídas do run_article.ps1
│   ├── article_table.md
│   ├── article_table.csv
│   └── benchmark_results.json
│
└── docs/
    └── INSTRUCOES_NOVO_PIPELINE.md
```

> ★ arquivos gerados pelo script principal `anomaly_compare.py`

---

## Uso — Comandos Principais

### Benchmark completo com GPU (recomendado)

```powershell
$env:PATH += ";C:\Program Files\Wireshark"
python anomaly_compare.py --skip-extraction --gpu
```

### Reproduzir Teste 1 (5k amostras, 12 épocas, k=8)

```powershell
python anomaly_compare.py --skip-extraction --gpu `
    --sample-size 5000 --ae-epochs 12 --kmeans-clusters 8
```

### Reproduzir Teste 2 (20k amostras, 20 épocas, k=10)

```powershell
python anomaly_compare.py --skip-extraction --gpu `
    --sample-size 20000 --ae-epochs 20 --kmeans-clusters 10
```

### Pipeline completo com extração (primeira execução)

```powershell
$env:PATH += ";C:\Program Files\Wireshark"
python anomaly_compare.py --gpu
```

> ⚠️ A extração dos 4 shards (~4,6 GB) leva aproximadamente **4,5 horas**.

### Somente CPU (sem GPU)

```powershell
python anomaly_compare.py --skip-extraction
```

---

## Parâmetros de Execução

### `anomaly_compare.py` (script principal)

| Parâmetro | Padrão | Descrição |
|---|---|---|
| `--outdir` | `./data/results` | Diretório de saída |
| `--cache-file` | `./data/results/features_cache.pkl` | Cache de features |
| `--skip-extraction` | `False` | Pula extração, usa cache |
| `--gpu` | `False` | Habilita GPU para AE e K-Means |
| `--sample-size` | `5000` | Amostras para benchmark (0 = todas) |
| `--ae-epochs` | `12` | Épocas do Autoencoder |
| `--ae-batch-size` | `256` | Batch size do Autoencoder |
| `--ae-latent-dim` | `8` | Dimensão do espaço latente |
| `--ae-lr` | `0.001` | Learning rate (Adam) |
| `--kmeans-clusters` | `8` | Número de clusters k |
| `--kmeans-max-iter` | `300` | Iterações máximas K-Means |

### `run_article.ps1` (benchmark legado)

| Parâmetro | Padrão | Descrição |
|---|---|---|
| `-CacheFile` | `.\data\results\features_cache.pkl` | Cache de entrada |
| `-Outdir` | `.` | Diretório raiz de saída |
| `-SampleSize` | `5000` | Amostras para benchmark |
| `-SyntheticSamples` | `6000` | Amostras sintéticas (sem cache) |
| `-SyntheticFeatures` | `16` | Features sintéticas (sem cache) |
| `-AeEpochs` | `12` | Épocas do Autoencoder |
| `-AeBatchSize` | `256` | Batch size |
| `-AeLatentDim` | `8` | Dimensão latente |
| `-KMeansClusters` | `8` | Clusters K-Means |
| `-KMeansMaxIter` | `300` | Iterações máximas |
| `-RunGpuAutoencoder` | `False` | Habilita GPU no Autoencoder |
| `-RunGpuKMeans` | `False` | Habilita GPU no K-Means |

---

## Pipeline Detalhado

### Etapa 1 — Extração de Features (`feature_extractor.py`)

Lê cada `.pcapng` pacote a pacote via `pyshark`, agrega por **fluxo de 5-tupla** e calcula 22 features por fluxo.

- **Identificação do fluxo:** `(src_ip_hash, dst_ip_hash, src_port, dst_port, proto)`
- **Anonimização:** IPs substituídos por SHA-256 truncado a 10 chars — conformidade com LGPD
- **TShark:** localizado via `$env:TSHARK_PATH` ou PATH do sistema
- **Tolerância a falhas:** captura parcial em caso de crash via `TSharkCrashException`
- **Taxa medida:** ~9 fluxos/s por arquivo de ~1,2 GB (~60–80 min/shard)

### Etapa 2 — Normalização e Cache

```python
# features_cache.pkl (~37 MB)
{
    "X_scaled": np.ndarray,    # (146699, 22) float32 — normalizado
    "X":        np.ndarray,    # (146699, 22) float32 — bruto
    "columns":  list[str],     # 22 nomes de features
    "shard_stats": list[dict], # metadados por shard
}
```

`StandardScaler` ajustado sobre todos os shards concatenados.

### Etapa 3 — Benchmarks (`anomaly_compare.py`)

Para cada método e hardware, coleta:

| Medição | Descrição |
|---|---|
| `train_s` | Tempo de treinamento com `cuda.synchronize()` |
| `infer_s` | Tempo de inferência/predição com `cuda.synchronize()` |
| `score_distribution` | Estatísticas completas dos scores (p50/p75/p90/p95/p99) |
| `threshold_analysis` | Anomaly rate para 5 valores de threshold (p90→p99) |
| `silhouette_score` | Qualidade dos clusters K-Means (amostrado em 5k) |
| `suspicious_clusters` | Clusters com <1% das amostras — candidatos a anomalia |

### Etapa 4 — Relatório Visual (`anomaly_report.png`)

7 painéis gerados automaticamente por `matplotlib`:

| Painel | Conteúdo |
|---|---|
| **Header** | Data, amostras, configuração, GPU |
| **Taxa de Anomalia** | Barras agrupadas por método e threshold (p90→p99) |
| **Distribuição de Scores** | p25/p50/p75/p95/p99 por método com threshold destacado |
| **Loss por Época** | Curva de convergência AE-CPU vs AE-GPU |
| **Silhouette e Inércia** | Barras duais KM-CPU vs KM-GPU |
| **Tabela Comparativa** | Treino/Infer/Total/Speedup/Anomalias/Score p95/Status |
| **Speedup + Anomaly Rate** | Painéis finais lado a lado |

---

## Features Extraídas

22 features calculadas por fluxo de rede:

| # | Feature | Tipo | Descrição |
|---|---|---|---|
| 1 | `duration` | float | Duração total do fluxo (s) |
| 2 | `proto` | int | Protocolo: 1=TCP, 0=outros |
| 3 | `src_port` | int | Porta de origem |
| 4 | `dst_port` | int | Porta de destino |
| 5 | `pkt_count` | int | Total de pacotes |
| 6 | `byte_count` | int | Total de bytes |
| 7 | `mean_pkt_size` | float | Tamanho médio dos pacotes (bytes) |
| 8 | `std_pkt_size` | float | Desvio padrão do tamanho |
| 9 | `mean_iat` | float | Inter-Arrival Time médio (s) |
| 10 | `std_iat` | float | Desvio padrão do IAT |
| 11 | `min_iat` | float | IAT mínimo |
| 12 | `max_iat` | float | IAT máximo |
| 13 | `flag_syn` | int | Contagem flags SYN |
| 14 | `flag_fin` | int | Contagem flags FIN |
| 15 | `flag_rst` | int | Contagem flags RST |
| 16 | `flag_psh` | int | Contagem flags PSH |
| 17 | `fwd_pkt_count` | int | Pacotes sentido forward |
| 18 | `bwd_pkt_count` | int | Pacotes sentido backward |
| 19 | `fwd_byte_ratio` | float | Razão bytes forward / total |
| 20 | `is_port_well_known` | int | 1 se `dst_port < 1024` |
| 21 | `is_ephemeral_src` | int | 1 se `src_port > 49151` |
| 22 | `bytes_per_pkt` | float | Média de bytes por pacote |

---

## Algoritmos e Arquiteturas

### Autoencoder (PyTorch) — CPU e GPU

Rede encoder-decoder para detecção por **erro de reconstrução**. Fluxos anômalos têm padrão diferente do tráfego normal e geram MSE alto.

```
Encoder: Input(22) → Linear → ReLU → Linear → ReLU → Latent(8)
Decoder: Latent(8) → Linear → ReLU → Linear → Output(22)
hidden_dim = max(32, n_features × 2) = 44
```

| Hiperparâmetro | Teste 1 | Teste 2 |
|---|---|---|
| Épocas | 12 | 20 |
| Batch size | 256 | 256 |
| Latent dim | 8 | 8 |
| Learning rate | 0,001 | 0,001 |
| Otimizador | Adam | Adam |
| Loss | MSE | MSE |

**Threshold de anomalia:** percentil 95 dos erros de reconstrução (configurável via análise de threshold).

### K-Means GPU — PyTorch puro

Implementação própria do algoritmo Lloyd vetorizada na GPU, sem dependência de cuML:

1. **Inicialização kmeans++** via `torch.multinomial` com distâncias por `torch.cdist`
2. **Loop de atualização** com detecção automática de convergência (`torch.equal`)
3. **Sincronização CUDA** (`torch.cuda.synchronize()`) para medição precisa
4. **Score de anomalia:** distância euclidiana de cada ponto ao seu centroide

**Diferença em relação ao CPU:** 1 re-start vs `n_init=10` do sklearn — troca qualidade por velocidade (silhouette ~5% menor, mas 12x mais rápido).

### K-Means CPU — scikit-learn

```python
KMeans(n_clusters=k, max_iter=300, n_init=10, random_state=42)
```

Múltiplos re-starts garantem solução próxima do ótimo global.

---

## Métricas de Qualidade

| Métrica | Algoritmo | Descrição | Referência |
|---|---|---|---|
| **MSE Loss** | Autoencoder | Erro de reconstrução médio por época | Menor = melhor representação |
| **Score Distribution** | Ambos | p50/p75/p90/p95/p99 dos scores | Cauda longa indica anomalias reais |
| **Anomaly Rate** | Ambos | % fluxos acima do threshold | Calibrável pelo percentil |
| **Threshold Analysis** | Ambos | Impacto de p90→p99 na taxa | Consistência entre métodos |
| **Silhouette Score** | K-Means | Coesão e separação dos clusters | -1→+1, >0,2 aceitável |
| **Inércia** | K-Means | Soma das distâncias quadráticas intra-cluster | Menor = clusters mais compactos |
| **Clusters Suspeitos** | K-Means | Clusters com <1% das amostras | Candidatos diretos a anomalia |

---

## Saídas Geradas

### `anomaly_compare.py` (principal)

| Arquivo | Descrição |
|---|---|
| `anomaly_compare.log` | Log completo com timestamps: épocas, scores, silhouette, threshold analysis, comparativo |
| `anomaly_report.png` | Relatório visual com 7 painéis |
| `anomaly_compare.json` | Resultados completos: args + métricas + threshold analysis + score_dist |
| `anomaly_compare.csv` | Linha por método: treino/infer/total/anomalias/silhouette/inércia |
| `features_cache.pkl` | Cache de features (gerado na extração) |

### Estrutura do `anomaly_compare.json`

```json
{
  "args": { "sample_size": 20000, "ae_epochs": 20, "kmeans_clusters": 10, ... },
  "timestamp": "2026-06-09T19:49:51",
  "results": {
    "AE-CPU": {
      "status": "ok", "train_s": 3.396, "infer_s": 0.098,
      "final_loss": 0.04574, "n_anomalies": 1000, "anomaly_rate": 5.0,
      "score_dist": { "mean": ..., "p50": ..., "p95": ..., "p99": ... },
      "threshold_analysis": { "p90": {...}, "p95": {...}, "p99": {...} }
    },
    "KM-CPU": {
      "quality": {
        "silhouette": 0.3337,
        "cluster_distribution": { "0": {"count": 3127, "pct": 15.63}, ... },
        "suspicious_clusters": [4, 7, 9],
        "n_suspicious_flows": 50
      }
    }
  }
}
```

---

## Histórico de Execuções

| # | Data | Amostras | Épocas | k | AE Speedup | KM Speedup | AE Loss (GPU) | Silhouette (CPU) |
|---|---|---|---|---|---|---|---|---|
| Teste 0 | 03/06/2026 | 5.000 | 12 | 8 | — (CPU only) | — | — | — |
| Teste 1 | 09/06/2026 19:46 | 5.000 | 12 | 8 | **0,56x** | **12,59x** | 0,1802 | 0,3529 |
| **Teste 2** | **09/06/2026 19:49** | **20.000** | **20** | **10** | **0,85x** | **12,77x** | **0,0253** | **0,3337** |

**Tendência observada:**
- Speedup AE melhora com mais amostras: 0,56x → 0,85x (aumentando → >1,0x esperado com 50k+)
- Speedup KM estável: ~12–13x independente do tamanho da amostra
- Loss AE GPU consistentemente melhor que CPU com mais épocas

---

## Configurações Avançadas

### Testar ponto de break-even do Autoencoder GPU

```powershell
# Teste com 50k amostras — esperado que GPU supere CPU
python anomaly_compare.py --skip-extraction --gpu --sample-size 50000 --ae-epochs 20
```

### Usar todas as amostras do cache

```powershell
python anomaly_compare.py --skip-extraction --gpu --sample-size 0 --ae-epochs 20
```

### Salvar em subdiretório por experimento

```powershell
python anomaly_compare.py --skip-extraction --gpu `
    --outdir .\data\results\exp_50k_k12 `
    --sample-size 50000 --kmeans-clusters 12 --ae-epochs 30
```

### Ajustar threshold de anomalia na análise

O script calcula automaticamente p90/p92/p95/p97/p99. Para usar um threshold fixo pós-execução, carregue o JSON:

```python
import json
with open("anomaly_compare.json") as f:
    results = json.load(f)
# Ver threshold para p97 no K-Means CPU
print(results["results"]["KM-CPU"]["threshold_analysis"]["p97"])
```

---

## Solução de Problemas

### `TShark não foi encontrado`

```powershell
$env:PATH += ";C:\Program Files\Wireshark"
# ou
$env:TSHARK_PATH = "C:\Program Files\Wireshark\tshark.exe"
tshark -v   # confirmar
```

### `CUDA: False` com GPU NVIDIA presente

```powershell
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.cuda.is_available())"
```

### `ERROR: Could not find a version` ao instalar PyTorch

Python 3.13 exige `cu128`. O índice `cu121` não tem suporte:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

### `cuML unavailable` no K-Means GPU

Esperado no Windows — cuML não está disponível via pip. `anomaly_compare.py` usa automaticamente a implementação **K-Means PyTorch puro** com speedup comprovado de 12–13x.

### `benchmark_article.py: unrecognized arguments: --shards`

O `benchmark_article.py` não aceita `--shards`. Use o fluxo correto:

```powershell
# Gera o cache primeiro
python anomaly_compare.py  # ou run_pipeline.py

# Depois roda o benchmark do artigo
.\run_article.ps1
```

### Cache corrompido

```powershell
Remove-Item .\data\results\features_cache.pkl
python anomaly_compare.py --gpu   # reprocessa tudo
```

### `Acesso ao Registro não é permitido` ao setar PATH

```powershell
$p = [System.Environment]::GetEnvironmentVariable("PATH", "User")
[System.Environment]::SetEnvironmentVariable("PATH", $p + ";C:\Program Files\Wireshark", "User")
# Feche e reabra o PowerShell
```

---

## Ambiente de Referência

| Componente | Versão |
|---|---|
| OS | Windows 11 |
| Python | 3.13.12 |
| PyTorch | 2.11.0+cu128 |
| scikit-learn | última estável |
| pandas | última estável |
| numpy | última estável |
| pyshark | última estável |
| matplotlib | última estável |
| NVIDIA Driver | 591.86 |
| CUDA Runtime | 13.1 |
| GPU | NVIDIA GeForce RTX 4060 (8 GB VRAM) |
| TShark | 4.x (Wireshark) |
| RAM | 16 GB |

---

*Projeto desenvolvido para artigo técnico sobre análise paralela de tráfego de rede com comparativo de qualidade de detecção de anomalias CPU vs GPU.*
