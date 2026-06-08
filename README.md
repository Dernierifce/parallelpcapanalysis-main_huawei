# Parallel PCAP Analysis

> Pipeline completo de análise de tráfego de rede com comparativo de desempenho **CPU vs GPU** em algoritmos de detecção de anomalias. Extrai features estatísticas de fluxo a partir de capturas `.pcapng`, normaliza os dados e executa benchmarks comparativos com geração automática de relatório visual e log detalhado.

---

## Índice

1. [Visão Geral](#visão-geral)
2. [Resultados Obtidos](#resultados-obtidos)
3. [Requisitos de Sistema](#requisitos-de-sistema)
4. [Instalação](#instalação)
5. [Estrutura do Projeto](#estrutura-do-projeto)
6. [Uso — Comandos Principais](#uso--comandos-principais)
7. [Parâmetros de Execução](#parâmetros-de-execução)
8. [Pipeline Detalhado](#pipeline-detalhado)
9. [Features Extraídas](#features-extraídas)
10. [Algoritmos e Arquiteturas](#algoritmos-e-arquiteturas)
11. [Saídas Geradas](#saídas-geradas)
12. [Histórico de Execuções](#histórico-de-execuções)
13. [Configurações Avançadas](#configurações-avançadas)
14. [Solução de Problemas](#solução-de-problemas)
15. [Ambiente de Referência](#ambiente-de-referência)

---

## Visão Geral

O projeto implementa um pipeline em quatro etapas para análise de tráfego de rede capturado em produção:

```
Arquivos .pcapng
      │
      ▼
feature_extractor.py  ──►  22 features por fluxo (5-tupla)
      │
      ▼
StandardScaler  ──►  features_cache.pkl
      │
      ├──► cpu_train.py          (Isolation Forest — CPU)
      ├──► gpu_train.py          (Autoencoder PyTorch — CPU/GPU)
      └──► benchmark_article.py  (Autoencoder + K-Means — CPU vs GPU)
                │
                ▼
         run_pipeline.py  ──►  pipeline.log + pipeline_report.png
```

**Três modos de execução:**

| Modo | Comando | Descrição |
|---|---|---|
| Pipeline completo | `python run_pipeline.py` | Extração + benchmark + relatório |
| Somente benchmark | `python run_pipeline.py --skip-extraction` | Usa cache existente |
| Benchmark rápido | `.\run_article.ps1` | Wrapper PowerShell para artigo |

---

## Resultados Obtidos

Execução de referência realizada em **03/06/2026** com tráfego real capturado em **14/04/2026**.

### Extração de Features

| Shard | Tamanho | Fluxos | Tempo | Taxa |
|---|---|---|---|---|
| shard\_00091\_20260414170212 | 1.192 MB | 40.287 | 77,4 min | ~9 fluxos/s |
| shard\_00092\_20260414170229 | 1.192 MB | 38.286 | 69,4 min | ~9 fluxos/s |
| shard\_00093\_20260414170247 | 1.192 MB | 36.294 | 67,3 min | ~9 fluxos/s |
| shard\_00094\_20260414170303 | 1.004 MB | 31.832 | 59,4 min | ~9 fluxos/s |
| **TOTAL** | **4.580 MB** | **146.699** | **273,5 min** | — |

### Benchmark — Autoencoder e K-Means

Condições: 5.000 amostras, k=8 clusters, 12 épocas, latent dim=8, RTX 4060.

| Método | Hardware | Treino | Inferência | Total | Speedup |
|---|---|---|---|---|---|
| Autoencoder | CPU | 0,553s | 0,023s | 0,576s | referência |
| Autoencoder | GPU (RTX 4060) | 0,734s | 0,042s | 0,776s | **0,74x** ⚠ |
| K-Means | CPU (sklearn) | 2,437s | 0,000s | 2,437s | referência |
| K-Means | GPU (PyTorch) | 0,144s | 0,001s | 0,145s | **16,84x** ✓ |

> ⚠ **Autoencoder GPU (0,74x):** overhead de inicialização CUDA domina com amostras pequenas (5k). Esperado superar CPU a partir de ~20k–50k amostras.
>
> ✓ **K-Means GPU (16,84x):** algoritmo Lloyd vetorizado com kmeans++ na GPU via PyTorch puro. Ganho expressivo mesmo com amostra pequena devido à natureza altamente paralelizável do algoritmo.

### Convergência do Autoencoder (CPU, 12 épocas)

| Época | MSE Loss |
|---|---|
| 1 | 0,877395 |
| 2 | 0,839093 |
| 4 | 0,613668 |
| 6 | 0,415520 |
| 8 | 0,284777 |
| 10 | 0,225393 |
| 12 | **0,174578** |

### Distribuição de Clusters K-Means (k=8, CPU)

| Cluster | Amostras | % |
|---|---|---|
| C0 | 365 | 7,3% |
| C1 | 2.060 | 41,2% |
| C2 | 97 | 1,9% |
| C3 | 1.538 | 30,8% |
| C4 | 5 | 0,1% |
| C5 | 329 | 6,6% |
| C6 | 583 | 11,7% |
| C7 | 23 | 0,5% |

Inércia final: **39.423,34** | Iterações reais: **12 / 300**

---

## Requisitos de Sistema

### Hardware

| Componente | Mínimo | Recomendado (testado) |
|---|---|---|
| CPU | 4 cores | Intel/AMD moderno |
| RAM | 8 GB | 16 GB+ |
| Armazenamento | 10 GB livres | SSD (extrações longas) |
| GPU | — | NVIDIA RTX 4060 (8 GB VRAM) |
| Driver NVIDIA | — | 591.86+ |

### Software

| Componente | Versão mínima | Versão testada |
|---|---|---|
| Python | 3.10+ | 3.13.12 |
| Windows | 10 | 11 |
| Wireshark/TShark | 4.0 | 4.x |
| CUDA Toolkit | 12.1 | 13.1 (cu128) |
| PyTorch | 2.0 | 2.11.0+cu128 |

### Dependências Python — `requirements.txt`

```
pyshark       # leitura de pcapng via TShark
pandas        # manipulação de DataFrames
numpy         # operações numéricas
scikit-learn  # StandardScaler, KMeans, IsolationForest
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

> **Nota:** cuML (RAPIDS) **não é necessário** no Windows. O K-Means GPU usa implementação PyTorch pura incluída no `run_pipeline.py`.

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
# Remova versão CPU se instalada
pip uninstall torch torchvision torchaudio -y

# Instale com CUDA 12.8 (compatível com Python 3.13 e driver 591+)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Verifique:

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
# Esperado: CUDA: True | GPU: NVIDIA GeForce RTX 4060
```

> **Atenção:** Para Python 3.13, use obrigatoriamente `cu128`. O índice `cu121` não tem pacotes para esta versão.

### 4. Instale o Wireshark / TShark

Baixe em https://www.wireshark.org/download.html e marque durante a instalação:
- ✅ **TShark**
- ✅ **Add Wireshark to the system PATH**

Adicione ao PATH da sessão atual:

```powershell
$env:PATH += ";C:\Program Files\Wireshark"
tshark -v   # deve retornar a versão
```

Para tornar permanente (sem permissão de administrador):

```powershell
$p = [System.Environment]::GetEnvironmentVariable("PATH", "User")
[System.Environment]::SetEnvironmentVariable("PATH", $p + ";C:\Program Files\Wireshark", "User")
# Feche e reabra o PowerShell
```

### 5. Copie `run_pipeline.py` e `log_utils.py` para a raiz do projeto

```powershell
# Confirme que ambos estão na raiz junto com feature_extractor.py
ls *.py
```

---

## Estrutura do Projeto

```
parallelpcapanalysis-main_huawei/
│
├── feature_extractor.py            # Extração de features de fluxo via pyshark/TShark
├── cpu_train.py                    # Benchmark Isolation Forest — CPU (scikit-learn)
├── gpu_train.py                    # Benchmark Autoencoder — CPU/GPU (PyTorch)
├── benchmark_article.py            # Benchmark Autoencoder + K-Means para artigo
├── run_pipeline.py                 # Orquestrador completo: extração + benchmark + log + relatório PNG
├── log_utils.py                    # Utilitários de logging (setup_run_logging, emit_report)
├── plots_pad_artigo.py             # Geração de plots para publicação do artigo
├── run_article.ps1                 # Script PowerShell para executar benchmark do artigo
├── requirements.txt                # Dependências Python (CPU)
├── requirements-gpu.txt            # Dependências adicionais GPU (cupy)
├── README.md                       # Este arquivo
│
├── data/
│   ├── pcaps/
│   │   ├── shard__00091_20260414170212.pcapng   # 1,19 GB — tráfego real
│   │   ├── shard__00092_20260414170229.pcapng   # 1,19 GB
│   │   ├── shard__00093_20260414170247.pcapng   # 1,19 GB
│   │   └── shard__00094_20260414170303.pcapng   # 1,00 GB
│   └── results/
│       ├── features_cache.pkl       # Cache de features normalizadas (~37 MB, gerado)
│       ├── pipeline.log             # Log detalhado de execução (gerado)
│       ├── pipeline_report.png      # Relatório visual automático (gerado)
│       ├── benchmark_results.json   # Resultados completos em JSON (gerado)
│       ├── benchmark_results.csv    # Resultados em CSV (gerado)
│       ├── teste 1/                 # Execução de referência — somente CPU
│       └── teste 2/                 # Execução com GPU habilitada
│
├── scripts/
│   └── benchmark_article.py        # Cópia do benchmark para uso via run_article.ps1
│
├── figures/                        # Gráficos gerados pelo run_article.ps1
│   ├── train_time.png
│   ├── inference_time.png
│   └── speedup.png
│
├── results/                        # Saídas do run_article.ps1 (raiz)
│   ├── article_table.md
│   ├── article_table.csv
│   └── benchmark_results.json
│
└── docs/
    └── INSTRUCOES_NOVO_PIPELINE.md  # Guia de uso do run_pipeline.py
```

---

## Uso — Comandos Principais

### Pipeline completo (extração + benchmark + relatório)

```powershell
$env:PATH += ";C:\Program Files\Wireshark"
python run_pipeline.py --run-gpu-autoencoder --run-gpu-kmeans
```

> ⚠️ A extração dos 4 shards (~4,6 GB) leva aproximadamente **4,5 horas**. Os resultados ficam em `data/results/`.

### Pular extração — usar cache existente

```powershell
python run_pipeline.py --skip-extraction --run-gpu-autoencoder --run-gpu-kmeans
```

### Somente CPU (sem flags de GPU)

```powershell
python run_pipeline.py --skip-extraction
```

### Benchmark do artigo via PowerShell

```powershell
# Sem GPU
.\run_article.ps1

# Com GPU
.\run_article.ps1 -RunGpuAutoencoder -RunGpuKMeans

# Com parâmetros customizados
.\run_article.ps1 -RunGpuAutoencoder -RunGpuKMeans -SampleSize 20000 -AeEpochs 20 -KMeansClusters 10
```

### Benchmark Isolation Forest (CPU)

```powershell
python cpu_train.py --cache-file .\data\results\features_cache.pkl --outdir .\data\results
```

### Benchmark Autoencoder GPU direto

```powershell
python gpu_train.py --cache-file .\data\results\features_cache.pkl --outdir .\data\results --epochs 20 --batch-size 1024 --latent-dim 16
```

### Extração de um único shard (teste)

```powershell
python feature_extractor.py ".\data\pcaps\shard__00091_20260414170212.pcapng"
```

---

## Parâmetros de Execução

### `run_pipeline.py`

| Parâmetro | Padrão | Descrição |
|---|---|---|
| `--outdir` | `./data/results` | Diretório de saída para todos os arquivos gerados |
| `--cache-file` | `./data/results/features_cache.pkl` | Caminho do cache de features |
| `--skip-extraction` | `False` | Pula extração e usa cache existente |
| `--sample-size` | `5000` | Número de amostras para benchmark (0 = todas) |
| `--ae-epochs` | `12` | Épocas de treinamento do Autoencoder |
| `--ae-batch-size` | `256` | Batch size do Autoencoder |
| `--ae-latent-dim` | `8` | Dimensão do espaço latente do Autoencoder |
| `--ae-lr` | `0.001` | Learning rate (Adam) do Autoencoder |
| `--kmeans-clusters` | `8` | Número de clusters K-Means (k) |
| `--kmeans-max-iter` | `300` | Iterações máximas K-Means |
| `--run-gpu-autoencoder` | `False` | Habilita Autoencoder na GPU (requer CUDA) |
| `--run-gpu-kmeans` | `False` | Habilita K-Means na GPU (PyTorch puro) |

### `run_article.ps1`

| Parâmetro | Padrão | Descrição |
|---|---|---|
| `-CacheFile` | `.\data\results\features_cache.pkl` | Cache de entrada |
| `-Outdir` | `.` | Diretório raiz de saída |
| `-SampleSize` | `5000` | Amostras para benchmark |
| `-SyntheticSamples` | `6000` | Amostras sintéticas quando não há cache |
| `-SyntheticFeatures` | `16` | Features sintéticas quando não há cache |
| `-AeEpochs` | `12` | Épocas do Autoencoder |
| `-AeBatchSize` | `256` | Batch size |
| `-AeLatentDim` | `8` | Dimensão latente |
| `-KMeansClusters` | `8` | Clusters K-Means |
| `-KMeansMaxIter` | `300` | Iterações máximas |
| `-RunGpuAutoencoder` | `False` | Habilita GPU no Autoencoder |
| `-RunGpuKMeans` | `False` | Habilita GPU no K-Means |

### `cpu_train.py`

| Parâmetro | Padrão | Descrição |
|---|---|---|
| `--cache-file` | `None` | Cache de features (preferencial) |
| `--shards` | `data/pcaps/*.pcapng` | Arquivos pcapng (modo legado) |
| `--outdir` | `data/results` | Diretório de saída |
| `--outfile` | `cpu_results.pkl` | Nome do arquivo de resultado |
| `--estimators` | `200` | Número de árvores do Isolation Forest |
| `--contamination` | `0.05` | Taxa esperada de anomalias (5%) |
| `--log-file` | `None` | Arquivo de log customizado |

### `gpu_train.py`

| Parâmetro | Padrão | Descrição |
|---|---|---|
| `--cache-file` | `None` | Cache de features (preferencial) |
| `--shards` | `data/pcaps/*.pcapng` | Arquivos pcapng (modo legado) |
| `--outdir` | `data/results` | Diretório de saída |
| `--outfile` | `gpu_results.pkl` | Nome do arquivo de resultado |
| `--epochs` | `20` | Épocas do Autoencoder |
| `--batch-size` | `1024` | Batch size |
| `--lr` | `0.001` | Learning rate |
| `--latent-dim` | `16` | Dimensão latente |
| `--anom-percentile` | `95.0` | Percentil para threshold de anomalia |
| `--log-file` | `None` | Arquivo de log customizado |

---

## Pipeline Detalhado

### Etapa 1 — Extração de Features (`feature_extractor.py`)

Lê cada arquivo `.pcapng` pacote a pacote via `pyshark`, agrega por **fluxo de 5-tupla** e calcula 22 features por fluxo.

**Identificação do fluxo:** `(src_ip_hash, dst_ip_hash, src_port, dst_port, proto)`

**Anonimização:** IPs são substituídos por SHA-256 truncado a 10 caracteres por padrão — conformidade com LGPD. Desative com `anonymize=False`.

**Detecção do TShark:** prioriza variável de ambiente `TSHARK_PATH`; fallback para PATH do sistema via `get_process_path()` do pyshark.

**Tolerância a falhas:** captura fluxos parciais em caso de crash do TShark via `TSharkCrashException`; emite `RuntimeWarning` e retorna dados já processados.

```
Taxa de extração medida: ~9 fluxos/s por arquivo de ~1,2 GB
Tempo por shard: 59–77 minutos
```

### Etapa 2 — Normalização e Cache

```python
# Estrutura do features_cache.pkl (~37 MB)
{
    "X_scaled": np.ndarray,   # shape (N, 22) — float32, normalizado
    "X":        np.ndarray,   # shape (N, 22) — float32, bruto
    "columns":  list[str],    # 22 nomes de features
    "shard_stats": list[dict] # metadados por shard (cpu_train/gpu_train)
}
```

`StandardScaler` é ajustado sobre todos os shards concatenados e aplicado globalmente.

### Etapa 3 — Benchmark

Métricas coletadas para cada método e hardware:

- `train_s` — tempo de treinamento (s)
- `infer_s` — tempo de inferência/predição (s)
- `classification_s` — train + infer
- `speedup` — `cpu_total / gpu_total`

**Sincronização CUDA:** `torch.cuda.synchronize()` antes e depois de cada medição garante precisão real do tempo GPU.

### Etapa 4 — Log e Relatório

`run_pipeline.py` gera ao final:
- `pipeline.log` — log estruturado com timestamps em cada evento
- `pipeline_report.png` — relatório visual com 6 painéis gerado por `matplotlib`

O relatório é gerado automaticamente via `plot_report(log_path, out_path)` que faz parse do `.log` e extrai todos os dados sem reprocessamento.

---

## Features Extraídas

22 features calculadas por fluxo de rede:

| # | Feature | Tipo | Descrição |
|---|---|---|---|
| 1 | `duration` | float | Duração total do fluxo em segundos |
| 2 | `proto` | int | Protocolo: 1=TCP, 0=outros |
| 3 | `src_port` | int | Porta de origem |
| 4 | `dst_port` | int | Porta de destino |
| 5 | `pkt_count` | int | Total de pacotes no fluxo |
| 6 | `byte_count` | int | Total de bytes transmitidos |
| 7 | `mean_pkt_size` | float | Tamanho médio dos pacotes (bytes) |
| 8 | `std_pkt_size` | float | Desvio padrão do tamanho dos pacotes |
| 9 | `mean_iat` | float | Inter-Arrival Time médio (s) |
| 10 | `std_iat` | float | Desvio padrão do IAT |
| 11 | `min_iat` | float | IAT mínimo |
| 12 | `max_iat` | float | IAT máximo |
| 13 | `flag_syn` | int | Contagem de pacotes com flag SYN |
| 14 | `flag_fin` | int | Contagem de pacotes com flag FIN |
| 15 | `flag_rst` | int | Contagem de pacotes com flag RST |
| 16 | `flag_psh` | int | Contagem de pacotes com flag PSH |
| 17 | `fwd_pkt_count` | int | Pacotes no sentido forward |
| 18 | `bwd_pkt_count` | int | Pacotes no sentido backward |
| 19 | `fwd_byte_ratio` | float | Razão de bytes forward / total |
| 20 | `is_port_well_known` | int | 1 se `dst_port < 1024` |
| 21 | `is_ephemeral_src` | int | 1 se `src_port > 49151` |
| 22 | `bytes_per_pkt` | float | Média de bytes por pacote |

---

## Algoritmos e Arquiteturas

### Autoencoder (PyTorch) — `gpu_train.py` e `run_pipeline.py`

Rede neural encoder-decoder simétrica para detecção de anomalias por erro de reconstrução:

```
Encoder:  Input(22) → Linear → ReLU → Linear → ReLU → Latent(8)
Decoder:  Latent(8) → Linear → ReLU → Linear → Output(22)

hidden_dim = max(32, n_features * 2) = 44
```

| Hiperparâmetro | Valor padrão |
|---|---|
| Otimizador | Adam |
| Learning rate | 0,001 |
| Loss | MSE |
| Épocas | 12 (run_pipeline) / 20 (gpu_train) |
| Batch size | 256 (run_pipeline) / 1024 (gpu_train) |
| Latent dim | 8 (run_pipeline) / 16 (gpu_train) |

**Detecção de anomalias:** amostras com erro de reconstrução acima do percentil 95 são classificadas como anômalas.

### K-Means GPU — PyTorch puro (`run_pipeline.py`)

Implementação própria do algoritmo Lloyd completamente vetorizada na GPU, sem dependência de cuML:

1. **Inicialização kmeans++** na GPU via `torch.multinomial` com distâncias calculadas por `torch.cdist`
2. **Loop de atualização** com detecção automática de convergência (`torch.equal` entre labels consecutivos)
3. **Sincronização CUDA** (`torch.cuda.synchronize()`) para medição precisa
4. Inércia calculada como soma das distâncias quadráticas mínimas ao centroide mais próximo

### K-Means CPU — scikit-learn (`run_pipeline.py` e `benchmark_article.py`)

```python
KMeans(n_clusters=8, max_iter=300, n_init=10, random_state=42)
```

### Isolation Forest — scikit-learn (`cpu_train.py`)

```python
IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1)
```

Utiliza todos os cores disponíveis (`n_jobs=-1`). Taxa de contaminação padrão de 5%.

---

## Saídas Geradas

### `run_pipeline.py`

| Arquivo | Local | Descrição |
|---|---|---|
| `pipeline.log` | `--outdir` | Log completo com timestamps de todas as etapas |
| `pipeline_report.png` | `--outdir` | Relatório visual com 6 painéis |
| `features_cache.pkl` | `--outdir` | Cache de features normalizadas (~37 MB) |
| `benchmark_results.json` | `--outdir` | Resultados + argumentos + métricas em JSON |
| `benchmark_results.csv` | `--outdir` | Resultados em CSV |

### `run_article.ps1` / `benchmark_article.py`

| Arquivo | Local | Descrição |
|---|---|---|
| `benchmark_results.json` | `results/` | Resultados completos |
| `article_table.md` | `results/` | Tabela em Markdown para artigo |
| `article_table.csv` | `results/` | Tabela em CSV |
| `train_time.png` | `figures/` | Gráfico de tempo de treino |
| `inference_time.png` | `figures/` | Gráfico de tempo de inferência |
| `speedup.png` | `figures/` | Gráfico de speedup relativo |

### `cpu_train.py` / `gpu_train.py`

| Arquivo | Descrição |
|---|---|
| `cpu_results.pkl` | Resultado completo: labels, scores, métricas, shard_stats |
| `gpu_results.pkl` | Resultado completo: errors, labels, threshold, métricas |
| `cpu_train_<timestamp>.log` | Log detalhado da execução |
| `gpu_train_<timestamp>.log` | Log detalhado da execução |

### Conteúdo do `pipeline.log`

```
PIPELINE INICIADO
  Data/hora | Python | PyTorch | CUDA | GPU | Outdir | Cache | Sample size

ETAPA 1 — EXTRAÇÃO DE FEATURES
  [Shard 1/4] Arquivo | Tamanho MB | Fluxos | Tempo (s e min) | Taxa fluxos/s
  ...
  Resumo: shards processados | total fluxos | tempo total | volume GB

CARREGANDO DADOS DO CACHE
  Total no cache | Amostras após sample

ETAPA 2 — BENCHMARK
  Autoencoder / CPU
    Épocas | Batch | Latent | LR | Amostras | Device
    [Autoencoder] Época N/12 | loss=X.XXXXXX  (uma linha por época)
    Treino | Inferência
  Autoencoder / GPU
    (idem + Speedup GPU vs CPU)
  K-Means / CPU
    Clusters | Max iter | Amostras | Device
    Iterações reais | Inércia final
    Cluster N : X amostras (Y%)  (uma linha por cluster)
    Treino | Inferência
  K-Means / GPU
    (idem + Speedup GPU vs CPU)

ETAPA 3 — COMPARAÇÃO DE MÉTODOS
  Tabela: Experimento | Hardware | Treino | Inferência | Total | Speedup | Status | Notas
  Speedups calculados por método

PIPELINE CONCLUÍDO
  Tempo total | Caminho do log
```

### Conteúdo do `pipeline_report.png`

| Painel | Conteúdo |
|---|---|
| **Tempo por Shard** | Barras com tempo de extração em minutos por pcapng |
| **Fluxos por Shard** | Barras com fluxos extraídos + total acumulado |
| **Convergência Autoencoder** | Curva MSE loss por época com box de métricas CPU/GPU/speedup |
| **Distribuição K-Means** | Barras agrupadas CPU vs GPU por cluster com inércia e iterações |
| **Tabela Comparativa** | Treino / Inferência / Total / Speedup de todos os métodos |
| **Timeline** | Linha do tempo horizontal da execução completa em minutos |

---

## Histórico de Execuções

| Execução | Data | GPU | Sample | AE Speedup | KM Speedup | Local |
|---|---|---|---|---|---|---|
| Teste 1 | 03/06/2026 | ✗ CPU only | 5.000 | — | — | `data/results/teste 1/` |
| Teste 2 | 03/06/2026 | ✓ RTX 4060 | 5.000 | 0,74x | 16,84x | `data/results/teste 2/` |
| **Referência** | **03/06/2026** | **✓ RTX 4060** | **5.000** | **0,74x** | **16,84x** | `data/results/` |

---

## Configurações Avançadas

### Aumentar amostra para speedups mais representativos

```powershell
python run_pipeline.py --skip-extraction --run-gpu-autoencoder --run-gpu-kmeans --sample-size 50000
```

> Com 50k+ amostras o Autoencoder GPU deve superar a CPU em treino.

### Ajustar hiperparâmetros do Autoencoder

```powershell
python run_pipeline.py --skip-extraction --run-gpu-autoencoder `
    --ae-epochs 30 --ae-batch-size 512 --ae-latent-dim 16 --ae-lr 0.0005
```

### Salvar em subdiretório por experimento

```powershell
python run_pipeline.py --skip-extraction --run-gpu-autoencoder --run-gpu-kmeans `
    --outdir .\data\results\experimento_50k --sample-size 50000
```

### Forçar TShark via variável de ambiente

```powershell
$env:TSHARK_PATH = "C:\Program Files\Wireshark\tshark.exe"
python run_pipeline.py
```

### Usar dados sintéticos (sem pcapng)

`benchmark_article.py` gera dados sintéticos automaticamente quando o cache não existe:

```powershell
python scripts\benchmark_article.py --outdir .\data\results --synthetic-samples 10000 --synthetic-features 22
```

---

## Solução de Problemas

### `TShark não foi encontrado`

```powershell
# Opção 1: adicionar ao PATH da sessão
$env:PATH += ";C:\Program Files\Wireshark"

# Opção 2: variável de ambiente direta
$env:TSHARK_PATH = "C:\Program Files\Wireshark\tshark.exe"

# Verificar se está acessível
tshark -v
```

### `No module named 'torch'` após desinstalação

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### `CUDA: False` com GPU NVIDIA presente

O PyTorch instalado é a versão CPU (`torch 2.x.x+cpu`). Reinstale:

```powershell
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.cuda.is_available())"
```

### `ERROR: Could not find a version` ao instalar PyTorch

O índice `cu121` não suporta Python 3.13. Use `cu128`:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

### `cuML unavailable` no K-Means GPU

cuML não está disponível no Windows via pip. Isso é esperado. O `run_pipeline.py` usa automaticamente a implementação **K-Means PyTorch puro** que produz resultados equivalentes com speedup comprovado (16,84x).

O `run_article.ps1` (via `benchmark_article.py`) registrará status `unavailable` — use `run_pipeline.py` para obter resultados GPU completos do K-Means.

### `benchmark_article.py: error: unrecognized arguments: --shards`

O `benchmark_article.py` não aceita `--shards`. Ele opera sobre cache já extraído. Fluxo correto:

```powershell
# 1. Extrair features (run_pipeline.py ou feature_extractor.py)
python run_pipeline.py   # gera features_cache.pkl

# 2. Rodar benchmark
.\run_article.ps1
```

### `can't open file benchmark_article.py`

O arquivo está em `scripts\`, não na raiz:

```powershell
python scripts\benchmark_article.py --cache-file .\data\results\features_cache.pkl --outdir .
# ou via wrapper:
.\run_article.ps1
```

### `Acesso ao Registro não é permitido` ao setar PATH

Use escopo de usuário (não requer admin):

```powershell
$p = [System.Environment]::GetEnvironmentVariable("PATH", "User")
[System.Environment]::SetEnvironmentVariable("PATH", $p + ";C:\Program Files\Wireshark", "User")
# Feche e reabra o PowerShell
```

### Cache corrompido ou features incompatíveis

```powershell
Remove-Item .\data\results\features_cache.pkl
python run_pipeline.py --run-gpu-autoencoder --run-gpu-kmeans
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

*Projeto desenvolvido para artigo técnico sobre análise paralela de tráfego de rede com comparativo CPU vs GPU.*
