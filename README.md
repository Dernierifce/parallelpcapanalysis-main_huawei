# Anomaly Detection Network IFCE26

Pipeline Python para extrair features de fluxos de rede a partir de arquivos `.pcapng` e comparar métodos de detecção de anomalias em CPU e GPU.

O fluxo atual é centrado em dois arquivos:

- `feature_extractor.py`: lê arquivos `.pcapng` via PyShark/TShark e gera 22 features por fluxo.
- `anomaly_compare.py`: executa a extração, cria ou reutiliza cache, separa treino/teste, roda Autoencoder e K-Means em CPU/GPU, e salva um TXT detalhado com todas as etapas e um relatório visual simplificado.

## Estrutura Atual

```text
Anomaly_Detection_Network-IFCE26/
├── anomaly_compare.py
├── feature_extractor.py
├── requirements.txt
├── requirements-gpu.txt
└── data/
    └── pcaps/
```

Coloque os arquivos `.pcapng` que serão analisados em `data/pcaps/`.

## Requisitos

- Python 3.10+
- Wireshark/TShark instalado
- GPU NVIDIA com CUDA, opcional, para os testes com `--gpu`

Dependências principais:

```powershell
python -m pip install -r requirements.txt
```

Dependência opcional de GPU:

```powershell
python -m pip install -r requirements-gpu.txt
```

No Windows, confirme se o TShark está disponível:

```powershell
tshark -v
```

Se necessário, informe o caminho manualmente:

```powershell
$env:TSHARK_PATH = "C:\Program Files\Wireshark\tshark.exe"
```

## Uso

### Rodar o pipeline completo

Executa a extração dos `.pcapng`, cria `features_cache.pkl` e roda os comparativos:

```powershell
python anomaly_compare.py
```

Com GPU:

```powershell
python anomaly_compare.py --gpu
```

### Reutilizar cache já gerado

Use quando `data/results/features_cache.pkl` já existir:

```powershell
python anomaly_compare.py --skip-extraction
```

Com GPU:

```powershell
python anomaly_compare.py --skip-extraction --gpu
```

### Ajustar tamanho e parâmetros do benchmark

```powershell
python anomaly_compare.py --skip-extraction --gpu --sample-size 20000 --ae-epochs 20 --kmeans-clusters 10
```

Usar todas as amostras do cache:

```powershell
python anomaly_compare.py --skip-extraction --sample-size 0
```

Salvar resultados em outro diretório:

```powershell
python anomaly_compare.py --outdir .\data\results\exp_01
```

Reservar uma fração diferente para teste:

```powershell
python anomaly_compare.py --skip-extraction --test-size 0.25
```

### Extrair features de um único arquivo

Para testar somente a extração:

```powershell
python feature_extractor.py .\data\pcaps\arquivo.pcapng
```

## Parâmetros do `anomaly_compare.py`

| Parâmetro | Padrão | Descrição |
|---|---:|---|
| `--outdir` | `data/results` | Diretório de saída |
| `--cache-file` | `data/results/features_cache.pkl` | Arquivo de cache das features |
| `--skip-extraction` | `False` | Pula a extração e usa o cache existente |
| `--gpu` | `False` | Habilita Autoencoder e K-Means em GPU quando CUDA estiver disponível |
| `--sample-size` | `5000` | Número de amostras usadas no benchmark; `0` usa todas |
| `--ae-epochs` | `12` | Épocas do Autoencoder |
| `--ae-batch-size` | `256` | Batch size do Autoencoder |
| `--ae-latent-dim` | `8` | Dimensão latente do Autoencoder |
| `--ae-lr` | `0.001` | Taxa de aprendizado do Autoencoder |
| `--kmeans-clusters` | `8` | Número de clusters do K-Means |
| `--kmeans-max-iter` | `300` | Iterações máximas do K-Means |
| `--test-size` | `0.30` | Fração dos dados reservada para teste |

## Features Extraídas

`feature_extractor.py` agrega pacotes por fluxo de 5-tupla:

```text
(src_ip, dst_ip, src_port, dst_port, protocolo)
```

Por padrão, IPs são anonimizados com SHA-256 truncado.

As 22 features geradas são:

| # | Feature | Descrição |
|---:|---|---|
| 1 | `duration` | Duração do fluxo em segundos |
| 2 | `proto` | `1` para TCP, `0` para outros protocolos |
| 3 | `src_port` | Porta de origem |
| 4 | `dst_port` | Porta de destino |
| 5 | `pkt_count` | Quantidade de pacotes |
| 6 | `byte_count` | Total de bytes |
| 7 | `mean_pkt_size` | Tamanho médio dos pacotes |
| 8 | `std_pkt_size` | Desvio padrão do tamanho dos pacotes |
| 9 | `mean_iat` | Média do intervalo entre pacotes |
| 10 | `std_iat` | Desvio padrão do intervalo entre pacotes |
| 11 | `min_iat` | Menor intervalo entre pacotes |
| 12 | `max_iat` | Maior intervalo entre pacotes |
| 13 | `flag_syn` | Contagem de flags TCP SYN |
| 14 | `flag_fin` | Contagem de flags TCP FIN |
| 15 | `flag_rst` | Contagem de flags TCP RST |
| 16 | `flag_psh` | Contagem de flags TCP PSH |
| 17 | `fwd_pkt_count` | Pacotes no sentido registrado do fluxo |
| 18 | `bwd_pkt_count` | Pacotes no sentido reverso |
| 19 | `fwd_byte_ratio` | Razão de bytes no sentido registrado |
| 20 | `is_port_well_known` | `1` se `dst_port < 1024` |
| 21 | `is_ephemeral_src` | `1` se `src_port > 49151` |
| 22 | `bytes_per_pkt` | Média de bytes por pacote |

## Saídas Geradas

Por padrão, os resultados ficam em `data/results/`:

| Arquivo | Descrição |
|---|---|
| `features_cache.pkl` | Cache com `X_scaled`, `X`, colunas e estatísticas dos arquivos processados |
| `anomaly_compare.txt` | Arquivo principal com log, descrição das etapas, parâmetros essenciais e detalhes por método |
| `anomaly_report.png` | Relatório visual simplificado, com fundo branco e indicadores essenciais |

## Métodos Comparados

Antes de treinar os métodos, o `anomaly_compare.py` divide as amostras em treino e teste com seed fixa (`random_state=42`). O threshold p95 é calculado somente no conjunto de treino e a detecção de anomalias é avaliada no conjunto de teste, evitando data leakage.

### Autoencoder

Implementado em PyTorch. Treina com o conjunto de treino, usa erro de reconstrução como score de anomalia e aplica no teste o threshold p95 calculado sobre os erros de treino.

### K-Means CPU

Implementado com `scikit-learn`, usando `KMeans(n_init=10, random_state=42)`. O modelo é ajustado no conjunto de treino e avaliado pelas distâncias ao centroide mais próximo no teste.

### K-Means GPU

Implementação própria em PyTorch, usando `torch.cdist` e inicialização estilo k-means++ na GPU. Também treina somente no conjunto de treino, avalia no teste e não depende de cuML/RAPIDS.

## Solução de Problemas

### `TShark não foi encontrado`

Instale o Wireshark com o componente TShark e garanta que `tshark.exe` esteja no PATH, ou defina:

```powershell
$env:TSHARK_PATH = "C:\Program Files\Wireshark\tshark.exe"
```

### `Cache não encontrado`

Esse erro aparece ao usar `--skip-extraction` sem ter gerado o cache antes.

Rode primeiro:

```powershell
python anomaly_compare.py
```

### CUDA indisponível

Verifique se o PyTorch reconhece a GPU:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

Se retornar `False`, instale uma versão do PyTorch compatível com seu driver/CUDA.
