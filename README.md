# Parallel PCAP Analysis

Projeto para análise de tráfego de rede em PCAP com foco em comparar o custo computacional da classificação de anomalias. A estrutura foi organizada para manter código, documentação e artefatos gerados separados, facilitando versionamento no Git.

## Estrutura

- feature_extractor.py: extração de features por fluxo com pyshark.
- preprocess_features.py: pré-processamento e cache das features.
- federated_train.py: treino federado em CPU com suporte a cache.
- gpu_train.py: benchmark de IsolationForest em GPU com suporte a cache.
- ae_train.py: benchmark de Autoencoder em PyTorch sobre o cache.
- plots_cpu_gpu_compare.py: comparação entre resultados de CPU e GPU.
- plots_pad_artigo.py: figura auxiliar para artigo.
- app_gpu_dashboard.py: dashboard Streamlit para upload, execução e visualização.

## Pastas

- docs/: instruções detalhadas e documentação complementar.
- data/pcaps/: local para capturas de rede usadas no pipeline.
- data/results/: local para caches, métricas e saídas geradas.

## Execução local

Crie as pastas do projeto, se ainda não existirem:

```powershell
New-Item -ItemType Directory -Path '.\data\pcaps' -Force
New-Item -ItemType Directory -Path '.\data\results' -Force
```

Instale as dependências principais com:

```bash
pip install -r requirements.txt
```

Se for usar a versão com GPU, instale também:

```bash
pip install -r requirements-gpu.txt
```

Observação: neste ambiente, `requirements-gpu.txt` instala apenas os pacotes de GPU com wheel compatível. O treino em `gpu_train.py` faz fallback automático para CPU quando `cuML` não estiver disponível.

Para Autoencoder, instale PyTorch conforme seu ambiente e versão de CUDA.

Para extração de features, este projeto depende do TShark. No Windows, instale o Wireshark e verifique se `tshark.exe` está no PATH, ou defina `TSHARK_PATH` com o caminho completo do executável.

## Dependências

## Fluxo recomendado

1. Gerar o cache de features:

```bash
python preprocess_features.py --shards ./data/pcaps/*.pcapng --outdir ./data/results --cache-file features_cache.pkl
```

1. Executar os benchmarks usando o cache:

```bash
python gpu_train.py --cache-file ./data/results/features_cache.pkl --outdir ./data/results --outfile gpu_results.pkl
python federated_train.py --cache-file ./data/results/features_cache.pkl --outdir ./data/results --outfile cpu_federated_results.pkl --rounds 6 --workers 4
python ae_train.py --cache-file ./data/results/features_cache.pkl --outdir ./data/results --outfile ae_results.pkl --epochs 20
```

1. Gerar os gráficos comparativos:

```bash
python plots_cpu_gpu_compare.py --cpu-results ./data/results/cpu_federated_results.pkl --gpu-results ./data/results/gpu_results.pkl --outdir ./data/results --basename cpu_gpu_comparison
```

## Dashboard

Inicie a interface com:

```bash
streamlit run app_gpu_dashboard.py
```

O dashboard organiza o fluxo em abas para hardware, PCAPs, treino CPU, benchmark GPU, métodos e gráficos.
