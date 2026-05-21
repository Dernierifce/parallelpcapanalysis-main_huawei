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

## Dependências

Instale as dependências principais com:

```bash
pip install -r requirements.txt
```

Se for usar a versão com GPU/cuML, instale também:

```bash
pip install -r requirements-gpu.txt
```

Para Autoencoder, instale PyTorch conforme seu ambiente e versão de CUDA.

## Fluxo recomendado

1. Gerar o cache de features:

```bash
python preprocess_features.py --shards /data/pcaps/*.pcapng --outdir /data/results --cache-file features_cache.pkl
```

2. Executar os benchmarks usando o cache:

```bash
python gpu_train.py --cache-file /data/results/features_cache.pkl --outdir /data/results --outfile gpu_results.pkl
python federated_train.py --cache-file /data/results/features_cache.pkl --outdir /data/results --outfile cpu_federated_results.pkl --rounds 6 --workers 4
python ae_train.py --cache-file /data/results/features_cache.pkl --outdir /data/results --outfile ae_results.pkl --epochs 20
```

3. Gerar os gráficos comparativos:

```bash
python plots_cpu_gpu_compare.py --cpu-results /data/results/cpu_federated_results.pkl --gpu-results /data/results/gpu_results.pkl --outdir /data/results --basename cpu_gpu_comparison
```

## Dashboard

Inicie a interface com:

```bash
streamlit run app_gpu_dashboard.py
```

O dashboard organiza o fluxo em abas para hardware, PCAPs, treino CPU, benchmark GPU, métodos e gráficos.

## Documentação adicional

Consulte [docs/INSTRUCOES_NOVO_PIPELINE.md](docs/INSTRUCOES_NOVO_PIPELINE.md) para o fluxo detalhado do pipeline com cache.

## Artefatos gerados

Os arquivos gerados em tempo de execução devem ficar fora do controle de versão. O repositório já ignora caches, logs, modelos serializados, imagens de resultado e capturas PCAP locais.

## Docker

Para build sem pacotes GPU, use:

```bash
docker compose build --build-arg INSTALL_GPU_PACKAGES=0
```

## Observação

A comparação principal usa classification_s, que mede treino + inferência. Quando o cache é usado, a extração fica fora da métrica para permitir comparação mais justa entre métodos.
