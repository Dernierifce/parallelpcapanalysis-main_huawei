# Parallel PCAP Analysis

Pipeline para análise de tráfego de rede em PCAP com:

- extração de features por fluxo,
- treinamento federado em CPU,
- benchmark de modelo em GPU para comparação,
- geração de gráficos,
- operação via dashboard Streamlit em container Docker.

## O que este projeto faz hoje

- `feature_extractor.py`: extrai features por fluxo (5-tupla) usando `pyshark`.
- `federated_train.py`: executa o fluxo federado em CPU com multiprocessing + FedAvg.
- `gpu_train.py`: executa benchmark de `IsolationForest` com backend GPU (`cuML`) em modo centralizado para comparação com os resultados de CPU.
- `plots_cpu_gpu_compare.py`: gera gráficos comparativos a partir dos arquivos `.pkl` de CPU e GPU.
- `plots_pad_artigo.py`: gera figura de artigo com dados simulados (não lê o `.pkl` real automaticamente).
- `app_gpu_dashboard.py`: interface Streamlit para upload, execução dos jobs e visualização de hardware.

## Dashboard Streamlit

O dashboard possui 5 abas:

- Hardware: inventário de CPU, RAM, disco e `nvidia-smi`.
- PCAPs: upload de arquivos `.pcap`/`.pcapng` para o volume persistente.
- Treino CPU Federado: executa `federated_train.py`.
- Benchmark GPU: executa `gpu_train.py` (com fallback opcional para CPU).
- Charts: executa `plots_cpu_gpu_compare.py` e exibe o `.png` gerado.

Observação importante:

- As execuções são síncronas: ao iniciar um treino pelo dashboard, a requisição fica ocupada até o fim do processo.

## Volumes e caminhos no container

O Compose cria dois volumes nomeados:

# Parallel PCAP Analysis

Pipeline para análise de tráfego de rede em PCAP com foco em medir e comparar o custo
computacional da classificação de anomalias (treino + inferência). O projeto agora suporta
pré-processamento (cache) das features para evitar reexecução custosa de I/O com `pyshark`.

Funcionalidades principais:

- extração de features por fluxo (cacheável),
- treinamento federado em CPU (modo `--cache-file` para medir só classificação),
- benchmark de `IsolationForest` em GPU (cuML) com modo cache,
- benchmark de Autoencoder (PyTorch) sobre cache,
- geração de gráficos comparativos,
- dashboard Streamlit com operação integrada (upload, cache, execuções multiprocessos).

## Arquivos relevantes (resumo)

- `feature_extractor.py`: extrai features por fluxo (5-tupla) usando `pyshark`.
- `preprocess_features.py`: novo — extrai features e salva em cache (`features_cache.pkl`).
- `federated_train.py`: suporta `--cache-file` (modo recomendado) para medir apenas classificação; mantém modo legado sem cache.
- `gpu_train.py`: suporta `--cache-file` (modo recomendado) e grava `classification_s` no resultado.
- `ae_train.py`: novo — benchmark simples de Autoencoder (PyTorch) sobre o cache.
- `plots_cpu_gpu_compare.py`: atualizado para priorizar comparação por `classification_s` quando disponível.
- `app_gpu_dashboard.py`: dashboard atualizado com aba `Methods` para pré-processar cache e executar múltiplos métodos automaticamente.

## Fluxo recomendado (rápido)

1. Gerar cache de features (executar uma vez):

```bash
python preprocess_features.py \
	--shards /data/pcaps/*.pcapng \
	--outdir /data/results \
	--cache-file features_cache.pkl
```

2. Executar benchmarks (usando cache — mede apenas classificação):

```bash
# GPU IsolationForest (usa cache)
python gpu_train.py --cache-file /data/results/features_cache.pkl --outdir /data/results --outfile gpu_results.pkl

# CPU federated (usa cache)
python federated_train.py --cache-file /data/results/features_cache.pkl --outdir /data/results --outfile cpu_federated_results.pkl --rounds 6 --workers 4

# Autoencoder (PyTorch)
python ae_train.py --cache-file /data/results/features_cache.pkl --outdir /data/results --outfile ae_results.pkl --epochs 20

Observação: o `ae_train.py` agora calcula erro de reconstrução por fluxo e salva thresholds e contagens adicionais no `.pkl`:

- `thresholds`: valores usados (`percentile` e `mean_3std`)
- `n_anomalies` / `anom_rate`: contagem/porcentagem baseada no percentil (`--anom-percentile`, padrão 95.0)
- `n_anomalies_mean3std` / `anom_rate_mean3std`: contagem/porcentagem baseada em `mean + 3*std`

Você pode ajustar o percentil via `--anom-percentile` (ex.: `--anom-percentile 99.0`).
```

3. Gerar gráficos comparativos (prioriza `classification_s`):

```bash
python plots_cpu_gpu_compare.py --cpu-results /data/results/cpu_federated_results.pkl --gpu-results /data/results/gpu_results.pkl --outdir /data/results --basename cpu_gpu_comparison
```

Observação: o `app_gpu_dashboard.py` facilita esses passos via interface (aba `Methods`).

## Saída e métricas

- Cada script gera um arquivo `.pkl` em `--outdir` com campos padronizados:
	- `mode`: "cache" ou "full"
	- `times`: inclui `train_s`, `infer_s` e `classification_s` (train+infer)
	- `n_flows`, `n_features`, `n_anomalies`, `anom_rate`
	- `scores` e metadados do método

- Métrica principal para comparação: `classification_s` (treino + inferência). Não inclui extração quando `--cache-file` é usado.

## Logs persistentes

O dashboard exibe saída em tempo real na área `Logs` (barra lateral) e também persiste todas as linhas de saída em um arquivo de log em `RESULTS_DIR/logs.txt` (por padrão `/data/results/logs.txt`).

Para acompanhar o log fora do dashboard em um container ou host, use:

```bash
tail -f /data/results/logs.txt
```

Dentro da interface, o histórico completo também é mantido em `st.session_state['app_logs']` e exibido na expander `Logs`.

## Dependências

Instale dependências principais:

```bash
pip install -r requirements.txt
```

Para treinamento de Autoencoders em GPU instale PyTorch conforme sua GPU/OS:

```bash
# Exemplo CPU-only
pip install torch

# Exemplo CUDA (exemplo, ajuste versão conforme ambiente)
pip install torch --index-url https://download.pytorch.org/whl/cu117
```

Se quiser o acelerador cuML para `gpu_train.py`, instale os pacotes listados em `requirements-gpu.txt` (container com CUDA).

## Dashboard Streamlit

Execute o dashboard e use a aba `Methods` para:

- salvar uploads em `PCAP_DIR` (por padrão `/data/pcaps`),
- gerar cache (`preprocess_features.py`) e apontar o arquivo de cache,
- selecionar e executar métodos (IsolationForest GPU/CPU federated, Autoencoder) sequencialmente,
- visualizar e baixar os `.pkl` gerados para análise e plotagem.

```bash
streamlit run app_gpu_dashboard.py
```

## Docker

As imagens/compose mantêm suporte; se usar GPUs no container, habilite CUDA e instale dependências GPU no build. Para uma versão leve sem pacotes GPU, use:

```bash
docker compose build --build-arg INSTALL_GPU_PACKAGES=0
```

## Notas finais

O foco principal deste rework é isolar o custo de classificação (treino + inferência) do custo de extração (I/O com `pyshark`). Use o `--cache-file` sempre que possível para obter comparações justas entre métodos.
