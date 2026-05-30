# Parallel PCAP Analysis

Projeto para análise de tráfego de rede em PCAP com foco em comparar o custo computacional da classificação de anomalias. A estrutura foi organizada para manter código, documentação e artefatos gerados separados, facilitando versionamento no Git.

## Estrutura

- feature_extractor.py: extração de features por fluxo com pyshark.
- preprocess_features.py: pré-processamento e cache das features.
- cpu_train.py: benchmark de Isolation Forest em CPU.
- federated_train.py: benchmark federado de Isolation Forest em CPU.
- gpu_train.py: benchmark de Autoencoder em GPU com PyTorch.
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

Observação: a etapa GPU usa PyTorch e roda em `cuda` quando disponível.

Para GPU, o pacote `requirements-gpu.txt` inclui PyTorch.

Para extração de features, este projeto depende do TShark. No Windows, instale o Wireshark e verifique se `tshark.exe` está no PATH, ou defina `TSHARK_PATH` com o caminho completo do executável.

Cada etapa também grava e reaproveita um log separado em `data/results/pipeline_steps.txt` por padrão. Se quiser outro destino, use `--log-file`.

## Deploy automatico para servidor remoto

O script [sync_local_server.ps1](sync_local_server.ps1) agora aceita um modo de deploy remoto com SSH/SCP e opcao de monitoramento continuo.

Exemplo de uso:

```powershell
.
\sync_local_server.ps1 -Watch -SshKeyPath C:\Users\voce\.ssh\id_ed25519
```

O script já vem com `172.16.3.103`, `LABHUAWEI\dernier.bruno` e `C:\Users\dernier.bruno\parallelpcapanalysis-main_huawei` como padrões. Se voce nao quiser modo continuo, remova `-Watch` para fazer apenas uma sincronizacao. O script copia os arquivos listados para o servidor remoto sempre que detecta mudancas.

Requisitos:

- `ssh` e `scp` disponiveis no PATH do Windows.
- Acesso por chave ou outra autenticacao sem prompt interativo.
- O diretorio remoto precisa ser gravavel pelo usuario SSH.

## Fluxo recomendado

1. Gerar o cache de features:

```bash
python preprocess_features.py --shards ./data/pcaps/*.pcapng --outdir ./data/results --cache-file features_cache.pkl
```

1. Executar os benchmarks usando o cache:

```bash
python cpu_train.py --cache-file ./data/results/features_cache.pkl --outdir ./data/results --outfile cpu_results.pkl
python federated_train.py --cache-file ./data/results/features_cache.pkl --outdir ./data/results --outfile cpu_federated_results.pkl
python gpu_train.py --cache-file ./data/results/features_cache.pkl --outdir ./data/results --outfile gpu_results.pkl
```

1. Gerar os gráficos comparativos:

```bash
python plots_cpu_gpu_compare.py --cpu-results ./data/results/cpu_results.pkl --gpu-results ./data/results/gpu_results.pkl --outdir ./data/results --basename cpu_gpu_comparison
```

## Treino federado

Se você quiser rodar o federado isoladamente, use:

```bash
python federated_train.py --cache-file ./data/results/features_cache.pkl --outdir ./data/results --outfile cpu_federated_results.pkl --rounds 6 --workers 4
```

Esse fluxo continua usando Isolation Forest na CPU, mas com agregação federada entre shards.

## Dashboard

Inicie a interface com:

```bash
streamlit run app_gpu_dashboard.py
```

O dashboard organiza o fluxo em abas para hardware, PCAPs, treino CPU, Autoencoder GPU, métodos e gráficos.
