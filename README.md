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
- data/pcaps/: local para capturas de rede usadas no pipeline (ponto de montagem no container: /data/pcaps).
- data/results/: local para caches, métricas e saídas geradas (ponto de montagem no container: /data/results).

## Acesso aos dados

O `docker-compose.yml` do repositório foi configurado para usar bind mounts no host. No servidor onde os containers rodam (ex.: 172.16.3.103) os caminhos usados nesta implantação são:

- `C:/docker_shared/Resultados/pcaps` → montado em `/data/pcaps` dentro do container
- `C:/docker_shared/Resultados/results` → montado em `/data/results` dentro do container

Se você preferir usar caminhos relativos ao repositório (útil para desenvolvimento local), o compose também aceita `./data/pcaps:/data/pcaps` e `./data/results:/data/results`.

Criar as pastas e ajustar permissões no host (PowerShell, execute como Administrador se necessário):

```powershell
New-Item -ItemType Directory -Path 'C:\docker_shared\Resultados\pcaps' -Force
New-Item -ItemType Directory -Path 'C:\docker_shared\Resultados\results' -Force
# conceder acesso (exemplo para ambientes em Português)
icacls 'C:\docker_shared\Resultados' /grant "Todos:(OI)(CI)F" /T
```

Subir / reiniciar a stack (no diretório do repositório):

```powershell
cd 'C:\caminho\para\parallelpcapanalysis-main_huawei'
docker compose down --remove-orphans
docker compose up -d --build
```

Verificar status do container e os mounts:

```powershell
docker ps --filter name=parallelpcapanalysis-dashboard --format "table {{.Names}}\t{{.Status}}"
docker inspect parallelpcapanalysis-dashboard --format "{{json .Mounts}}"
```

Procure em `Mounts` os objetos com `Destination` `/data/pcaps` e `/data/results` e confirme que `Source` aponta para os caminhos `C:/docker_shared/Resultados/pcaps` e `C:/docker_shared/Resultados/results`.

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

## Docker

O compose do projeto suporta builds com e sem pacotes GPU. Por padrão o `Dockerfile` pode instalar dependências GPU quando o build-arg `INSTALL_GPU_PACKAGES=1` está ativo.

Build sem pacotes GPU:

```powershell
docker compose build --build-arg INSTALL_GPU_PACKAGES=0
```

Build com pacotes GPU (requer NVIDIA + WSL/Windows com suporte):

```powershell
docker compose build --build-arg INSTALL_GPU_PACKAGES=1
```

Se o container falhar ao iniciar com GPU, mensagens típicas são relacionadas ao `nvidia-container-cli` (ex.: "WSL environment detected but no adapters were found"). Para habilitar GPU corretamente veja a seção "Checklist GPU" abaixo.

## Portainer

Para atualizar automaticamente no Portainer:

1. Publique este repositório no GitHub.
2. No Portainer, crie a stack usando a opção "Git repository".
3. Aponte para este `docker-compose.yml` e use o branch `main`.
4. Ative o webhook da stack no Portainer.
5. Em cada `git push`, o webhook poderá acionar o redeploy no Portainer.

Observação: se a stack usar `build:` o Portainer recompõe a imagem a partir do repositório. Para deploys mais rápidos em produção é recomendado publicar a imagem num registry e usar `image:` no `docker-compose.yml`.

## Checklist GPU (se quiser usar aceleração)

1. Confirme drivers NVIDIA instalados no host:

```powershell
nvidia-smi
```

2. Se estiver em Windows com WSL2:
- Instale o driver NVIDIA com suporte a WSL (download no site NVIDIA).
- Habilite WSL2 e integração no Docker Desktop (Settings → Resources → WSL Integration).

3. Teste com um container base:

```powershell
docker run --gpus all --rm nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

4. Se o comando acima mostrar placas, reative `gpus: all` no `docker-compose.yml` e reinicie a stack.

Se receber erro do tipo "WSL environment detected but no adapters were found", verifique se o Docker Desktop está usando WSL2 e se o driver NVIDIA para WSL está instalado.
