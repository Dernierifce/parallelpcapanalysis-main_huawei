# parallelpcapanalysis-main_huawei

Projeto enxuto para um artigo com dois experimentos comparativos:

- Autoencoder: CPU vs GPU
- K-Means: CPU (scikit-learn) vs GPU (RAPIDS cuML)

O foco e medir tempo de treino, tempo de inferencia e speedup para comparar Deep Learning e Machine Learning classico em CPU e GPU.

## Execucao

Instale as dependencias principais:

```powershell
pip install -r requirements.txt
```

Se for usar GPU com Autoencoder, PyTorch precisa estar disponivel. Para o K-Means em GPU, use um ambiente com RAPIDS cuML compativel com sua stack CUDA.

Rode o artigo em um comando:

```powershell
.\run_article.ps1
```

Para incluir os testes de GPU quando o ambiente suportar:

```powershell
.\run_article.ps1 -RunGpuAutoencoder -RunGpuKMeans
```

## Saidas

- `results/benchmark_results.json`
- `results/article_table.md`
- `results/article_table.csv`
- `figures/train_time.png`
- `figures/inference_time.png`
- `figures/speedup.png`

## Estrutura

- `scripts/benchmark_article.py`: executa os benchmarks e gera tabela/figuras
- `run_article.ps1`: comando unico para rodar o artigo

## Observacao

Se o cache `data/results/features_cache.pkl` existir, o runner pode reutiliza-lo. Caso contrario, ele usa dados sinteticos para manter o fluxo executavel mesmo sem PCAPs.
