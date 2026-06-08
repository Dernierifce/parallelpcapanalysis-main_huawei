| Experimento | Hardware | Treino | Inferencia | Classificacao | Speedup | Status | Notas |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Autoencoder | CPU | 0.65s | 0.02s | 0.67s | 1.00x | ok | threshold sample=5000 |
| Autoencoder | GPU | 0.75s | 0.04s | 0.80s | 0.85x | ok | threshold sample=5000 |
| K-Means | CPU | 1.71s | 0.00s | 1.71s | 1.00x | ok | labels=5000 |
| K-Means | GPU | n/a | n/a | n/a | n/a | unavailable | cuML unavailable: No module named 'cuml' |
