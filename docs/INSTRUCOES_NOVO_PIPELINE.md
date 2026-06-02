# Instruções do artigo simplificado

Este repositório está sendo reduzido para um artigo mais curto e convincente. A versão-alvo da pipeline terá apenas dois experimentos:

1. Autoencoder: CPU vs GPU.
2. K-Means: CPU com scikit-learn vs GPU com RAPIDS cuML.

O foco passa a ser comparar Deep Learning e Machine Learning clássico, sempre destacando o contraste entre CPU e GPU.

## Fluxo recomendado

### 1. Preparar os dados

Faça a extração e a organização das features uma única vez. A partir daí, reutilize o mesmo conjunto para todos os benchmarks.

### 2. Experimento 1: Autoencoder

- Treinar e avaliar em CPU.
- Treinar e avaliar em GPU.
- Registrar apenas treino, inferência e speedup.

### 3. Experimento 2: K-Means

- Executar a versão CPU com scikit-learn.
- Executar a versão GPU com RAPIDS cuML.
- Registrar os mesmos indicadores de tempo para comparação justa.

### 4. Gerar a saída do artigo

Concentre a saída final em uma tabela curta e em poucos gráficos:

- tempo de treino
- tempo de inferência
- tempo total de classificação
- speedup CPU/GPU

## Estrutura ideal de scripts

Quando a refatoração for feita, o projeto deve ficar com uma estrutura simples, sem pipeline federada ou etapas extras desnecessárias:

- um script para Autoencoder CPU
- um script para Autoencoder GPU
- um script para K-Means CPU
- um script para K-Means GPU
- um script para gerar a tabela e os gráficos finais

## Observação sobre o código atual

Os arquivos antigos da pipeline atual podem ser tratados como legado enquanto a refatoração não for concluída. A documentação principal deve seguir o recorte acima para manter o artigo enxuto.
