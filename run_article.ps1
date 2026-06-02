param(
    [string]$CacheFile = ".\data\results\features_cache.pkl",
    [string]$Outdir = ".",
    [int]$SampleSize = 5000,
    [int]$SyntheticSamples = 6000,
    [int]$SyntheticFeatures = 16,
    [int]$AeEpochs = 12,
    [int]$AeBatchSize = 256,
    [int]$AeLatentDim = 8,
    [int]$KMeansClusters = 8,
    [int]$KMeansMaxIter = 300,
    [switch]$RunGpuAutoencoder,
    [switch]$RunGpuKMeans
)

$scriptArgs = @(
    "scripts/benchmark_article.py",
    "--cache-file", $CacheFile,
    "--outdir", $Outdir,
    "--sample-size", $SampleSize,
    "--synthetic-samples", $SyntheticSamples,
    "--synthetic-features", $SyntheticFeatures,
    "--ae-epochs", $AeEpochs,
    "--ae-batch-size", $AeBatchSize,
    "--ae-latent-dim", $AeLatentDim,
    "--kmeans-clusters", $KMeansClusters,
    "--kmeans-max-iter", $KMeansMaxIter
)

if ($RunGpuAutoencoder) {
    $scriptArgs += "--run-gpu-autoencoder"
}

if ($RunGpuKMeans) {
    $scriptArgs += "--run-gpu-kmeans"
}

python @scriptArgs
