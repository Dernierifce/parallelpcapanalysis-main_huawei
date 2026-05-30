param(
    [string]$TargetPath = "C:\Users\dernier.bruno\parallelpcapanalysis-main_huawei",
    [switch]$UseGitPull
)

$sourcePath = Split-Path -Parent $MyInvocation.MyCommand.Path
$filesToSync = @(
    ".gitignore",
    "README.md",
    "app_gpu_dashboard.py",
    "cpu_train.py",
    "feature_extractor.py",
    "federated_train.py",
    "gpu_train.py",
    "log_utils.py",
    "plots_cpu_gpu_compare.py",
    "preprocess_features.py",
    "requirements.txt",
    "requirements-gpu.txt",
    "docs/INSTRUCOES_NOVO_PIPELINE.md"
)

if (-not (Test-Path $TargetPath)) {
    throw "Destino não encontrado: $TargetPath"
}

if ($UseGitPull -or (Test-Path (Join-Path $TargetPath ".git"))) {
    Write-Host "Sincronizando via git pull em $TargetPath"
    git -C $TargetPath status
    git -C $TargetPath pull origin main
    exit $LASTEXITCODE
}

Write-Host "Sincronizando por cópia em $TargetPath"
foreach ($relativeFile in $filesToSync) {
    $sourceFile = Join-Path $sourcePath $relativeFile
    $targetFile = Join-Path $TargetPath $relativeFile
    $targetFolder = Split-Path -Parent $targetFile

    if (-not (Test-Path $sourceFile)) {
        Write-Warning "Arquivo ausente na origem: $relativeFile"
        continue
    }

    if (-not (Test-Path $targetFolder)) {
        New-Item -ItemType Directory -Path $targetFolder -Force | Out-Null
    }

    Copy-Item -Path $sourceFile -Destination $targetFile -Force
    Write-Host "Copiado: $relativeFile"
}

Write-Host "Sincronização concluída."