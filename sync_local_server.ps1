[CmdletBinding()]
param(
    [string]$TargetPath = "C:\Users\dernier.bruno\parallelpcapanalysis-main_huawei",
    [switch]$UseGitPull,
    [switch]$ForceClone,
    [switch]$Watch,
    [string]$RemoteHost = "172.16.3.103",
    [string]$RemoteUser = "LABHUAWEI\dernier.bruno",
    [int]$RemotePort = 22,
    [string]$RemotePath = "C:\Users\dernier.bruno\parallelpcapanalysis-main_huawei",
    [string]$SshKeyPath
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

function Test-RemoteMode {
    return -not [string]::IsNullOrWhiteSpace($RemoteHost)
}

function Get-RemoteTarget {
    if ([string]::IsNullOrWhiteSpace($RemoteUser)) {
        return $RemoteHost
    }

    return "$RemoteUser@$RemoteHost"
}

function Escape-RemoteShellArgument {
    param([string]$Value)
    # Escape single quotes for safe single-quoted PowerShell argument on remote side
    return "'" + ($Value -replace "'", "''") + "'"
}

function Join-RemotePath {
    param(
        [string]$BasePath,
        [string]$RelativePath
    )

    $normalizedBase = $BasePath.TrimEnd('\', '/')
    $normalizedRelative = $RelativePath -replace '\\', '/'
    return "$normalizedBase/$normalizedRelative"
}

function Get-RemoteParentPath {
    param([string]$Path)

    $normalized = $Path -replace '\\', '/'
    $lastSlash = $normalized.LastIndexOf('/')
    if ($lastSlash -lt 0) {
        return ""
    }

    return $normalized.Substring(0, $lastSlash)
}

function Convert-ToScpRemotePath {
    param([string]$Path)

    if ($Path -match '^[A-Za-z]:\\') {
        $drive = $Path.Substring(0, 1)
        $rest = $Path.Substring(2) -replace '\\', '/'
        return ("/" + $drive + ":" + $rest)
    }

    return ($Path -replace '\\', '/')
}

function Convert-ToRemoteShellPath {
    param([string]$Path)

    return ($Path -replace '/', '\\')
}

function Assert-CommandAvailable {
    param([string]$CommandName)

    if (-not (Get-Command $CommandName -ErrorAction SilentlyContinue)) {
        throw "$CommandName nao encontrado no PATH. Instale o cliente OpenSSH ou ajuste o ambiente antes de usar deploy remoto."
    }
}

function Invoke-RemoteCommand {
    param([string]$Command)

    Assert-CommandAvailable -CommandName "ssh"

    $sshArgs = @()
    if ($SshKeyPath) {
        $sshArgs += @("-i", $SshKeyPath)
    }
    if ($RemotePort -gt 0) {
        $sshArgs += @("-p", $RemotePort)
    }

    & ssh @sshArgs (Get-RemoteTarget) $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Falha ao executar comando remoto: $Command"
    }
}

function Invoke-RemoteCopy {
    param(
        [string]$SourceFile,
        [string]$DestinationFile
    )

    Assert-CommandAvailable -CommandName "scp"

    $scpArgs = @()
    if ($SshKeyPath) {
        $scpArgs += @("-i", $SshKeyPath)
    }
    if ($RemotePort -gt 0) {
        $scpArgs += @("-P", $RemotePort)
    }

    $remoteDestination = Convert-ToScpRemotePath -Path $DestinationFile
    & scp @scpArgs $SourceFile "$((Get-RemoteTarget)):$remoteDestination"
    if ($LASTEXITCODE -ne 0) {
        throw "Falha ao copiar $SourceFile para $DestinationFile"
    }
}

function Get-SourceSnapshot {
    $items = foreach ($relativeFile in $filesToSync) {
        $sourceFile = Join-Path $sourcePath $relativeFile
        if (Test-Path $sourceFile) {
            $item = Get-Item $sourceFile
            [PSCustomObject]@{
                Path = $relativeFile
                Length = $item.Length
                LastWriteTimeUtc = $item.LastWriteTimeUtc.Ticks
            }
        }
        else {
            [PSCustomObject]@{
                Path = $relativeFile
                Missing = $true
            }
        }
    }

    return ($items | ConvertTo-Json -Compress -Depth 3)
}

function Sync-LocalCopy {
    param([string]$DestinationPath)

    if (-not (Test-Path $DestinationPath)) {
        throw "Destino nao encontrado: $DestinationPath"
    }

    Write-Host "Sincronizando por copia em $DestinationPath"
    foreach ($relativeFile in $filesToSync) {
        $sourceFile = Join-Path $sourcePath $relativeFile
        $targetFile = Join-Path $DestinationPath $relativeFile
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
}

function Sync-RemoteCopy {
    $remoteTarget = Get-RemoteTarget
    $normalizedRemoteRoot = $RemotePath.TrimEnd('/')
    $remoteShellRoot = Convert-ToRemoteShellPath -Path $normalizedRemoteRoot

    Write-Host ("Sincronizando remotamente em {0}:{1}" -f $remoteTarget, $normalizedRemoteRoot)
    $mkdirCmd = "New-Item -ItemType Directory -Force -Path $($remoteShellRoot) | Out-Null"
    $enc = [Convert]::ToBase64String([System.Text.Encoding]::Unicode.GetBytes($mkdirCmd))
    Invoke-RemoteCommand "powershell -NoProfile -EncodedCommand $enc"

    foreach ($relativeFile in $filesToSync) {
        $sourceFile = Join-Path $sourcePath $relativeFile

        if (-not (Test-Path $sourceFile)) {
            Write-Warning "Arquivo ausente na origem: $relativeFile"
            continue
        }

        $remoteFile = Join-RemotePath -BasePath $normalizedRemoteRoot -RelativePath $relativeFile
        $remoteFolder = Get-RemoteParentPath -Path $remoteFile

        if (-not [string]::IsNullOrWhiteSpace($remoteFolder)) {
            $remoteShellFolder = Convert-ToRemoteShellPath -Path $remoteFolder
            $mkdirCmd2 = "New-Item -ItemType Directory -Force -Path $($remoteShellFolder) | Out-Null"
            $enc2 = [Convert]::ToBase64String([System.Text.Encoding]::Unicode.GetBytes($mkdirCmd2))
            Invoke-RemoteCommand "powershell -NoProfile -EncodedCommand $enc2"
        }

        Invoke-RemoteCopy -SourceFile $sourceFile -DestinationFile $remoteFile
        Write-Host "Copiado: $relativeFile"
    }
}

function Sync-Project {
    if (Test-RemoteMode) {
        if ($UseGitPull) {
            Write-Host "Executando git pull/clone no remoto $((Get-RemoteTarget)):$RemotePath"

            # Obter origin URL local para usar em clone remoto, se necessario
            $originUrl = (git -C $sourcePath config --get remote.origin.url) -join "" 
            if (-not $originUrl) {
                throw "Nao foi possivel obter remote.origin.url do repositorio local. Configure o remote origin antes de usar -UseGitPull remoto."
            }

            $remoteShellPath = Convert-ToRemoteShellPath -Path $RemotePath
            $escapedPath = Escape-RemoteShellArgument -Value $remoteShellPath
            $escapedOrigin = Escape-RemoteShellArgument -Value $originUrl

                if ($ForceClone) {
                    Write-Host "Forcando clone remoto (removendo pasta existente, se houver)."
                    $remoteCmd = "if (Test-Path $($remoteShellPath)) { Remove-Item -Recurse -Force $($remoteShellPath) } ; git clone $($originUrl) $($remoteShellPath)"
                }
                else {
                    $remoteCmd = "if (Test-Path ($($remoteShellPath) + '\\.git')) { git -C $($remoteShellPath) pull origin main } else { git clone $($originUrl) $($remoteShellPath) }"
                }

                $b64 = [Convert]::ToBase64String([System.Text.Encoding]::Unicode.GetBytes($remoteCmd))
                $psCmd = "powershell -NoProfile -EncodedCommand $b64"

                Invoke-RemoteCommand $psCmd
            Write-Host "Operacao git remota concluida."
            return
        }

        Sync-RemoteCopy
        return
    }

    if ($UseGitPull -or (Test-Path (Join-Path $TargetPath ".git"))) {
        if (-not (Test-Path $TargetPath)) {
            throw "Destino nao encontrado: $TargetPath"
        }

        Write-Host "Sincronizando via git pull em $TargetPath"
        git -C $TargetPath status
        git -C $TargetPath pull origin main
        if ($LASTEXITCODE -ne 0) {
            throw "git pull falhou em $TargetPath"
        }

        return
    }

    Sync-LocalCopy -DestinationPath $TargetPath
}

if ($Watch) {
    Write-Host "Modo monitoramento ativo. Pressione Ctrl+C para encerrar."
    $lastSnapshot = $null

    while ($true) {
        $currentSnapshot = Get-SourceSnapshot
        if ($currentSnapshot -ne $lastSnapshot) {
            Sync-Project
            $lastSnapshot = $currentSnapshot
        }

        Start-Sleep -Seconds 2
    }
}

Sync-Project