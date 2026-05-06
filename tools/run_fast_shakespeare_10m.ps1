param(
    [string]$Dataset = "build\datasets\tiny_shakespeare.txt",
    [int]$Steps = 100,
    [int]$Batch = 2,
    [string]$MatmulTile = ""
)

$ErrorActionPreference = "Stop"

if (!(Test-Path $Dataset)) {
    New-Item -ItemType Directory -Force (Split-Path $Dataset) | Out-Null
    Invoke-WebRequest `
        -Uri "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt" `
        -OutFile $Dataset
}

cmake --preset release | Write-Host
cmake --build --preset release --target 08_shakespeare_10m_train | Write-Host

if ($MatmulTile) {
    $env:MOTIFCL_MATMUL_F32_TILE = $MatmulTile
} else {
    Remove-Item Env:\MOTIFCL_MATMUL_F32_TILE -ErrorAction SilentlyContinue
}

Write-Host "Fast mode: Release, batch=$Batch, steps=$Steps, dataset=$Dataset, matmul_tile=$($env:MOTIFCL_MATMUL_F32_TILE)"
& ".\build\release\examples\cpp\08_shakespeare_10m_train.exe" $Dataset $Steps $Batch
