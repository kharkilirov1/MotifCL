[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$TokenFile,
    [int]$Steps = 1000,
    [int]$Batch = 2,
    [int]$Seq = 128,
    [double]$Lr = 0.0003,
    [string]$Checkpoint = "checkpoints\fog_v3_rx580_lexical.mclp",
    [int]$SaveEvery = 250
)
$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$exe = Join-Path $root "build\rx580-release\examples\cpp\09_fog_v3_rx580_pretrain.exe"
if (-not (Test-Path $exe)) { throw "Trainer not built. Run ports/rx580/fog/build-fog-vulkan.ps1 first." }
$env:MOTIFCL_REQUIRE_VULKAN_COMPUTE = "1"
$env:MOTIFCL_REQUIRE_VULKAN_MATMUL = "1"
$env:MOTIFCL_REQUIRE_VULKAN_ATTENTION = "1"
& $exe $TokenFile $Steps $Batch $Seq $Lr $Checkpoint $SaveEvery
if ($LASTEXITCODE -ne 0) { throw "FOG training failed with exit code $LASTEXITCODE" }
