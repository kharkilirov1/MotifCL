[CmdletBinding()]
param(
    [string]$SourceDir = "",
    [switch]$SkipTests,
    [switch]$SkipBenchmarks
)

$ErrorActionPreference = "Stop"

if (-not $SourceDir) {
    $SourceDir = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
}

function Set-Rx580RuntimeEnv {
    $env:MOTIFCL_MATMUL_F32_TILE = "16"
    $env:MOTIFCL_FA_TILE = "16"
    $env:MOTIFCL_FA_WG = "128"
    $env:MOTIFCL_OPENCL_BUILD_OPTIONS = "-cl-mad-enable -cl-no-signed-zeros"
}

cmake --preset rx580-release
if ($LASTEXITCODE -ne 0) { throw "rx580-release configure failed" }

cmake --build --preset rx580-release --parallel
if ($LASTEXITCODE -ne 0) { throw "rx580-release build failed" }

$buildDir = Join-Path $SourceDir "build\rx580-release"

if (-not $SkipTests) {
    ctest --test-dir $buildDir --output-on-failure
    if ($LASTEXITCODE -ne 0) { throw "rx580-release tests failed" }
}

Set-Rx580RuntimeEnv

$dumpInfo = Join-Path $buildDir "tools\motifcl_dump_opencl_info.exe"
if (Test-Path $dumpInfo) {
    & $dumpInfo
}

if (-not $SkipBenchmarks) {
    $tuner = Join-Path $buildDir "tools\motifcl_kernel_tuner.exe"
    if (Test-Path $tuner) {
        & $tuner
    }
}
