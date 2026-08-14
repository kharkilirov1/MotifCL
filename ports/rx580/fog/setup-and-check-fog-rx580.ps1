[CmdletBinding()]
param(
    [switch]$SkipShaderRegen,
    [int]$OperatorSteps = 200,
    [int]$MachineSteps = 300
)
$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path

function Require-Command([string]$Name, [string]$Hint) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "$Name not found. $Hint"
    }
}

Require-Command "python" "Install Python 3 and add it to PATH."
Require-Command "cmake" "Install CMake and add it to PATH."
Require-Command "ninja" "Install Ninja and add it to PATH."

if (-not $SkipShaderRegen) {
    $glslc = Get-Command glslc -ErrorAction SilentlyContinue
    if (-not $glslc -and $env:VULKAN_SDK) {
        $candidate = Join-Path $env:VULKAN_SDK "Bin\glslc.exe"
        if (Test-Path $candidate) { $env:MOTIFCL_GLSLC = $candidate }
    }
    if (-not (Get-Command glslc -ErrorAction SilentlyContinue) -and -not $env:MOTIFCL_GLSLC) {
        throw "glslc not found. Install the LunarG Vulkan SDK (or set MOTIFCL_GLSLC to glslc.exe), then rerun this script."
    }
}

Write-Host "=== FOG v3 / MotifCL / RX580: build ==="
& (Join-Path $PSScriptRoot "build-fog-vulkan.ps1") -SourceDir $root -SkipShaderRegen:$SkipShaderRegen
if ($LASTEXITCODE -ne 0) { throw "FOG build failed" }

Write-Host "=== FOG v3 / MotifCL / RX580: hardware gates ==="
& (Join-Path $PSScriptRoot "check-fog-rx580.ps1") -OperatorSteps $OperatorSteps -MachineSteps $MachineSteps
if ($LASTEXITCODE -ne 0) { throw "FOG hardware gates failed" }

Write-Host ""
Write-Host "FOG RX580 PORT IS READY FOR LEXICAL PRETRAINING"
Write-Host "Next: prepare a token stream and run ports/rx580/fog/train-fog-v3.ps1"
