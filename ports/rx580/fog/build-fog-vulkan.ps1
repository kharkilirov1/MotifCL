[CmdletBinding()]
param(
    [string]$SourceDir = "",
    [switch]$SkipShaderRegen
)
$ErrorActionPreference = "Stop"
if (-not $SourceDir) { $SourceDir = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path }
Push-Location $SourceDir
try {
    if (-not $SkipShaderRegen) {
        $glslc = Get-Command glslc -ErrorAction SilentlyContinue
        if (-not $glslc -and $env:VULKAN_SDK) {
            $candidate = Join-Path $env:VULKAN_SDK "Bin\glslc.exe"
            if (Test-Path $candidate) { $env:MOTIFCL_GLSLC = $candidate }
        }
        if (-not (Get-Command glslc -ErrorAction SilentlyContinue) -and -not $env:MOTIFCL_GLSLC) {
            throw "glslc not found. Install LunarG Vulkan SDK once, or set MOTIFCL_GLSLC to glslc.exe. It is only needed to embed the new FOG shaders."
        }
        python tools/gen_vulkan_spirv.py
        if ($LASTEXITCODE -ne 0) { throw "Vulkan shader generation failed" }
    }

    cmake --preset rx580-release -DMOTIFCL_ENABLE_OPENCL=OFF
    if ($LASTEXITCODE -ne 0) { throw "CMake configure failed" }
    cmake --build --preset rx580-release --target 09_fog_v3_rx580_pretrain 10_fog_v3_operator_gate 11_fog_v3_structured_machine_gate test_fog_ops --parallel
    if ($LASTEXITCODE -ne 0) { throw "FOG Vulkan build failed" }
    Write-Host "Built FOG RX580 Vulkan targets: test_fog_ops, 10_fog_v3_operator_gate, 11_fog_v3_structured_machine_gate, 09_fog_v3_rx580_pretrain"
} finally { Pop-Location }
