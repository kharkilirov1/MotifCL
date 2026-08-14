[CmdletBinding()]
param(
    [int]$OperatorSteps = 200,
    [int]$OperatorBatch = 32,
    [double]$OperatorLr = 0.003,
    [int]$DModel = 320,
    [int]$Depth = 8,
    [int]$MachineSteps = 300,
    [int]$MachineBatch = 64,
    [double]$MachineLr = 0.002
)
$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$build = Join-Path $root "build\rx580-release"
$ops = Join-Path $build "tests\test_fog_ops.exe"
$gate = Join-Path $build "examples\cpp\10_fog_v3_operator_gate.exe"
$machine = Join-Path $build "examples\cpp\11_fog_v3_structured_machine_gate.exe"
if (-not (Test-Path $ops) -or -not (Test-Path $gate) -or -not (Test-Path $machine)) {
    throw "FOG targets not built. Run ports/rx580/fog/build-fog-vulkan.ps1 first."
}
$env:MOTIFCL_REQUIRE_VULKAN_COMPUTE = "1"
$env:MOTIFCL_REQUIRE_VULKAN_MATMUL = "1"
$env:MOTIFCL_REQUIRE_VULKAN_ATTENTION = "1"
Write-Host "[1/3] FOG Vulkan primitive/autograd + structured smoke"
& $ops
if ($LASTEXITCODE -ne 0) { throw "test_fog_ops failed: $LASTEXITCODE" }
Write-Host "[2/3] FOG hard-grammar training/recurrent gate"
& $gate $OperatorSteps $OperatorBatch $OperatorLr $DModel $Depth
if ($LASTEXITCODE -ne 0) { throw "operator gate failed: $LASTEXITCODE" }
Write-Host "[3/3] FOG structured binder -> generated state -> recurrent re-address gate"
& $machine $MachineSteps $MachineBatch $MachineLr 8
if ($LASTEXITCODE -ne 0) { throw "structured machine gate failed: $LASTEXITCODE" }
Write-Host "FOG RX580 Vulkan hardware gates PASSED"
