# MN-MoE15M comparison series on the RX 580 (Vulkan FP32).
#
# Runs the arms sequentially on one GPU, each with its own OUT_DIR and log, and
# prints the parsed summary at the end. Every arm is resumable: re-running the
# script with the same -OutRoot continues from each arm's meta.txt.
#
# Arms:
#   1. moe_e192       MoE, 7 experts x hidden 192 + shared 192  (active 384, capacity 1536)
#   2. dense_h768     dense SwiGLU hidden 768                  (active 768) - the project baseline
#   3. dense_h384     dense SwiGLU hidden 384                  (active 384) - equal ACTIVE width to arm 1
#   4. moe_e384       MoE, 7 experts x hidden 384 + shared 192 (active 576) - capacity scaling arm
#   5. moe_aux3e3     MoE e192 with a 3x weaker auxiliary balance gradient
#   6. moe_aux0       MoE e192 with bias-only balancing (no auxiliary gradient)
#
# Arms 5-6 are the balancing ablation: arm 1 balances by construction, and the
# expected trade-off is specialization (weaker aux -> sharper routing) against
# expert collapse (bias-only collapsed within 40 steps at clamp 2 in the first
# implementation).
#
# MOTIFCL_VK_PREFER_DEVICE_LOCAL=1 is required on this box: with the current
# default (host-visible/GTT placement) the same configuration runs 2.0-2.8x
# slower at bit-identical losses (docs/MOE_PRETRAIN_RESULTS.md).
param(
    [int]$Steps = 12000,
    [int]$Batch = 8,
    [int]$Seq = 256,
    [string]$DataDir = 'data/tinystories',
    [string]$OutRoot = 'out',
    [string[]]$Arms = @('moe_e192', 'dense_h768', 'dense_h384', 'moe_e384'),
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
# `powershell -File` passes an array parameter as one comma-joined string; normalize.
$Arms = @($Arms | ForEach-Object { $_ -split ',' } | Where-Object { $_ })
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
Set-Location $repo

$moeExe = Join-Path $repo 'build/release/tools/motifcl_train_moe15m.exe'
$denseExe = Join-Path $repo 'build/release/tools/motifcl_train_ternary15m.exe'
foreach ($exe in @($moeExe, $denseExe)) {
    if (-not (Test-Path $exe)) { throw "missing executable $exe - build it first" }
}

$env:DATA_DIR = $DataDir
$env:BATCH = "$Batch"
$env:SEQ = "$Seq"
$env:SEED = '1337'
$env:RESUME = '1'
$env:EVAL_EVERY = '500'
$env:EVAL_RECORDS = '96'
$env:CKPT_EVERY = '2000'
$env:LOG_EVERY = '100'
$env:TIE_HEAD = '0'
$env:MOTIFCL_VK_PREFER_DEVICE_LOCAL = '1'   # required: see the header comment

function Invoke-Arm {
    param([string]$Name, [string]$Exe, [hashtable]$Env)
    $outDir = Join-Path $OutRoot $Name
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    $env:OUT_DIR = $outDir
    $env:STEPS = "$Steps"
    foreach ($key in $Env.Keys) { Set-Item -Path "env:$key" -Value $Env[$key] }
    $log = Join-Path $outDir 'log.txt'
    Write-Host "=== arm $Name -> $outDir (steps=$Steps batch=$Batch seq=$Seq) ==="
    if ($DryRun) { return }
    $proc = Start-Process -FilePath $Exe -NoNewWindow -PassThru -Wait `
        -WorkingDirectory $repo -RedirectStandardOutput $log
    Write-Host "=== arm $Name exit=$($proc.ExitCode) ==="
}

foreach ($arm in $Arms) {
    switch ($arm) {
        'moe_e192'   { Invoke-Arm $arm $moeExe   @{ SAVE_CKPT = '1'; BALANCE_GAMMA = '0.01'; AUX_WEIGHT = '0.01'; HIDDEN_EXP = '192'; HIDDEN_SHARED = '192'; BALANCE_CLAMP = '4' } }
        'moe_e384'   { Invoke-Arm $arm $moeExe   @{ SAVE_CKPT = '1'; BALANCE_GAMMA = '0.01'; AUX_WEIGHT = '0.01'; HIDDEN_EXP = '384'; HIDDEN_SHARED = '192'; BALANCE_CLAMP = '4' } }
        'moe_aux3e3' { Invoke-Arm $arm $moeExe   @{ SAVE_CKPT = '1'; BALANCE_GAMMA = '0.01'; AUX_WEIGHT = '0.003'; HIDDEN_EXP = '192'; HIDDEN_SHARED = '192'; BALANCE_CLAMP = '4' } }
        'moe_aux0'   { Invoke-Arm $arm $moeExe   @{ SAVE_CKPT = '1'; BALANCE_GAMMA = '0.01'; AUX_WEIGHT = '0'; HIDDEN_EXP = '192'; HIDDEN_SHARED = '192'; BALANCE_CLAMP = '4' } }
        'dense_h768' { Invoke-Arm $arm $denseExe @{ HIDDEN = '768' } }
        'dense_h384' { Invoke-Arm $arm $denseExe @{ HIDDEN = '384' } }
        default      { throw "unknown arm '$arm'" }
    }
}

if (-not $DryRun) {
    $logs = $Arms | ForEach-Object { Join-Path (Join-Path $OutRoot $_) 'log.txt' }
    & 'C:\Users\Kharki\AppData\Local\Programs\Python\Python310\python.exe' 'tools/moe_log_summary.py' @logs
}
