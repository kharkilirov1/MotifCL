[CmdletBinding()]
param(
    [string]$SourceDir = "",
    [string]$BuildDir = "",
    [string]$Abi = "arm64-v8a",
    [int]$Api = 29,
    [switch]$ProbeOnly,
    [switch]$SkipRun
)

$ErrorActionPreference = "Stop"

if (-not $SourceDir) {
    $SourceDir = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
}

if (-not $BuildDir) {
    $BuildDir = Join-Path $SourceDir "build\android-mi9-arm64"
}

function Require-Command($Name) {
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if (-not $cmd) {
        throw "$Name not found in PATH"
    }
    return $cmd.Source
}

function Find-AndroidSdk {
    $candidates = @(@(
        $env:ANDROID_HOME,
        $env:ANDROID_SDK_ROOT,
        (Join-Path $env:LOCALAPPDATA "Android\Sdk"),
        (Join-Path $env:USERPROFILE "AppData\Local\Android\Sdk"),
        "C:\Android\Sdk"
    ) | Where-Object { $_ -and (Test-Path $_) } | Select-Object -Unique)
    if ($candidates.Count -eq 0) {
        throw "Android SDK not found. Set ANDROID_HOME or ANDROID_SDK_ROOT."
    }
    return (Resolve-Path $candidates[0]).Path
}

function Find-AndroidNdk([string]$SdkRoot) {
    $candidates = @(@(
        $env:ANDROID_NDK_HOME,
        $env:ANDROID_NDK_ROOT
    ) | Where-Object { $_ -and (Test-Path $_) })
    if ($candidates.Count -gt 0) {
        return (Resolve-Path $candidates[0]).Path
    }

    $ndkRoot = Join-Path $SdkRoot "ndk"
    if (-not (Test-Path $ndkRoot)) {
        throw "Android NDK not found under $ndkRoot. Install an NDK or set ANDROID_NDK_HOME."
    }
    $latest = Get-ChildItem -LiteralPath $ndkRoot -Directory |
        Sort-Object Name -Descending |
        Select-Object -First 1
    if (-not $latest) {
        throw "Android NDK directory is empty: $ndkRoot"
    }
    return $latest.FullName
}

function Invoke-AdbShell([string]$Command) {
    & adb shell $Command
    if ($LASTEXITCODE -ne 0) {
        throw "adb shell failed: $Command"
    }
}

$adbPath = Require-Command adb
Write-Host "adb: $adbPath"

$model = (& adb shell getprop ro.product.model).Trim()
$device = (& adb shell getprop ro.product.device).Trim()
$abis = (& adb shell getprop ro.product.cpu.abilist).Trim()
$sdkLevel = (& adb shell getprop ro.build.version.sdk).Trim()
$platform = (& adb shell getprop ro.board.platform).Trim()

Write-Host "device: $model / $device / $platform / sdk=$sdkLevel / abi=$abis"
if ($abis -notmatch [regex]::Escape($Abi)) {
    throw "Connected device does not advertise ABI $Abi; abilist=$abis"
}

$remoteOpenCl = if ($Abi -eq "arm64-v8a") { "/vendor/lib64/libOpenCL.so" } else { "/vendor/lib/libOpenCL.so" }
& adb shell "test -f $remoteOpenCl"
if ($LASTEXITCODE -ne 0) {
    throw "OpenCL loader not found on device at $remoteOpenCl"
}
Write-Host "OpenCL loader: $remoteOpenCl"

if ($ProbeOnly) {
    Write-Host "ProbeOnly requested; stopping before SDK/NDK configure."
    exit 0
}

$sdkRoot = Find-AndroidSdk
$ndkRoot = Find-AndroidNdk $sdkRoot
$toolchain = Join-Path $ndkRoot "build\cmake\android.toolchain.cmake"
if (-not (Test-Path $toolchain)) {
    throw "Android CMake toolchain not found: $toolchain"
}
Require-Command cmake | Out-Null
Require-Command ninja | Out-Null

$openClDir = Join-Path $BuildDir "opencl\$Abi"
New-Item -ItemType Directory -Force -Path $openClDir | Out-Null
$localOpenCl = Join-Path $openClDir "libOpenCL.so"
& adb pull $remoteOpenCl $localOpenCl
if ($LASTEXITCODE -ne 0) {
    throw "failed to pull $remoteOpenCl"
}

& cmake -S $SourceDir -B $BuildDir -G Ninja `
    "-DCMAKE_TOOLCHAIN_FILE=$toolchain" `
    "-DANDROID_ABI=$Abi" `
    "-DANDROID_PLATFORM=android-$Api" `
    "-DANDROID_STL=c++_static" `
    "-DCMAKE_BUILD_TYPE=Release" `
    "-DMOTIFCL_OPENCL_LIBRARY=$localOpenCl" `
    "-DMOTIFCL_BUILD_TESTS=OFF" `
    "-DMOTIFCL_BUILD_EXAMPLES=OFF" `
    "-DMOTIFCL_BUILD_TOOLS=ON" `
    "-DMOTIFCL_BUILD_BENCHMARKS=OFF" `
    "-DMOTIFCL_BUILD_PYTHON=OFF" `
    "-DMOTIFCL_INSTALL=OFF" `
    "-DMOTIFCL_OPENCL_FAST_RELAXED_MATH=ON" `
    "-DMOTIFCL_ANDROID_MI9_TUNING=ON"
if ($LASTEXITCODE -ne 0) {
    throw "CMake configure failed"
}

& cmake --build $BuildDir --target motifcl_dump_opencl_info motifcl_compare_cpu_gpu --parallel
if ($LASTEXITCODE -ne 0) {
    throw "Android native build failed"
}

if ($SkipRun) {
    exit 0
}

$remoteDir = "/data/local/tmp/motifcl"
Invoke-AdbShell "mkdir -p $remoteDir"
& adb push (Join-Path $BuildDir "tools\motifcl_dump_opencl_info") "$remoteDir/motifcl_dump_opencl_info"
if ($LASTEXITCODE -ne 0) {
    throw "failed to push motifcl_dump_opencl_info"
}
& adb push (Join-Path $BuildDir "tools\motifcl_compare_cpu_gpu") "$remoteDir/motifcl_compare_cpu_gpu"
if ($LASTEXITCODE -ne 0) {
    throw "failed to push motifcl_compare_cpu_gpu"
}
if (Test-Path (Join-Path $BuildDir "kernels")) {
    & adb push (Join-Path $BuildDir "kernels") "$remoteDir/kernels"
    if ($LASTEXITCODE -ne 0) {
        throw "failed to push kernels"
    }
}
Invoke-AdbShell "chmod 755 $remoteDir/motifcl_dump_opencl_info"
Invoke-AdbShell "chmod 755 $remoteDir/motifcl_compare_cpu_gpu"
Invoke-AdbShell "cd $remoteDir && export LD_LIBRARY_PATH=/vendor/lib64:/system/vendor/lib64:/vendor/lib:/system/vendor/lib:`$LD_LIBRARY_PATH && export MOTIFCL_KERNEL_DIR=$remoteDir/kernels && ./motifcl_dump_opencl_info"
Invoke-AdbShell "cd $remoteDir && export LD_LIBRARY_PATH=/vendor/lib64:/system/vendor/lib64:/vendor/lib:/system/vendor/lib:`$LD_LIBRARY_PATH && export MOTIFCL_KERNEL_DIR=$remoteDir/kernels && ./motifcl_compare_cpu_gpu"
