@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 ( echo VCVARS_FAILED & exit /b 1 )
set "PATH=C:\Program Files\LLVM\bin;%PATH%"
cd /d C:\Users\Kharki\Desktop\motifcl_production\motifcl_production
clang-cl /std:c++17 /EHsc /O2 /MD /DCL_TARGET_OPENCL_VERSION=120 /I include /I third_party\opencl test_reversible_attn_gpu.cpp build\dev\motifcl.lib build\dev\OpenCL.lib /Fe:test_reversible_attn_gpu.exe
if errorlevel 1 ( echo TEST_BUILD_FAILED & exit /b 1 )
echo === TEST_BUILD_OK ===
