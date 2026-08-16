@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 ( echo VCVARS_FAILED & exit /b 1 )
set "PATH=C:\Program Files\LLVM\bin;C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%"
cd /d C:\Users\Kharki\Desktop\motifcl_production\motifcl_production
cmake --build build\dev --target test_counter_state
if errorlevel 1 ( echo TEST_BUILD_FAILED & exit /b 1 )
echo === TEST_BUILD_OK ===
