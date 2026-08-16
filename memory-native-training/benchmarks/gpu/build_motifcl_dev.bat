@echo off
REM Configure + build MotifCL core lib with Ninja + clang-cl (warnings NOT errors).
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 ( echo VCVARS_FAILED & exit /b 1 )
set "PATH=C:\Program Files\LLVM\bin;C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%"
cd /d C:\Users\Kharki\Desktop\motifcl_production\motifcl_production
echo === CONFIGURE ===
cmake -B build\dev -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl ^
  -DMOTIFCL_WARNINGS_AS_ERRORS=OFF -DMOTIFCL_BUILD_PYTHON=OFF ^
  -DMOTIFCL_BUILD_TESTS=ON -DMOTIFCL_BUILD_TOOLS=OFF ^
  -DMOTIFCL_BUILD_EXAMPLES=OFF -DMOTIFCL_BUILD_BENCHMARKS=OFF -DMOTIFCL_INSTALL=OFF
if errorlevel 1 ( echo CONFIGURE_FAILED & exit /b 1 )
echo === CONFIGURE_OK ===
echo === BUILD motifcl ===
cmake --build build\dev --target motifcl
if errorlevel 1 ( echo BUILD_FAILED & exit /b 1 )
echo === BUILD_OK ===
