@echo off
setlocal
set "ROOT=%~dp0"
where python >nul 2>nul
if %ERRORLEVEL%==0 (
  python "%ROOT%tools\motifcl_cli.py" %*
  exit /b %ERRORLEVEL%
)
where py >nul 2>nul
if %ERRORLEVEL%==0 (
  py -3 "%ROOT%tools\motifcl_cli.py" %*
  exit /b %ERRORLEVEL%
)
echo Python was not found. Install Python 3 or add it to PATH.
exit /b 1
