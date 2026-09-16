@echo off
setlocal
rem Configure + build the simple renderer (CMake, Visual Studio generator).
rem crast.exe lands next to this file (Linux: build.sh). A failure prints
rem "BUILD FAILED: ..." and returns the failing step's exit code.
set "ROOT=%~dp0"
set "ROOT=%ROOT:~0,-1%"
if not defined BUILD_DIR set "BUILD_DIR=%ROOT%\build"

where cmake >nul 2>nul
if errorlevel 1 (
    echo BUILD FAILED: cmake is not on PATH.
    exit /b 1
)
cmake -S "%ROOT%" -B "%BUILD_DIR%" -A x64 >nul
set "RC=%errorlevel%"
if not "%RC%"=="0" (
    echo BUILD FAILED: cmake configure exited with %RC%.
    exit /b %RC%
)
cmake --build "%BUILD_DIR%" --config Release --parallel -- /nologo /verbosity:minimal
set "RC=%errorlevel%"
if not "%RC%"=="0" (
    echo BUILD FAILED: the build exited with %RC%.
    exit /b %RC%
)
exit /b 0
