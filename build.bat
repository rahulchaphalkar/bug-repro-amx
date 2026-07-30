@echo off
setlocal

where clang >nul 2>nul
if errorlevel 1 (
  echo ERROR: clang not found in PATH.
  exit /b 1
)

set XSTATE_FLAG=
if /I "%~1"=="xstate" set XSTATE_FLAG=-DUSE_XSTATE_API

clang -O0 -g -Wall -Wextra %XSTATE_FLAG% -mamx-tile -mamx-bf16 -mavx512bf16 -c main.c -o main.obj
if errorlevel 1 exit /b 1

clang -O0 -g -Wall -Wextra -mamx-tile -mamx-bf16 -mavx512bf16 -c amx_kernel.c -o amx_kernel.obj
if errorlevel 1 exit /b 1

clang main.obj amx_kernel.obj -o amx_repro.exe
if errorlevel 1 exit /b 1

echo Built amx_repro.exe
exit /b 0
