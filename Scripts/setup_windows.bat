@echo off
setlocal enabledelayedexpansion

:: Check for admin rights
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo Requesting administrative privileges...
    goto UACPrompt
) else ( goto gotAdmin )

:UACPrompt
    echo Set UAC = CreateObject^("Shell.Application"^) > "%temp%\getadmin.vbs"
    echo UAC.ShellExecute "%~s0", "", "", "runas", 1 >> "%temp%\getadmin.vbs"
    "%temp%\getadmin.vbs"
    exit /B

:gotAdmin
    if exist "%temp%\getadmin.vbs" ( del "%temp%\getadmin.vbs" )
    pushd "%CD%"
    CD /D "%~dp0"

:: Check if Chocolatey is installed
where choco >nul 2>nul
if %errorlevel% neq 0 (
    echo Chocolatey is not installed. Installing Chocolatey...
    @"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -InputFormat None -ExecutionPolicy Bypass -Command "[System.Net.ServicePointManager]::SecurityProtocol = 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))" && SET "PATH=%PATH%;%ALLUSERSPROFILE%\chocolatey\bin"
) else (
    echo Chocolatey is already installed.
)

:: Check if Miniconda is installed
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo Miniconda is not installed. Installing Miniconda...
    choco install miniconda3 -y
    call refreshenv
) else (
    echo Miniconda is already installed.
)

:: Ensure we're in the NivlacSignals Directory
cd /d %~dp0
if not "%CD%"=="%CD:\Scripts=%" (
    cd ..
)

:: Check for requirements.txt
if not exist requirements.txt (
    echo Error: requirements.txt not found in the current directory.
    exit /b 1
)

:: Check if conda environment exists, create if it doesn't, or update if it does
conda info --envs | findstr /C:"ns_env" >nul
if %errorlevel% neq 0 (
    echo Creating new conda environment 'ns_env' with Python 3.12...
    call conda create -n ns_env python=3.12 -y
    if %errorlevel% neq 0 (
        echo Error: Failed to create conda environment. Exiting.
        exit /b 1
    )
) else (
    echo Conda environment 'ns_env' already exists. Updating...
    call conda activate ns_env
    call conda update --all -y
    if %errorlevel% neq 0 (
        echo Error: Failed to update conda environment. Exiting.
        exit /b 1
    )
)

:: Activate the environment and update PATH
call conda activate ns_env
set "PATH=%CONDA_PREFIX%;%CONDA_PREFIX%\Scripts;%PATH%"

:: Verify conda and pip are accessible
where conda
where pip
if %errorlevel% neq 0 (
    echo Error: conda or pip not found in PATH. Exiting.
    exit /b 1
)

:: Install or update requirements from requirements.txt
echo Installing/updating requirements from requirements.txt...
pip install -r requirements.txt --upgrade --no-cache-dir
if %errorlevel% neq 0 (
    echo Error: Failed to install/update requirements. Please check your requirements.txt file and try again.
    echo Attempting to install packages one by one...
    for /F "tokens=1 delims==" %%i in (requirements.txt) do (
        echo Installing %%i...
        pip install %%i --upgrade --no-cache-dir
        if %errorlevel% neq 0 (
            echo Warning: Failed to install %%i. Continuing with next package...
        )
    )
)

:: Create alpaca_config.yaml if it doesn't exist
set "CONFIG_FILE=alpaca_config.yaml"
if not exist %CONFIG_FILE% (
    echo Creating configuration file: %CONFIG_FILE%
    echo alpaca:> %CONFIG_FILE%
    echo   api_key: "YOUR_API_KEY_HERE">> %CONFIG_FILE%
    echo   secret_key: "YOUR_SECRET_KEY_HERE">> %CONFIG_FILE%
    echo   base_url: "https://paper-api.alpaca.markets">> %CONFIG_FILE%
    echo Configuration file created: %CONFIG_FILE%
) else (
    echo Configuration file already exists: %CONFIG_FILE%
)

echo Setup completed successfully.