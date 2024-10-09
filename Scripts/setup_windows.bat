@echo off

REM Exit script if any command fails
setlocal enabledelayedexpansion

REM Function to check if a command exists
:command_exists
where %1 >nul 2>nul
if not errorlevel 1 (exit /b 0) else (exit /b 1)

REM Check for Chocolatey, install if not found
call :command_exists choco
if errorlevel 1 (
    echo Chocolatey not found. Installing Chocolatey...
    @powershell -NoProfile -ExecutionPolicy Bypass -Command "iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))"
)

REM Check for Miniconda, install if not found
call :command_exists conda
if errorlevel 1 (
    echo Miniconda not found. Installing Miniconda via Chocolatey...
    choco install miniconda -y
    call "%SystemDrive%\tools\miniconda3\Scripts\conda.exe" init
    echo Miniconda installed. Please restart your terminal and re-run the script.
    exit /b 0
)

REM Navigate to the NivlacSignals directory
cd /d "%~dp0\.."

REM Check if requirements.txt exists
if not exist "requirements.txt" (
    echo Error: requirements.txt file not found!
    exit /b 1
)

REM Create conda environment name
set "ENV_NAME=ns_env"

REM Check if the environment already exists
conda info --envs | findstr /c:"%ENV_NAME%" >nul
if %errorlevel% == 0 (
    echo Updating conda environment: %ENV_NAME%
    conda env update -n %ENV_NAME% --file requirements.txt --prune
) else (
    echo Creating conda environment: %ENV_NAME%
    conda create -n %ENV_NAME% python=3.12 -y
    REM Activate the new environment and install requirements
    call "%SystemDrive%\tools\miniconda3\Scripts\activate" base
    conda activate %ENV_NAME%
    pip install -r requirements.txt
)

REM Activate the conda environment
echo Activating environment...
call "%SystemDrive%\tools\miniconda3\Scripts\activate" base
conda activate %ENV_NAME%

REM Create a YAML configuration file for Alpaca API credentials
set "CONFIG_FILE=alpaca_config.yaml"

REM Check if the YAML configuration file already exists
if exist "%CONFIG_FILE%" (
    echo Configuration file %CONFIG_FILE% already exists. Skipping creation.
) else (
    echo Creating configuration file: %CONFIG_FILE%
    echo alpaca:> %CONFIG_FILE%
    echo   api_key: "YOUR_API_KEY_HERE">> %CONFIG_FILE%
    echo   secret_key: "YOUR_SECRET_KEY_HERE">> %CONFIG_FILE%
    echo   base_url: "https://paper-api.alpaca.markets">> %CONFIG_FILE%
    echo Configuration file created. Please edit %CONFIG_FILE% to add your Alpaca API credentials.
)