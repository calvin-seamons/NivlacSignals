# Check for admin rights and elevate if necessary
if (-NOT ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator"))  
{  
  $arguments = "& '" + $myinvocation.mycommand.definition + "'"
  Start-Process powershell -Verb runAs -ArgumentList $arguments
  Break
}

# Ensure we're in the NivlacSignals Directory
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $scriptPath
if ((Split-Path -Leaf $PWD) -eq "Scripts") {
    Set-Location ..
}

# Check if Chocolatey is installed
if (!(Get-Command choco -ErrorAction SilentlyContinue)) {
    Write-Host "Chocolatey is not installed. Installing Chocolatey..."
    Set-ExecutionPolicy Bypass -Scope Process -Force
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
    Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
} else {
    Write-Host "Chocolatey is already installed."
}

# Check if Miniconda is installed
if (!(Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Host "Miniconda is not installed. Installing Miniconda..."
    choco install miniconda3 -y
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
} else {
    Write-Host "Miniconda is already installed."
}

# Check for requirements.txt
if (!(Test-Path requirements.txt)) {
    Write-Host "Error: requirements.txt not found in the current directory."
    exit 1
}

# Check if conda environment exists, create if it doesn't, or update if it does
$envExists = conda info --envs | Select-String "ns_env" -Quiet
if (!$envExists) {
    Write-Host "Creating new conda environment 'ns_env' with Python 3.12..."
    conda create -n ns_env python=3.12 -y
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: Failed to create conda environment. Exiting."
        exit 1
    }
} else {
    Write-Host "Conda environment 'ns_env' already exists. Updating..."
    conda activate ns_env
    conda update --all -y
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: Failed to update conda environment. Exiting."
        exit 1
    }
}

# Activate the environment and update PATH
conda activate ns_env
$env:Path = "$env:CONDA_PREFIX;$env:CONDA_PREFIX\Scripts;" + $env:Path

# Verify conda and pip are accessible
Get-Command conda -ErrorAction Stop
Get-Command pip -ErrorAction Stop

# Install or update requirements from requirements.txt
Write-Host "Installing/updating requirements from requirements.txt..."
pip install -r requirements.txt --upgrade --no-cache-dir
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to install/update requirements. Please check your requirements.txt file and try again."
    Write-Host "Attempting to install packages one by one..."
    Get-Content requirements.txt | ForEach-Object {
        $package = $_.Split('=')[0]
        Write-Host "Installing $package..."
        pip install $package --upgrade --no-cache-dir
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Warning: Failed to install $package. Continuing with next package..."
        }
    }
}

# Create alpaca_config.yaml if it doesn't exist
$CONFIG_FILE = "alpaca_config.yaml"
if (!(Test-Path $CONFIG_FILE)) {
    Write-Host "Creating configuration file: $CONFIG_FILE"
    @"
alpaca:
  api_key: "YOUR_API_KEY_HERE"
  secret_key: "YOUR_SECRET_KEY_HERE"
  base_url: "https://paper-api.alpaca.markets"
"@ | Set-Content $CONFIG_FILE
    Write-Host "Configuration file created: $CONFIG_FILE"
} else {
    Write-Host "Configuration file already exists: $CONFIG_FILE"
}

Write-Host "Setup completed successfully."