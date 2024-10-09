#!/bin/bash

# Exit script if any command fails
set -e

# Function to check if a command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Check for Homebrew, install if not found
if ! command_exists brew; then
  echo "Homebrew not found. Installing Homebrew..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# Check for Miniconda, install if not found
if ! command_exists conda; then
  echo "Miniconda not found. Installing Miniconda via Homebrew..."
  brew install --cask miniconda
  conda init "$(basename $SHELL)"
  echo "Miniconda installed. Please restart your terminal and re-run the script."
  exit 0
fi

# Navigate to the NivlacSignals directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "$SCRIPT_DIR/../"

# Check if requirements.txt exists
if [[ ! -f "requirements.txt" ]]; then
  echo "Error: requirements.txt file not found!"
  exit 1
fi

# Create conda environment name
ENV_NAME="ns_env"

# Check if the environment already exists
if conda info --envs | grep -q "$ENV_NAME"; then
  echo "Updating conda environment: $ENV_NAME"
  conda env update -n $ENV_NAME --file requirements.txt --prune
else
  echo "Creating conda environment: $ENV_NAME"
  conda create -n $ENV_NAME python=3.12 -y
  # Activate the new environment and install requirements
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate $ENV_NAME
  pip install -r requirements.txt
fi

# Activate the conda environment
echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# Create a YAML configuration file for Alpaca API credentials
CONFIG_FILE="alpaca_config.yaml"

# Check if the YAML configuration file already exists
if [[ -f "$CONFIG_FILE" ]]; then
  echo "Configuration file $CONFIG_FILE already exists. Skipping creation."
else
  echo "Creating configuration file: $CONFIG_FILE"
  cat <<EOL > $CONFIG_FILE
alpaca:
  api_key: "YOUR_API_KEY_HERE"
  secret_key: "YOUR_SECRET_KEY_HERE"
  base_url: "https://paper-api.alpaca.markets"
EOL
  echo "Configuration file created. Please edit $CONFIG_FILE to add your Alpaca API credentials."
fi