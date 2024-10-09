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

# Create a new conda environment with Python
echo "Creating conda environment: $ENV_NAME"
conda create -n $ENV_NAME python=3.12 -y

# Activate the new environment
echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# Install the requirements from requirements.txt
echo "Installing requirements from requirements.txt"
pip install -r requirements.txt

source ~/.bash_profile
source ~/.zshrc

conda activate $ENV_NAME

echo "Setup complete. To activate the environment, run 'conda activate $ENV_NAME'"