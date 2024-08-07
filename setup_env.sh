#!/bin/bash

# Script to set up the project environment

# Name of the conda environment
ENV_NAME="gigatech"

# Path to the requirements.txt file
REQUIREMENTS_FILE="requirements.txt"

# Function to create conda environment
create_conda_env() {
    echo "Creating conda environment: $ENV_NAME with Python 3.10.14..."
    conda create -y -n $ENV_NAME python=3.10.14
}

# Function to activate conda environment
activate_conda_env() {
    echo "Activating conda environment: $ENV_NAME..."
    source activate $ENV_NAME
}

# Function to install packages from requirements.txt
install_requirements() {
    echo "Installing packages from $REQUIREMENTS_FILE..."
    pip install -r $REQUIREMENTS_FILE
}

# Main function
main() {
    create_conda_env
    activate_conda_env
    install_requirements
    echo "Environment setup is complete. To activate the environment, run: conda activate $ENV_NAME"
}

# Run the main function
main
