#!/bin/bash

# Create a new environment
conda create --name pgp python=3.5 -y

# Activate the environment and install packages
# Note: Direct activation (conda activate) doesn't work in scripts as it requires an interactive shell.
# Therefore, we'll use conda run for commands that need to run within the environment.

# Install packages with pip and conda run
conda run -n pgp pip install matplotlib==3.0.3 numpy==1.18.5 pandas==0.20.0 scipy==1.4.1

# Update pip within the environment
conda run -n pgp pip install pip==20.3.4

# Install other required packages
conda run -n pgp pip install ccxt==1.72.1 tensorflow-gpu==1.15

# For installing packages directly from git repositories using pip,
# we use a slightly different approach since conda run might not work seamlessly with git URLs in some cases.
# An alternative approach is to activate the environment in a subshell for this specific command.
echo "Attempting to install tflearn from GitHub repository..."
(
  eval "$(conda shell.bash hook)" # Initialize Conda for script usage
  conda activate pgp
  pip install git+https://github.com/MihaMarkic/tflearn.git@fix/is_sequence_missing
)

echo "Setup completed. please type 'conda activate pgp' to activate the environment."
