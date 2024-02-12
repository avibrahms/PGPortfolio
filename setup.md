# If the setup.sh script doesn't work, you can use the following commands to create the environment and install the required packages:

# Note: this script installs the repo as is was on 2018-03-16, the environment.yml file is an attempt to update the packages to the latest version, but it doesn't work as pandas panels need to be replaced (deprecated after pandas=0.25.0) and the tensorflow version is not compatible with tensorflow 2.x

# Note: this script works on ubuntu 22 with an intel architecture (x86_64) , doesn't work on arm64 (mac)

# First create a new environment with the following command:
conda create --name pgp python=3.5
# Then activate the environment with the following command:
conda activate pgp
# Then install the required packages with the following command (pandas==0.20.0 is needed for panels and is only accessible with the pip version from python=3.5)
pip install matplotlib==3.0.3 numpy==1.18.5 pandas==0.20.0 scipy==1.4.1
# Only then uptade pip (not before) with the following command:
pip install pip==20.3.4
# Then install the required packages with the following command (accessible in pip==20.3.4):
pip install ccxt==1.72.1 tensorflow-gpu==1.15 git+https://github.com/MihaMarkic/tflearn.git@fix/is_sequence_missing
