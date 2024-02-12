
# PGPavi Setup Instructions

If the `setup.sh` script doesn't work, you can manually create the environment and install the required packages by following the steps below. 

**Note:** This script installs the repository as it was on 2018-03-16. The `environment.yml` file attempts to update the packages to the latest version, but it may not work due to the deprecation of pandas panels (after `pandas=0.25.0`) and incompatibility with `tensorflow 2.x`.

**Important:** This script is designed to work on Ubuntu 22 with an Intel architecture (`x86_64`). It does not work on ARM64 architectures (Mac).

## Creating a New Environment

First, create a new environment with the following command:

```bash
conda create --name pgp python=3.5
```

Then, activate the environment with this command:

```bash
conda activate pgp
```

## Installing Required Packages

Install the required packages with the following command:

```bash
pip install matplotlib==3.0.3 numpy==1.18.5 pandas==0.20.0 scipy==1.4.1
```

Only then update `pip` (not before) with the following command:

```bash
pip install pip==20.3.4
```

After updating `pip`, install the required packages with the following command (accessible in `pip==20.3.4`):

```bash
pip install ccxt==1.72.1 tensorflow-gpu==1.15 git+https://github.com/MihaMarkic/tflearn.git@fix/is_sequence_missing
```
