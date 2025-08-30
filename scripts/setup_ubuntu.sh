#!/usr/bin/env bash
set -euo pipefail
echo 'Installing Ubuntu system packages required for building native backends...'
echo 'This script requires sudo.'
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build gcc g++ make pkg-config git python3-dev python3-pip libsndfile1-dev libsndfile1
sudo apt-get install -y gcc-arm-none-eabi binutils-arm-none-eabi
echo 'Done. Now run: pip install -e .'
