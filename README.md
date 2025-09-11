# AOA Sandbox

A simulation framework for experimenting with **angle-of-arrival (AoA)**, **time-of-arrival (ToA)**, and sound propagation effects in microphone arrays.  
The library supports 3D sensor placement, quaternions for orientation, SPL-based source loudness, propagation modeling (attenuation, frequency-dependent loss), noise injection, and AoA/ToA-based localization.

---

## Features
- Define multiple sensors and microphones in a TOML config file
- Simulate sound propagation with distance-based attenuation and frequency-dependent loss
- Add realistic noise and amplitude scaling based on source SPL and microphone sensitivity
- Estimate **AoA** from microphone array recordings
- Estimate **event position** using **least-squares triangulation** or ToA-based multilateration
- Supports ECEF coordinates and quaternions for global orientation handling

---

## Installation

Clone the repository:

git clone https://github.com/Gyoorey/aoa_multilang.git

cd aoa_multilang

run scripts/setup_ubuntu.sh with sudo

pip install -e .


## Example usage
aoa-sim run src/aoa_sandbox/configs/exp1.toml