# spl.py
import numpy as np

P_REF = 20e-6  # 20 µPa reference RMS pressure


def spl_to_pressure(spl_db: float) -> float:
    """Convert dB SPL to RMS pressure in Pascal."""
    return P_REF * (10.0 ** (spl_db / 20.0))


def pressure_to_spl(pressure_pa: float) -> float:
    """Convert RMS pressure in Pascal to dB SPL."""
    return 20.0 * np.log10(pressure_pa / P_REF)
