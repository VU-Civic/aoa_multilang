import numpy as np
import math
from scipy.fft import rfft, irfft

# This method was implemented by ChatGPT based on ISO-9613-1:1993
# and various online resources. It is not an exact implementation of the standard.
# It is intended for audio simulation use only, not for precise acoustical
# calculations.

# -------------------------------
# Helpers
# -------------------------------


def saturation_vapor_pressure_pa(T):
    """Magnus-Tetens approximation. T in Kelvin -> Pa."""
    T_c = T - 273.15
    e_s_hPa = 6.112 * math.exp((17.62 * T_c) / (243.12 + T_c))
    return e_s_hPa * 100.0


def default_alpha_db_per_m(
    freqs_hz: np.ndarray,
    temperature_c: float = 20.0,
    rel_humidity: float = 50.0,
    pressure_pa: float = 101325.0,
) -> np.ndarray:
    """
    Approximate atmospheric absorption (dB/m) vs frequency.
    ISO-9613-like approximate implementation for audio simulation use.
    """
    T = temperature_c + 273.15
    T0 = 293.15
    p = pressure_pa
    p0 = 101325.0

    freqs = np.asarray(freqs_hz, dtype=float)
    f = freqs

    e_s = saturation_vapor_pressure_pa(T)
    h = rel_humidity / 100.0 * (e_s / p)
    h = np.clip(h, 1e-8, 0.999)

    # Relaxation frequencies (oxygen, nitrogen) approximations
    fr_O = p / p0 * (24.0 + 40400.0 * h * (0.02 + h) / (0.391 + h))
    fr_N = p / p0 * (T / T0) ** (-0.5) * (
        9.0 + 280.0 * h * math.exp(-4.17 * ((T0 / T) ** (1 / 3.0)))
    )

    A = 1.84e-11 * (p0 / p) * math.sqrt(T0 / T)
    B = (T / T0) ** (-5.0 / 2.0)
    pref = 8.686  # convert Np -> dB

    C1 = 0.01275 * math.exp(-2239.1 / T)
    C2 = 0.1068 * math.exp(-3352.0 / T)

    denom_O = fr_O + (f**2) / (fr_O + 1e-30)
    denom_N = fr_N + (f**2) / (fr_N + 1e-30)

    alpha = pref * (f**2) * (A + B * (C1 / denom_O + C2 / denom_N))
    return np.maximum(alpha, 0.0)


# -------------------------------
# Propagation core
# -------------------------------

def apply_propagation(
    signal: np.ndarray,
    fs: int,
    distance: float,
    temperature_c: float = 20.0,
    rel_humidity: float = 50.0,
    pressure_pa: float = 101325.0,
) -> np.ndarray:
    """
    Apply propagation effects (spherical spreading + atmospheric absorption).
    - signal: time-domain np.array
    - fs: sampling rate
    - distance: source-to-mic distance in meters
    """
    N = len(signal)
    X = rfft(signal)
    freqs_r = np.fft.rfftfreq(N, 1.0 / fs)

    # frequency-dependent attenuation
    alpha_db = default_alpha_db_per_m(
        freqs_r, temperature_c=temperature_c,
        rel_humidity=rel_humidity, pressure_pa=pressure_pa
    )
    attn = 10.0 ** (-alpha_db * distance / 20.0)

    # spherical spreading
    spread = 1.0 / max(distance, 1e-6)

    Y = X * attn * spread  # type: ignore
    y = irfft(Y, n=N)
    return y  # type: ignore
