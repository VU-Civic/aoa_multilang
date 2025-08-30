import numpy as np
from scipy.signal import correlate

C = 343.0  # speed of sound [m/s]


def gcc_phat(sig, refsig, fs, max_tau=None, interp=1):
    """
    Compute time delay estimation between sig and refsig using GCC-PHAT.
    Returns the delay in seconds.
    """
    n = sig.shape[0] + refsig.shape[0]

    # FFT
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)

    # PHAT weighting
    R /= np.abs(R) + 1e-15

    # IFFT
    cc = np.fft.irfft(R, n=(interp * n))
    max_shift = int(interp * n / 2)

    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))
    shift = np.argmax(np.abs(cc)) - max_shift

    return shift / float(interp * fs)


def estimate_aoa(signals, fs, mic_positions):
    """
    Estimate AoA using GCC-PHAT across all microphone pairs.

    Parameters
    ----------
    signals : list of np.ndarray
        List of microphone signals (all same length).
    fs : int
        Sampling rate.
    mic_positions : np.ndarray
        Nx2 array of mic coordinates (meters).

    Returns
    -------
    est_angle : float
        Estimated source angle [radians] relative to array x-axis.
    """
    n_mics = len(signals)
    tdoa_measurements = []
    pair_positions = []

    # --- 1. Extract all TDOAs ---
    for i in range(n_mics):
        for j in range(i+1, n_mics):
            tau_ij = gcc_phat(signals[i], signals[j], fs)
            tdoa_measurements.append(tau_ij)
            pair_positions.append((i, j))
    tdoa_measurements = np.array(tdoa_measurements)

    # --- 2. Solve AoA with least squares ---
    # For a plane wave assumption: tau_ij ≈ ( (ri - rj) · u ) / c
    # where u = [cos θ, sin θ]
    A = []
    b = []
    for tau, (i, j) in zip(tdoa_measurements, pair_positions):
        ri = mic_positions[i]
        rj = mic_positions[j]
        diff = ri - rj
        A.append(diff)
        b.append(C * tau)
    A = np.vstack(A)    # shape (M,2)
    b = np.array(b)     # shape (M,)

    # Least squares solution for u
    u, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    u /= np.linalg.norm(u) + 1e-15

    # Convert to angle
    est_angle = np.arctan2(u[1], u[0])
    return est_angle
