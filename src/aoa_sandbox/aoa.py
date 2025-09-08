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
    delay = shift / float(interp * fs)
    return delay


def estimate_aoa(signals, fs, mic_positions):
    """
    Estimate AoA vector from microphone signals.
    Args:
        signals: list of time-domain signals (already aligned in length)
        fs: sample rate
        mic_positions: (N_mics, 3) array of mic positions in ECEF
    Returns:
        aoa_vec: unit vector in ECEF pointing toward source
    """
    N = len(signals)
    if N < 2:
        raise ValueError("Need at least 2 microphones for AoA")

    # Compute TDOAs using GCC-PHAT between mic 0 and others
    ref = signals[0]
    tdoas = []
    baselines = []
    for i in range(1, N):
        tau = gcc_phat(ref, signals[i], fs)
        tdoas.append(tau)
        baselines.append(mic_positions[i] - mic_positions[0])

    # Solve least-squares for direction vector
    A = []
    b = []
    c = 343.0  # speed of sound
    for tau, baseline in zip(tdoas, baselines):
        baseline = np.asarray(baseline)
        # Equation: (baseline · aoa) = c * tau
        A.append(baseline.reshape(1, -1))
        b.append(c * tau)
    A = np.vstack(A)
    b = np.array(b)

    # Least squares solution for direction
    aoa_vec, *_ = np.linalg.lstsq(A, b, rcond=None)
    aoa_vec /= np.linalg.norm(aoa_vec)  # normalize

    return aoa_vec
