import numpy as np
import scipy.signal
import librosa
import matplotlib.pyplot as plt


def max_filter_freq(logS, max_size=3):
    """
    Apply a maximum filter along the frequency axis of a spectrogram.
    logS: 2D array (frequency bins x time frames)
    """
    n_bins, n_frames = logS.shape
    if max_size <= 1 or n_bins == 1:
        return logS.copy()

    max_size = min(max_size, n_bins)
    pad = max_size // 2

    # Manual numpy implementation
    S_padded = np.pad(logS, ((pad, pad), (0, 0)), mode="edge")
    out = np.empty_like(logS)
    for f in range(n_bins):
        out[f, :] = np.max(S_padded[f: f + max_size, :], axis=0)
    return out


def superflux_from_log_spectrogram(logS, lag=1, max_size=3, method="loop"):
    """
    Compute SuperFlux onset envelope from a log-amplitude spectrogram.
    Returns an array of onset strengths (length = n_frames).
    """
    if lag < 1:
        raise ValueError("lag must be >= 1")

    logS_max = max_filter_freq(logS, max_size=max_size)
    diff = np.maximum(0.0, logS_max[:, lag:] - logS_max[:, :-lag])
    onset = np.sum(diff, axis=0)
    onset = np.concatenate((np.zeros(lag, dtype=onset.dtype), onset))
    onset = np.abs(onset)
    m = np.max(onset)
    onset = onset / 1e3
    onset = np.clip(onset, 0.0, 1.0)
    # if m > 0:
    #     # onset = onset / (m + 1e-12)
    #     # print(m)
    #     onset = onset / 1e3
    return onset


def superflux_stft(y, sr=22050, n_fft=1024, hop_length=512, lag=1, max_size=3, method="loop"):
    """
    SuperFlux onset detection using STFT magnitude.
    Returns: onset_env, times, logS
    """
    _, _, Zxx = scipy.signal.stft(
        y, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length, boundary=None)  # type: ignore
    S = np.abs(Zxx)
    logS = np.log1p(S)
    onset_env = superflux_from_log_spectrogram(
        logS, lag=lag, max_size=max_size, method=method)
    times = librosa.frames_to_time(
        np.arange(len(onset_env)), sr=sr, hop_length=hop_length)
    return onset_env, times, logS


def superflux_mel(y, sr=22050, n_fft=1024, hop_length=512, n_mels=80, lag=1, max_size=3, method="loop"):
    """
    SuperFlux onset detection using mel spectrogram.
    Returns: onset_env, times, logS
    """
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    logS = librosa.power_to_db(S)
    onset_env = superflux_from_log_spectrogram(
        logS, lag=lag, max_size=max_size, method=method)
    times = librosa.frames_to_time(
        np.arange(len(onset_env)), sr=sr, hop_length=hop_length)
    return onset_env, times, logS


def superflux_general(S, sr=96000, hop_length=512, lag=1, max_size=3):
    onset_env = superflux_from_log_spectrogram(S, lag=lag, max_size=max_size)
    times = librosa.frames_to_time(
        np.arange(len(onset_env)), sr=sr, hop_length=hop_length)
    return onset_env, times, S


def stretch_away(x, t=0.5, alpha=2.0):
    """
    Stretch values in [0,1] away from a threshold t.
    x: array-like, input values in [0,1]
    t: float, threshold (0 < t < 1)
    alpha: float > 1, stretching strength
    """
    x = np.asarray(x)
    f = np.empty_like(x, dtype=float)
    mask = x <= t
    f[mask] = t * (x[mask] / t) ** alpha
    f[~mask] = 1 - (1 - t) * ((1 - x[~mask]) / (1 - t)) ** alpha
    return f


# ---------------- Example usage ----------------
if __name__ == "__main__":
    y, sr = librosa.load("examples/Campbell/s1.wav", sr=None)
    print(f"Audio loaded: {y.shape[0]} samples at {sr} Hz")
    params = dict(n_fft=2**12, hop_length=2**10, lag=1, max_size=3)

    env_stft, t_stft, logS_stft = superflux_stft(
        y, sr, **params, method="loop")  # type: ignore
    env_mel,  t_mel,  logS_mel = superflux_mel(
        y, sr, **params, n_mels=64, method="loop")  # type: ignore

    env_stft = stretch_away(env_stft, t=0.5, alpha=2.0)
    env_mel = stretch_away(env_mel, t=0.5, alpha=2.0)

    plt.figure(figsize=(12, 4))
    plt.plot(t_stft, env_stft, label="SuperFlux (STFT)")
    plt.plot(t_mel, env_mel, label="SuperFlux (Mel)", alpha=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Normalized onset strength")
    plt.title("SuperFlux: STFT vs Mel")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 4))
    t_signal = np.arange(len(y)) / sr
    plt.plot(t_signal, y / np.max(np.abs(y)),
             label="Original Signal", alpha=0.7)
    plt.plot(t_mel, env_mel, label="SuperFlux (Mel)", linewidth=2, alpha=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude / Onset Strength")
    plt.title("Original Signal and SuperFlux Onset Envelope")
    plt.legend()
    plt.tight_layout()
    plt.show()
