# utils.py
"""
Utility functions for AoA sandbox:
- audio I/O and normalization
- upsampling to a high internal rate (resample_poly)
- ADC sampling simulation (discrete sampling instants)
- compute per-microphone delay samples (aligned to farthest mic)
- assemble per-mic high-rate signals (same length, farthest-mic baseline)
"""

from __future__ import annotations
import math
from typing import Tuple, Sequence
import numpy as np
from scipy.signal import resample_poly
import soundfile as sf


def speed_of_sound(temperature_c: float | None = None) -> float:
    """
    Return speed of sound in m/s.
    If temperature_c is None, return the conventional 343.0 m/s.
    Otherwise use approximation v = 331.4 + 0.6 * T (valid near room temp).
    """
    if temperature_c is None:
        return 343.0
    return 331.4 + 0.6 * float(temperature_c)


def load_audio_mono(path: str, normalize: bool = True) -> Tuple[np.ndarray, int]:
    """
    Load audio (file) and return mono float32 array in range [-1,1] and sample rate.
    Uses soundfile (libsndfile).
    """
    data, fs = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    data = data.astype("float32")
    if normalize:
        maxv = np.max(np.abs(data))
        if maxv > 0:
            data = data / maxv
    return data, int(fs)


def upsample_to(signal: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    """
    Upsample `signal` from fs_in to fs_out using polyphase resampling.
    fs_in and fs_out must be positive ints. Returns float32 array.
    Implementation uses gcd to get integer up/down factors for resample_poly.
    """
    fs_in = int(fs_in)
    fs_out = int(fs_out)
    if fs_in == fs_out:
        return signal.astype(np.float32)

    g = math.gcd(fs_out, fs_in)
    up = fs_out // g
    down = fs_in // g
    # resample_poly accepts up/down as ints; it filters internally
    out = resample_poly(signal, up, down)
    return out.astype(np.float32)


def adc_downsample(signal_up: np.ndarray, up_fs: int, adc_fs: int,
                   start_offset_s: float = 0.0) -> np.ndarray:
    """
    Simulate ADC sampling: pick samples from `signal_up` (sampled at up_fs Hz)
    at instants t = start_offset_s + k / adc_fs for k=0..n_samples-1.
    If n_samples is None, compute the maximal integer number of samples that fit inside
    the length of signal_up (floor(len_up * adc_fs / up_fs)).
    Uses rounding of indexes to nearest upsample index (zero-order hold analog snapshot).
    """
    up_fs = int(up_fs)
    adc_fs = int(adc_fs)
    L_up = len(signal_up)
    # default n_samples preserving same time-length

    n_samples = int(math.floor(L_up * (adc_fs / up_fs)))
    if n_samples < 1:
        n_samples = 1

    k = np.arange(n_samples, dtype=np.float64)
    times = start_offset_s + k / float(adc_fs)
    idx = np.rint(times * up_fs).astype(int)
    # clip to valid range
    idx = np.clip(idx, 0, L_up - 1)
    return signal_up[idx].astype(np.float32)


def compute_delay_samples_for_sensor(source_pos: Sequence[float],
                                     sensor_pos: Sequence[float],
                                     mic_positions: np.ndarray,
                                     speed_c: float,
                                     up_fs: int) -> np.ndarray:
    """
    Compute per-microphone integer sample offsets (non-negative) at upsample rate `up_fs`
    relative to the farthest mic.

    Parameters
    ----------
    source_pos : (3,) sequence
        World coordinates of the acoustic source.
    sensor_pos : (3,) sequence
        World coordinate of the sensor node (array origin).
    mic_positions : (M,3) numpy array
        Microphone coordinates relative to sensor_pos.
    speed_c : float
        Speed of sound (m/s).
    up_fs : int
        High internal sampling rate in Hz.

    Returns
    -------
    delay_samples : (M,) int numpy array
        delay_samples[i] = number of upsample ticks that the i-th mic is **earlier**
        than the farthest mic (i.e., how many samples to shift the baseline audio to form
        the i-th mic signal). Values are >= 0.
    """
    src = np.asarray(source_pos, dtype=float)
    sensor = np.asarray(sensor_pos, dtype=float)
    mpos = np.asarray(mic_positions, dtype=float)
    abs_pos = sensor[None, :] + mpos  # (M,3)
    distances = np.linalg.norm(abs_pos - src[None, :], axis=1)
    taus = distances / float(speed_c)
    tau_max = np.max(taus)
    deltas = tau_max - taus  # >= 0 for all mics (farthest mic delta=0)
    delay_samples = np.rint(deltas * float(up_fs)).astype(int)
    return delay_samples


def assemble_mic_highrate_signals(audio_up: np.ndarray, delay_samples: np.ndarray) -> np.ndarray:
    """
    Produce per-microphone high-rate signals (aligned to the farthest mic baseline).
    - audio_up: 1D array (length L)
    - delay_samples: (M,) ints (>=0) that indicate how many samples to shift the slice start
      for each mic: mic_signal = audio_up_padded[delay : delay + L]
    Returns signals_up: array shape (M, L) dtype float32
    """
    audio_up = np.asarray(audio_up, dtype=np.float32)
    L = len(audio_up)
    delay_samples = np.asarray(delay_samples, dtype=int)
    max_delay = int(np.max(delay_samples))
    if max_delay < 0:
        raise ValueError("delay_samples must be non-negative")

    # pad end so that audio_up_padded[delay + L - 1] exists for all delays
    if max_delay > 0:
        pad = np.zeros(max_delay, dtype=audio_up.dtype)
        audio_up_padded = np.concatenate([audio_up, pad])
    else:
        audio_up_padded = audio_up

    M = len(delay_samples)
    signals = np.zeros((M, L), dtype=np.float32)
    for i, d in enumerate(delay_samples):
        start = int(d)
        signals[i, :] = audio_up_padded[start:start + L]
    return signals


def save_signals_npz(path: str, signals: np.ndarray, fs: int):
    """
    Save multichannel signals into a .npz file (signals, fs).
    signals shape (M, N)
    """
    np.savez(path, signals=signals.astype(np.float32), fs=int(fs))


def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def quat_conjugate(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def normalize_quat(q):
    """Return normalized quaternion (w,x,y,z)."""
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def quat_to_rotmat(q):
    """
    Convert quaternion (w,x,y,z) to 3x3 rotation matrix.
    Rotates a vector in local frame into global frame by R @ v_local.
    """
    w, x, y, z = q

    # Normalize to ensure unit quaternion
    n = w*w + x*x + y*y + z*z
    if n < 1e-8:
        return np.eye(3)
    s = 2.0 / n

    # Compute products
    wx, wy, wz = w*x, w*y, w*z
    xx, xy, xz = x*x, x*y, x*z
    yy, yz = y*y, y*z
    zz = z*z

    R = np.array([
        [1 - s*(yy + zz),     s*(xy - wz),       s*(xz + wy)],
        [s*(xy + wz),         1 - s*(xx + zz),   s*(yz - wx)],
        [s*(xz - wy),         s*(yz + wx),       1 - s*(xx + yy)]
    ])

    return R


def quat_rotate(q, v):
    """
    Rotate 3-vector v using quaternion q (w,x,y,z).
    Equivalent to R @ v where R = quat_to_rotmat(q).
    """
    R = quat_to_rotmat(q)
    return R @ np.asarray(v, dtype=float)


def plot_aoa_and_signals(aoa_list,
                         aoa_full_rec,
                         signals_adc,
                         fs_mic,
                         sensor_name,
                         frame_length):
    import numpy as np
    import matplotlib.pyplot as plt

    print("=== DEBUG INFO: plot_aoa_and_signals ===")
    print(f"type(aoa_list)={type(aoa_list)}, len(aoa_list)={len(aoa_list)}")
    if len(aoa_list) > 0:
        print(
            f"type(aoa_list[0])={type(aoa_list[0])}, shape={np.shape(aoa_list[0])}")
    print(f"type(aoa_full_rec)={type(aoa_full_rec)}, len={len(aoa_full_rec)}")
    print(
        f"type(signals_adc)={type(signals_adc)}, shape={np.shape(signals_adc)}")
    print(f"fs_mic={fs_mic}, type={type(fs_mic)}")
    print(f"frame_length={frame_length}, type={type(frame_length)}")
    print("========================================")

    aoa_array = np.vstack(aoa_list)  # shape (n_frames, n_dims)
    n_frames = len(aoa_list)
    n_dims = int(aoa_array.shape[1])

    # Convert frame_length [seconds] -> samples
    frame_length_samples = int(round(frame_length * fs_mic))
    total_length = n_frames * frame_length_samples

    print(
        f"frame_length={frame_length}s → frame_length_samples={frame_length_samples}, total_length={total_length}")

    # Create aoa_signal: shape (n_dims, total_length)
    aoa_signal = np.zeros((n_dims, total_length), dtype=np.float32)
    for i in range(n_frames):
        aoa_signal[:, i * frame_length_samples:(i + 1)
                   * frame_length_samples] = aoa_array[i][:, None]

    # Truncate signals_adc to total_length
    signals_adc = np.asarray(signals_adc)
    signals_adc_trunc = signals_adc[:, :total_length]

    t = np.arange(total_length) / fs_mic

    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    print(f"t.shape={t.shape}, t[0:5]={t[0:5]}")

    for i in range(n_dims):
        print(f"Plotting AOA dim {i}")
        axs[0].plot(t, aoa_signal[i], label=f"AOA dim {i}")
        if len(aoa_full_rec) > i:
            axs[0].hlines(aoa_full_rec[i], t[0], t[-1],
                          colors=f"C{i}", linestyles="dashed", label=f"Full rec dim {i}")
        else:
            print(f"⚠️ Warning: aoa_full_rec has no entry for dim {i}")

    axs[0].set_title(f"AOA signal over time for {sensor_name}")
    axs[0].set_ylabel("AOA vector components")
    axs[0].legend()
    axs[0].grid(True)

    # Plot ADC signals for each mic
    for i, s in enumerate(signals_adc_trunc):
        print(f"Plotting Mic {i}, signal length={len(s)}")
        axs[1].plot(t, s, label=f"Mic {i}")
    axs[1].set_title(f"ADC Signals for {sensor_name}")
    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel("Voltage [V]")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()
