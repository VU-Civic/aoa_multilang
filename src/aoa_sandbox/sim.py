from logging import config
import numpy as np
from scipy.io import wavfile
import scipy.signal as sps

from aoa_sandbox.propagation import apply_propagation
from .utils import adc_downsample, quat_to_rotmat, speed_of_sound, plot_aoa_and_signals
from .aoa import estimate_aoa
from .spl import spl_to_pressure


def apply_mic_chain(pressure_signal: np.ndarray,
                    mic_sensitivity_v_per_pa: float,
                    gain: float,
                    noise_std: float) -> np.ndarray:
    """
    Convert pressure waveform [Pa] to voltage at ADC input.
    - mic_sensitivity_v_per_pa: microphone sensitivity
    - gain: preamp gain (linear)
    Returns: ADC input voltage signal clipped to [-1, 1] V
    """
    # Mic converts Pa → Volts
    v_signal = pressure_signal * mic_sensitivity_v_per_pa
    # Apply amplifier gain
    v_signal *= gain
    v_signal = v_signal + np.random.normal(0, noise_std, size=v_signal.shape)
    # Clip to ADC full scale [-1, 1]
    v_signal = np.clip(v_signal, -1.0, 1.0)
    return v_signal


def compute_aoa_over_frames(signals,
                            fs,
                            mic_positions,
                            source_pos=None,
                            sensor_pos=None,
                            frame_length=0.015,
                            aggregation="histogram"):
    """
    Compute AOA vectors for multiple time frames.

    Args:
        signals: list of ndarray, shape (n_sensors, n_samples)
        fs: float
            Sampling frequency [Hz].
        mic_positions: ndarray, shape (n_sensors, 3)
            Cartesian coordinates of microphones.
        frame_length: float
            Frame duration in seconds (default: 15 ms).
        aggregation: str
            Aggregation method for global AOA ("mean", "median").

    Returns:
        aoa_list: list of ndarray
            List of AOA vectors, one per frame.
        aoa_agg: ndarray
            Aggregated AOA vector.
    """
    n_signals = len(signals)
    if n_signals < 2:
        raise ValueError(
            "At least two signals are required for AOA estimation.")
    signals = np.array(signals)
    n_samples = signals.shape[1]
    frame_size = int(frame_length * fs)
    n_frames = n_samples // frame_size

    aoa_list = []

    for k in range(n_frames):
        start = k * frame_size
        end = start + frame_size
        frame_signals = signals[:, start:end]

        # --- Call your existing per-frame AOA estimation ---
        aoa_vec = estimate_aoa(
            frame_signals, fs, mic_positions, source_pos, sensor_pos)
        aoa_list.append(aoa_vec)

    # --- Aggregation ---
    aoa_array = np.vstack(aoa_list)
    if aggregation == "mean":
        aoa_agg = np.mean(aoa_array, axis=0)
    elif aggregation == "median":
        aoa_agg = np.median(aoa_array, axis=0)
    elif aggregation == "max":
        aoa_agg = aoa_array[np.argmax(np.linalg.norm(aoa_array, axis=1))]
    elif aggregation == "histogram":
        # Histogram aggregation: for each dimension, pick the most frequent bin center
        aoa_agg = np.zeros(aoa_array.shape[1])
        for i in range(aoa_array.shape[1]):
            hist, bin_edges = np.histogram(aoa_array[:, i], bins="auto")
            max_bin_idx = np.argmax(hist)
            if max_bin_idx < len(bin_edges) - 1:
                aoa_agg[i] = 0.5 * (bin_edges[max_bin_idx] +
                                    bin_edges[max_bin_idx + 1])
            else:
                # Fallback: use the last bin edge if somehow at the end
                aoa_agg[i] = bin_edges[-1]
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")

    return aoa_list, aoa_agg


def simulate_event(source_pos, sensors,
                   fs_mic=48000,
                   up_fs=1_000_000,
                   sound_file=None,
                   source_spl_db=100.0,
                   noise_std=0.01,
                   window_length=0.02,
                   **kwargs):
    """
    Simulate event for multiple sensors with mic arrays and quaternions.
    Loudness is defined in dB SPL at 1 m distance.

    Returns: {sensor_name: {signals, aoa, pos, mic_positions}}
    """
    if sound_file is None:
        raise ValueError("sound_file must be provided")

    # Load and normalize sound file
    sr, sig = wavfile.read(sound_file)
    if sig.ndim > 1:
        sig = sig.mean(axis=1)  # force mono
    sig = sig.astype(np.float32)
    sig /= np.max(np.abs(sig))  # normalized to ±1
    # Scale to desired SPL at 1 m
    target_p_rms = spl_to_pressure(source_spl_db)
    sig_rms = np.sqrt(np.mean(sig**2))
    print(
        f"Input signal RMS: {sig_rms:.6f}, target RMS: {target_p_rms:.6f} Pa")
    sig *= target_p_rms / (sig_rms + 1e-12)
    # Upsample to high rate
    sig_up = sps.resample_poly(sig, up_fs, sr)
    L = len(sig_up)

    results = {}
    for sensor in sensors:
        sensor_pos = sensor["position"]
        R = quat_to_rotmat(sensor["quaternion"])
        mic_local = sensor["mics"]
        # Compute absolute mic positions
        mic_positions = np.array([sensor_pos + R @ m for m in mic_local])

        dists = np.array([np.linalg.norm(source_pos - mic)
                          for mic in mic_positions])
        min_dist = np.min(dists)
        toa = min_dist / speed_of_sound()  # TODO: algorithm to find TOA
        delays = (dists - min_dist) / speed_of_sound()
        max_delay = max(delays)
        delay_samples = [int(round((d - max_delay) * up_fs)) for d in delays]

        signals = []
        for di, ds in zip(dists, delay_samples):
            if ds == 0:
                sig_shifted = sig_up
            else:
                start = -ds
                sig_shifted = np.pad(sig_up[start:], (0, start))
            sig_shifted = sig_shifted[:L]
            # Apply attenuation (1/r) + frequency-dependent loss
            sig_shifted = apply_propagation(sig_shifted, up_fs, di)
            signals.append(sig_shifted)
        # Apply mic chain and downsample to ADC rate
        signals_ds = [adc_downsample(s, up_fs, fs_mic) for s in signals]
        mic_sens = sensor["mic_sensitivity"]
        gain = sensor["gain"]
        signals_adc = [apply_mic_chain(s, mic_sens, gain, noise_std)
                       for s in signals_ds]
        # Estimate AOA
        aoa_list, aoa_agg = compute_aoa_over_frames(
            signals=signals_adc,
            fs=fs_mic,
            mic_positions=mic_positions,
            source_pos=source_pos,
            sensor_pos=sensor_pos,
            frame_length=window_length,
            aggregation="median"
        )

        aoa_full_rec = estimate_aoa(
            signals_adc, fs_mic, mic_positions, source_pos=source_pos)

        # Optional: plot signals and AOA
        if kwargs.get("plot_signals", False):
            plot_aoa_and_signals(aoa_list, aoa_full_rec,
                                 signals_adc, fs_mic, sensor["name"])

        results[sensor["name"]] = {
            "signals": signals_adc,
            "aoa": aoa_agg,
            "aoa_full_rec": aoa_full_rec,
            "pos": sensor_pos,
            "mic_positions": mic_positions,
            "toa": toa,
            "toa_mic": mic_positions[np.argmin(dists)]
        }

    return results
