import numpy as np
from scipy.io import wavfile
import scipy.signal as sps

from aoa_sandbox.propagation import apply_propagation
from .utils import adc_downsample, quat_to_rotmat, speed_of_sound
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


def simulate_event(source_pos, sensors,
                   fs_mic=48000,
                   up_fs=1_000_000,
                   sound_file=None,
                   source_spl_db=100.0,
                   noise_std=0.01):
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
                sig_shifted = sig_up[start:] if start >= 0 else np.pad(
                    sig_up, (-start, 0))
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
        aoa_vec = estimate_aoa(signals_adc, fs_mic, mic_positions)

        results[sensor["name"]] = {
            "signals": signals_adc,
            "aoa": aoa_vec,
            "pos": sensor_pos,
            "mic_positions": mic_positions,
            "toa": toa,
            "toa_mic": mic_positions[np.argmin(dists)]
        }

    return results
