import numpy as np
from scipy.io import wavfile

from .utils import quat_rotate, quat_to_rotmat, speed_of_sound, adc_downsample
from .aoa import estimate_aoa
import scipy.signal as sps


def simulate_event(source_pos, sensors, fs_mic=48000, up_fs=1_000_000, sound_file=None):
    """
    Simulate event for multiple sensors with mic arrays and quaternions.
    Returns: {sensor_name: {signals, aoa, pos, mic_positions}}
    """
    if sound_file is None:
        raise ValueError("sound_file must be provided")

    sr, sig = wavfile.read(sound_file)
    if sig.ndim > 1:
        sig = sig.mean(axis=1)  # force mono
    sig = sig.astype(np.float32)
    sig /= np.max(np.abs(sig))  # normalized signal

    # Upsample input waveform
    sig_up = sps.resample_poly(sig, up_fs, sr)
    L = len(sig_up)

    results = {}
    for sensor in sensors:
        sensor_pos = sensor["position"]
        R = quat_to_rotmat(sensor["quaternion"])
        mic_local = sensor["mics"]
        mic_positions = np.array([sensor_pos + R @ m for m in mic_local])

        dists = np.array([np.linalg.norm(source_pos - mic)
                         for mic in mic_positions])
        min_dist = np.min(dists)
        toa = min_dist / speed_of_sound()
        delays = (dists - min_dist) / speed_of_sound()
        max_delay = max(delays)
        delay_samples = [int(round((d - max_delay) * up_fs)) for d in delays]

        tdoas = [delays[0]-d for d in delays]

        # Apply delays
        signals = []
        for ds in delay_samples:
            if ds == 0:
                sig_shifted = sig_up
            else:
                start = -ds  # ds is negative for closer mics
                sig_shifted = sig_up[start:] if start >= 0 else np.pad(
                    sig_up, (-start, 0))
            sig_shifted = sig_shifted[:L]
            signals.append(sig_shifted)

        # Downsample to mic fs
        signals_ds = [sps.resample_poly(s, fs_mic, up_fs) for s in signals]

        # AoA estimation
        aoa_vec = estimate_aoa(signals_ds, fs_mic, mic_positions)

        results[sensor["name"]] = {
            "signals": signals_ds,
            "aoa": aoa_vec,
            "pos": sensor_pos,
            "mic_positions": mic_positions,
            "toa": toa,
            "toa_mic": mic_positions[np.argmin(dists)]
        }

    return results
