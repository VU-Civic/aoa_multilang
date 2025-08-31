import numpy as np
from scipy.io import wavfile

from .utils import speed_of_sound, adc_downsample
from .aoa import estimate_aoa


def simulate_event(source_pos, sensors, fs_mic=48000, up_fs=1_000_000, sound_file=None):
    """
    Simulate event for multiple sensors.
    Returns: {sensor_name: (signals, aoa_vector)}
    """
    if sound_file is None:
        raise ValueError("sound_file must be provided")

    sr, sig = wavfile.read(sound_file)
    if sig.ndim > 1:
        sig = sig.mean(axis=1)  # mono

    sig = sig.astype(np.float32)
    sig /= np.max(np.abs(sig))

    # Upsample to fine grid
    import scipy.signal as sps
    sig_up = sps.resample_poly(sig, up_fs, sr)
    results = {}

    for sensor in sensors:
        sensor_pos = np.array(sensor["pos"])
        mic_positions = np.array(sensor["mics"])

        delays = []
        for mic in mic_positions:
            dx = source_pos[0] - mic[0]
            dy = source_pos[1] - mic[1]
            dist = np.sqrt(dx**2 + dy**2)
            delay_sec = dist / speed_of_sound()
            delays.append(delay_sec)
        max_delay = max(delays)
        delay_samples = [int((d - max_delay) * up_fs) for d in delays]

        # Generate mic signals (same length, aligned to farthest mic)
        signals = []
        L = len(sig_up)
        for ds in delay_samples:
            if ds == 0:
                # farthest mic → keep as-is
                sig = sig_up
            else:
                # closer mic → trim the beginning
                start = -ds  # ds is negative
                sig = sig_up[start:]
            # ensure consistent length
            sig = sig[:L]
            signals.append(sig)

        # 4. Downsample back to microphone sample rate
        signals_ds = [sps.resample_poly(s, fs_mic, up_fs) for s in signals]

        # AoA estimation
        aoa_vec = estimate_aoa(signals_ds, fs_mic, mic_positions)

        results[sensor["name"]] = {
            "signals": signals_ds,
            "aoa": aoa_vec,
            "pos": sensor_pos,
        }

    return results
