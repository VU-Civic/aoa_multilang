import numpy as np
from scipy.signal import resample_poly
from .geometry import get_mic_positions
from .aoa import estimate_aoa

C = 343.0  # speed of sound


def adc_downsample(signal_up, up_fs, adc_fs):
    """
    Simulate ADC sampling of an upsampled signal.
    Instead of resampling with polyphase filtering,
    we directly pick samples at discrete time points.
    """
    n_samples = int(len(signal_up) * adc_fs / up_fs)
    idx = np.round(np.arange(n_samples) * (up_fs / adc_fs)).astype(int)
    idx = np.clip(idx, 0, len(signal_up) - 1)
    return signal_up[idx]


def simulate_event(audio, fs_src, fs_mic, mic_array, source_pos):
    # 1. Upsample input to ~1 MHz
    up_fs = 1_000_000
    audio_up = resample_poly(audio, up_fs, fs_src)

    # 2. Place microphones
    mic_positions = get_mic_positions(mic_array)

    # 3. Compute delays for each mic
    delays = []
    for mic in mic_positions:
        dx = source_pos[0] - mic[0]
        dy = source_pos[1] - mic[1]
        dist = np.sqrt(dx**2 + dy**2)
        delay_sec = dist / C
        delays.append(delay_sec)

    # Normalize delays so the farthest mic has 0 delay
    max_delay = max(delays)
    delay_samples = [int((d - max_delay) * up_fs) for d in delays]

    # Generate mic signals (same length, aligned to farthest mic)
    signals = []
    L = len(audio_up)
    for ds in delay_samples:
        if ds == 0:
            # farthest mic → keep as-is
            sig = audio_up
        else:
            # closer mic → trim the beginning
            start = -ds  # ds is negative
            sig = audio_up[start:]
        # ensure consistent length
        sig = sig[:L]
        signals.append(sig)

    # 4. Downsample back to microphone sample rate
    signals_ds = [adc_downsample(s, up_fs, fs_mic) for s in signals]

    # import matplotlib.pyplot as plt

    # plt.figure(figsize=(12, 6))
    # for i, sig in enumerate(signals_ds):
    #     plt.plot(sig, label=f'Mic {i}')
    # plt.title('Microphone Signals')
    # plt.xlabel('Sample')
    # plt.ylabel('Amplitude')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # 5. Estimate AoA
    est_pos = estimate_aoa(signals_ds, fs_mic, mic_positions)

    return est_pos
