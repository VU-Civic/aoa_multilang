import numpy as np
import matplotlib.pyplot as plt


def plot_sensor_results(results, sensor_name, fs=48000, max_samples=None):
    """
    Plot relevant simulation information for a given sensor.

    Parameters
    ----------
    results : dict
        Output from simulate_event()
    sensor_name : str
        Which sensor to visualize
    fs : int
        Sampling rate of signals (Hz)
    max_samples : int
        Limit time-domain plot to this many samples for readability
    """
    if sensor_name is None:
        sensor_name = list(results.keys())[0]

    sensor = results[sensor_name]
    signals = sensor["signals"]
    mic_positions = sensor["mic_positions"]
    aoa = sensor["aoa"]
    toa = sensor.get("toa", None)
    pos = sensor["pos"]

    n_mics = len(signals)
    t = np.arange(len(signals[0])) / fs
    max_samples = len(signals[0]) if max_samples is None else min(
        max_samples, len(signals[0]))

    fig, axes = plt.subplots(figsize=(12, 8))
    fig.suptitle(f"Sensor: {sensor_name}", fontsize=14)

    ax = axes
    for i, s in enumerate(signals):
        ax.plot(t[:max_samples], s[:max_samples], label=f"Mic {i}")
    ax.set_title("Mic signals (time domain)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    ax.legend()

    plt.tight_layout()
    plt.show()
