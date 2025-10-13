import click
import toml
import numpy as np

from aoa_sandbox.aoa import estimate_aoa
from aoa_sandbox.toa import estimate_position_from_toa
from aoa_sandbox.utils import quat_to_rotmat
from aoa_sandbox.visuals import plot_sensor_results

from .sim import compute_aoa_over_frames, simulate_event
from .fusion import triangulate
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R


@click.group()
def cli():
    pass


@cli.command()
@click.argument("config_file", type=click.Path(exists=True))
@click.option("--w", type=float, help="Time window length for AOA estimation", default=0.02)
@click.option("--plot_signals", is_flag=True, help="Plot signals and AOA")
def run(config_file, w, plot_signals):
    config = toml.load(config_file)

    source_pos = np.array(config["source"]["position"])
    source_spl_db = config["source"]["spl_db"]
    sensors = config["sensor"]
    fs_mic = config["simulation"]["fs_mic"]
    up_fs = config["simulation"]["up_fs"]
    sound_file = config["source"]["sound_file"]

    extra_kwargs = {}
    if plot_signals:
        extra_kwargs["plot_signals"] = True

    # Run the simulation
    results = simulate_event(source_pos, sensors, fs_mic,
                             up_fs, sound_file, source_spl_db,
                             window_length=w, **extra_kwargs)

    aoas = [results[s]["aoa"] for s in results]  # type: ignore
    aoa_full_recs = [results[s]["aoa_full_rec"]
                     for s in results]  # type: ignore
    positions = [results[s]["pos"] for s in results]  # type: ignore

    est_pos = triangulate(aoas, positions)
    est_pos_full = triangulate(aoa_full_recs, positions)

    click.echo(f"True source position: {source_pos}")
    click.echo(f"Estimated position (frames):   {est_pos}")
    click.echo(f"Estimated position (full rec): {est_pos_full}")
    import matplotlib.pyplot as plt

    positions = np.array(positions)
    est_pos = np.array(est_pos)
    source_pos = np.array(source_pos)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(positions[:, 0],
               positions[:, 1],
               positions[:, 2],  # type: ignore
               c='blue',
               label='Sensors')
    ax.scatter(source_pos[0],
               source_pos[1],
               source_pos[2],
               c='green',
               marker='*', s=200, label='True Source')
    ax.scatter(est_pos[0],
               est_pos[1],
               est_pos[2],
               c='orange',
               marker='x', s=100, label='Estimated Source')
    ax.scatter(est_pos_full[0],
               est_pos_full[1],
               est_pos_full[2],
               c='cyan',
               marker='x', s=100, label='Estimated Source (full rec)')

    # Calculate min and max values for axis limits based on all points
    all_points = np.vstack(
        [positions, source_pos.reshape(1, -1), est_pos.reshape(1, -1)])
    margin = 0.1 * (all_points.max(axis=0) - all_points.min(axis=0))
    x_min, y_min, z_min = all_points.min(axis=0) - margin
    x_max, y_max, z_max = all_points.max(axis=0) + margin

    # # Draw AOAs as vectors from each sensor position
    for i, (pos, aoa) in enumerate(zip(positions, aoas)):
        aoa_vec = np.array(aoa)
        label = "AOA - frames" if i == 0 else None
        ax.quiver(pos[0], pos[1], pos[2],
                  aoa_vec[0], aoa_vec[1], aoa_vec[2],
                  length=10, color='orange', arrow_length_ratio=0.05, label=label)
    # Draw AOAs from full recordings as vectors from each sensor position
    for i, (pos, aoa_full) in enumerate(zip(positions, aoa_full_recs)):
        aoa_vec = np.array(aoa_full)
        label = "AOA - full rec" if i == 0 else None
        ax.quiver(pos[0], pos[1], pos[2],
                  aoa_vec[0], aoa_vec[1], aoa_vec[2],
                  length=10, color='cyan', arrow_length_ratio=0.05, label=label)

    # Draw sensor orientation arrows using quaternions
    for sensor in sensors:
        sensor_pos = np.array(sensor["position"])
        R = quat_to_rotmat(sensor["quaternion"])
        # Sensor's forward direction in local frame (e.g. x-axis)
        forward = R @ np.array([1, 0, 0])
        ax.quiver(sensor_pos[0], sensor_pos[1], sensor_pos[2],
                  forward[0], forward[1], forward[2],
                  length=0.1, color='purple', arrow_length_ratio=0.1, label=None)  # type: ignore

    # Plot small dots for microphone positions
    for sensor in sensors:
        sensor_pos = sensor["position"]
        R = quat_to_rotmat(sensor["quaternion"])
        mic_local = sensor["mics"]
        mic_positions = np.array([sensor_pos + R @ m for m in mic_local])
        for mic_pos in mic_positions:
            mic_pos_arr = np.array(mic_pos)
            ax.scatter(mic_pos_arr[0], mic_pos_arr[1], mic_pos_arr[2],
                       c='black', marker='o', s=10, label=None)

    # ax.set_xlim(x_min, x_max)
    # ax.set_ylim(y_min, y_max)
    # ax.set_zlim(z_min, z_max)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Sensor Scenario (3D)')

    sensor_positions = [res["toa_mic"] for res in results.values()]
    toas = [res["toa"] for res in results.values()]
    est_pos = estimate_position_from_toa(sensor_positions, toas)
    click.echo(f"True source position: {source_pos}")
    click.echo(f"Estimated position from ToA:   {est_pos}")

    # Plot estimated position from ToA
    est_pos_toa = np.array(est_pos)
    ax.scatter(est_pos_toa[0],
               est_pos_toa[1],
               est_pos_toa[2],
               c='magenta',
               marker='^', s=120, label='Estimated Source (ToA)')

    ax.legend()
    plt.show()

    # plot_sensor_results(results, sensor_name=None,
    #                     fs=fs_mic, max_samples=40000)


@cli.command()
@click.argument("config_file", type=click.Path(exists=True))
@click.option("--w", type=float, help="Time window length for AOA estimation", default=0.02)
@click.option("--plot_signals", is_flag=True, help="Plot signals and AOA")
@click.option("--wav_file", type=click.Path(exists=True), required=True, help="Input wave file path")
@click.option("--start", type=float, default=0.0, help="Start time (seconds) of the audio section to process")
@click.option("--length", type=float, default=None, help="Length (seconds) of the audio section to process")
def aoa(config_file, w, plot_signals, wav_file, start, length):
    from scipy.io import wavfile
    from .utils import plot_aoa_and_signals
    # Load config
    cfg = toml.load(config_file)
    mic_positions = np.array(cfg["microphones"]["positions"], dtype=float)

    # Load 4-channel WAV
    sr, sig = wavfile.read(wav_file)
    if sig.ndim != 2 or sig.shape[1] != 4:  # type: ignore
        raise ValueError("Expected a 4-channel WAV file")
    sig = sig.astype(np.float32) / 2**14
    start_sample = int(start * sr)
    if length is not None:
        end_sample = start_sample + int(length * sr)
        sig = sig[start_sample:end_sample, :]  # type: ignore
    else:
        sig = sig[start_sample:, :]  # type: ignore
    signals = [sig[:, i] for i in range(4)]  # type: ignore

    # Run AoA estimation
    aoa_vectors, aoa_agg = compute_aoa_over_frames(
        signals=signals,
        fs=sr,
        mic_positions=mic_positions,
        source_pos=None,    # Unknown
        sensor_pos=None,    # Unknown
        frame_length=w,
        aggregation="median"
    )

    aoa_full_rec = estimate_aoa(
        signals,
        sr,
        mic_positions,
        source_pos=None,
        sensor_pos=None
    )

    if plot_signals:
        plot_aoa_and_signals(aoa_list=aoa_vectors,
                             aoa_full_rec=aoa_full_rec,
                             signals_adc=signals,
                             fs_mic=sr,
                             sensor_name="Test Sensor",
                             frame_length=w)


@cli.command()
@click.option("--process_only", is_flag=True, help="Process only, do not plot")
@click.option("--wav_file", type=click.Path(exists=True), required=True, help="Input wave file path")
@click.option("--start", type=float, default=0.0, help="Start time (seconds) of the audio section to process")
@click.option("--length", type=float, default=None, help="Length (seconds) of the audio section to process")
@click.option("--output_wav", type=click.Path(), help="Output WAV file path")
def onset_detection(wav_file, start, length, process_only, output_wav):
    import numpy as np
    import librosa
    import soundfile as sf
    import matplotlib.pyplot as plt

    from aoa_sandbox.onset import superflux_general, stretch_away

    sr = 96000
    win_length = int(0.025 * sr)   # 2400 samples
    hop_length = int(0.010 * sr)   # 960 samples
    n_fft = 4096                   # power-of-two >= win_length for efficiency
    n_mels = 64                   # typical mel filterbank size

    chunk_size = sr * 1  # 10 seconds of audio per chunk

    if output_wav:
        out_sf = sf.SoundFile(output_wav, mode='w',
                              samplerate=sr, channels=2, subtype='FLOAT')
    else:
        out_sf = None

    with sf.SoundFile(wav_file, 'r') as f:
        # Calculate start and end frames
        start_frame = int(start * sr)
        if length is not None:
            end_frame = start_frame + int(length * sr)
        else:
            end_frame = f.frames

        f.seek(start_frame)
        frames_to_read = end_frame - start_frame
        print(
            f"Will process {frames_to_read} frames\n_______________________________")

        while frames_to_read > 0:
            read_frames = min(chunk_size, frames_to_read)
            data = f.read(frames=read_frames, dtype='float32')
            if len(data) == 0:
                break

            frames_to_read -= len(data)
            print("Frames left: ", frames_to_read)

            # If stereo, convert to mono
            if data.ndim > 1:
                data = data[:, 0]

            # Compute mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=data,
                sr=sr,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                n_mels=n_mels,
                power=2.0
            )
            mel_spec = librosa.power_to_db(mel_spec, ref=0.001)

            env_mel, t_mel, _ = superflux_general(
                mel_spec, sr=sr, hop_length=hop_length, lag=1, max_size=5)
            env_mel = stretch_away(env_mel, t=0.5, alpha=2.0)

            if not process_only:
                plt.figure(figsize=(12, 4))
                t_signal = np.arange(len(data)) / sr + \
                    (f.tell() - len(data)) / sr
                plt.plot(t_signal, data,
                         label="Original Signal", alpha=0.7)
                plt.plot(t_mel + t_signal[0], env_mel, label="SuperFlux (Mel)",
                         linewidth=2, alpha=0.8)
                plt.xlabel("Time (s)")
                plt.ylabel("Amplitude / Onset Strength")
                plt.title("Original Signal and SuperFlux Onset Envelope")
                plt.ylim(-1.1, 1.1)
                plt.legend()
                plt.tight_layout()
                plt.show()
            else:
                import scipy.signal

                # Upsample envelope to match original signal length using rectangular (step) repetition
                repeats = int(np.ceil(len(data) / len(env_mel)))
                env_mel_upsampled = np.repeat(env_mel, repeats)[:len(data)]

                # Stack original and envelope as two channels
                output = np.stack([data, env_mel_upsampled], axis=-1)

                # Write to output file
                if out_sf is not None:
                    out_sf.write(output)

            try:
                pass  # All processing is above
            except KeyboardInterrupt:
                print("Interrupted by user. Exiting.")
                if out_sf is not None:
                    out_sf.close()
                break
