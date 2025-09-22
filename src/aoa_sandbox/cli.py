import click
import toml
import numpy as np

from aoa_sandbox.toa import estimate_position_from_toa
from aoa_sandbox.utils import quat_to_rotmat
from aoa_sandbox.visuals import plot_sensor_results

from .sim import simulate_event
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
