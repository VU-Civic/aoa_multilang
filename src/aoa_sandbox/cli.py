import click
import toml
import numpy as np

from aoa_sandbox.utils import quat_to_rotmat

from .sim import simulate_event
from .fusion import triangulate
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R


@click.group()
def cli():
    pass


@cli.command()
@click.argument("config_file", type=click.Path(exists=True))
def run(config_file):
    config = toml.load(config_file)

    source_pos = np.array(config["source"]["position"])
    sensors = config["sensor"]
    fs_mic = config["simulation"]["fs_mic"]
    up_fs = config["simulation"]["up_fs"]
    sound_file = config["simulation"]["sound_file"]

    results = simulate_event(source_pos, sensors, fs_mic, up_fs, sound_file)
    aoas = [results[s]["aoa"] for s in results]  # type: ignore
    positions = [results[s]["pos"] for s in results]  # type: ignore

    est_pos = triangulate(aoas, positions)

    click.echo(f"True source position: {source_pos}")
    click.echo(f"Estimated position:   {est_pos}")
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
               c='red',
               marker='x', s=100, label='Estimated Source')

    # Calculate min and max values for axis limits based on all points
    all_points = np.vstack(
        [positions, source_pos.reshape(1, -1), est_pos.reshape(1, -1)])
    margin = 0.1 * (all_points.max(axis=0) - all_points.min(axis=0))
    x_min, y_min, z_min = all_points.min(axis=0) - margin
    x_max, y_max, z_max = all_points.max(axis=0) + margin

    # # Draw AOAs as vectors from each sensor position
    for pos, aoa in zip(positions, aoas):
        # Normalize AOAs for visualization
        aoa_vec = np.array(aoa)
        aoa_vec = aoa_vec / np.linalg.norm(aoa_vec)
        ax.quiver(pos[0], pos[1], pos[2],
                  aoa_vec[0], aoa_vec[1], aoa_vec[2],
                  length=10, color='orange', arrow_length_ratio=0.05, label=None)

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
        print(mic_positions)
        for mic_pos in mic_positions:
            mic_pos_arr = np.array(mic_pos)
            ax.scatter(mic_pos_arr[0], mic_pos_arr[1], mic_pos_arr[2],
                       c='black', marker='o', s=10, label=None)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Sensor Scenario (3D)')
    plt.show()
