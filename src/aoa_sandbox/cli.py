import click
import toml
import numpy as np

from .sim import simulate_event
from .fusion import triangulate
from mpl_toolkits.mplot3d import Axes3D


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

    for i, pos in enumerate(positions):
        ax.text(pos[0], pos[1], pos[2], f"S{i}", fontsize=9, ha='right')

    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Sensor Scenario (3D)')
    plt.show()
