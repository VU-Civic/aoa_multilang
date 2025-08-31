import click
import toml
import numpy as np

from .sim import simulate_event
from .fusion import triangulate


@click.group()
def cli():
    pass


@cli.command()
@click.argument("config_file", type=click.Path(exists=True))
def run(config_file):
    config = toml.load(config_file)

    source_pos = np.array(config["source"]["pos"])
    sensors = config["sensors"]
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

    plt.figure(figsize=(6, 6))
    plt.scatter(positions[:, 0], positions[:, 1], c='blue', label='Sensors')
    plt.scatter(source_pos[0], source_pos[1], c='green',
                marker='*', s=200, label='True Source')
    plt.scatter(est_pos[0], est_pos[1], c='red',
                marker='x', s=100, label='Estimated Source')

    for i, pos in enumerate(positions):
        plt.text(pos[0], pos[1], f"S{i}", fontsize=9, ha='right')

    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Sensor Scenario')
    plt.grid(True)
    plt.axis('equal')
    plt.show()
