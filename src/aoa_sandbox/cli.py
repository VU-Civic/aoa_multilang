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
