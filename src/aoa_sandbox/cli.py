import click
import numpy as np
import soundfile as sf
from aoa_sandbox.sim import simulate_event


@click.group()
def cli():
    """AoA Sandbox CLI"""
    pass


@cli.command()
@click.argument("wavfile", type=click.Path(exists=True))
@click.option("--sr", default=48000, help="Target microphone sample rate (Hz)")
@click.option("--mic-array", default="square", help="Microphone array type (square, linear, etc.)")
@click.option("--sound-x", default=1.0, help="Sound source X position (m)")
@click.option("--sound-y", default=0.0, help="Sound source Y position (m)")
def run(wavfile, sr, mic_array, sound_x, sound_y):
    """Run AoA estimation from a WAV file"""
    audio, fs = sf.read(wavfile)
    print(f"Loaded {wavfile} with SR={fs}, {audio.shape[0]} samples")
    est = simulate_event(audio, fs, sr, mic_array, (sound_x, sound_y))
    click.echo(f"Estimated source position: {180 + np.degrees(est)}")
