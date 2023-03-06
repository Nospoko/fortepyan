import subprocess

import numpy as np

from fortepyan.audio.render import midi_to_mp3
from fortepyan.midi.structures import MidiPiece
from fortepyan.animation import evolution as evolution_animation
from fortepyan.demo.diffusion import process as diffusion_process


def merge_diffused_pieces(pieces: list[MidiPiece]) -> MidiPiece:
    n_steps = len(pieces)
    duration = max([piece.df.end.max() for piece in pieces])
    time_per_step = duration / n_steps

    df = pieces[0].df.copy()

    # During a diffusion process, the start of the note can
    # show up during two different diffusion steps. We want
    # to be sure we hear only the first start, without the
    # note repetition, if there is a second start later.
    # We don't need this for *end*, because we want the latest
    # value
    df["started"] = False

    start = 0
    finish = time_per_step
    for piece in pieces:
        # Take note start and volume
        start_ids = (piece.df.start >= start) & (piece.df.start < finish)
        update_start_ids = start_ids & (~df.started)
        df.loc[update_start_ids, "start"] = piece.df[update_start_ids].start
        df.loc[update_start_ids, "velocity"] = piece.df[update_start_ids].velocity
        df.loc[update_start_ids, "started"] = True

        # Take note end
        update_end_ids = (piece.df.end >= start) & (piece.df.end < finish)
        df.loc[update_end_ids, "end"] = piece.df[update_end_ids].end

        # Go to next diffusion step
        start += time_per_step
        finish += time_per_step

    source = dict(pieces[0].source)
    source["history"] = "evolution of the diffusion process"
    new_piece = MidiPiece(
        df=df,
        source=source,
    )

    return new_piece


def cosine_schedule(T: int) -> np.array:
    steps = T + 1
    X = np.linspace(0, T, steps)
    s = 0.008
    alpha_cumprod = np.cos(((X / T) + s) / (1 + s) * np.pi * 0.5) ** 2
    return alpha_cumprod


def sigmoid_schedule(T: int) -> np.array:
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    beta_start = 0.0001
    beta_end = 1

    X = np.linspace(-6, 6, T)
    betas = sigmoid(X) * (beta_end - beta_start) + beta_start

    alphas = 1.0 - betas
    alpha_cumprod = np.cumprod(alphas)

    return alpha_cumprod


def animate_diffusion(
    piece: MidiPiece,
    movie_path: str = "tmp/tmp.mp4",
    cmap="PuBuGn",
):
    T = 200
    alpha_cumprod = cosine_schedule(T)
    pieces = diffusion_process.scheduled_diffusion(piece, alpha_cumprod)

    # We want to see the reverse diffusion as well
    pieces = 100 * pieces[:1] + pieces + pieces[::-1] + 100 * pieces[:1]
    # pieces = pieces[::-1]
    evolved_piece = merge_diffused_pieces(pieces)

    scene = evolution_animation.EvolvingPianoRollScene(
        pieces,
        title_format="Mean Time Error: {:.3f} [s]",
        title_key="diffusion_t_amplitude",
        cmap=cmap,
    )
    scene_frames_dir = scene.render_mp()

    mp3_path = midi_to_mp3(evolved_piece.to_midi())

    command = f"""
        ffmpeg -y -f image2 -framerate 30 -i {str(scene_frames_dir)}/10%4d.png\
        -loglevel quiet -i {mp3_path} -map 0:v:0 -map 1:a:0\
        {movie_path}
    """
    print("Rendering a movie to file:", movie_path)
    subprocess.call(command, shell=True)
