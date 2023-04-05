import subprocess

import numpy as np
import pandas as pd

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


def animate_stable_diffusion(
    piece: MidiPiece,
    title: str,
    movie_path: str = "tmp/tmp.mp4",
    cmap="PuBuGn",
):
    T = 200
    alpha_cumprod = cosine_schedule(T)
    pieces = diffusion_process.scheduled_diffusion(
        midi_piece=piece,
        alpha_cumprod=alpha_cumprod,
        diffuse_start=False,
        diffuse_velocity=True,
    )

    # We want to see the reverse diffusion as well
    pieces = 10 * pieces[:1] + pieces + pieces[::-1] + 10 * pieces[:1]
    # pieces = pieces[::-1]
    evolved_piece = merge_diffused_pieces(pieces)

    chart_data = [p.source["diffusion_v_amplitude"] for p in pieces]

    scene = evolution_animation.EvolvingPianoRollSceneWithChart(
        pieces,
        chart_data=chart_data,
        title=title,
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


def animate_step_diffusion(
    piece: MidiPiece,
    title: str = "diffusion",
    movie_path: str = "tmp/tmp.mp4",
    cmap="PuBuGn",
):
    diffusion_amplitudes = [0.8, 0.5, 0.25, 0]
    time_step = np.ceil(piece.duration + 0.1)
    noise = np.random.normal(size=piece.size, scale=0.7)

    # Time in seconds
    max_t_shift = 0.05

    pieces = []
    for it, amp in enumerate(diffusion_amplitudes):
        next_frame = piece.df.copy()
        time_shift = it * time_step

        source = dict(piece.source)

        t_amplitude = max_t_shift * amp
        start_noise = t_amplitude * noise
        next_frame.start += start_noise

        next_frame.start += time_shift
        next_frame.end += time_shift
        source["diffusion_t_amplitude"] = t_amplitude

        # v_noise = v_amplitude * noise
        v_amplitude = amp * 0.6
        velocity = next_frame.velocity.values / 127 - 0.5
        velocity = np.sqrt(1 - amp) * velocity + noise * v_amplitude
        velocity = 127 * (velocity + 0.5)
        next_frame.velocity = velocity.clip(0, 127)

        # next_frame.velocity = (np.sqrt(alpha_cumulative) * next_frame.velocity + v_noise).clip(0, 127)
        source["diffusion_v_amplitude"] = v_amplitude

        next_piece = MidiPiece(df=next_frame, source=source)
        pieces.append(next_piece)

    chart_data = [p.source["diffusion_v_amplitude"] for p in pieces]

    new_piece = MidiPiece(df=pd.concat([p.df for p in pieces]))

    scene = evolution_animation.EvolvingPianoRollSceneWithChart(
        [new_piece] * len(chart_data),
        chart_data=chart_data,
        title=title,
        cmap=cmap,
    )
    scene_frames_dir = scene.render_mp()

    mp3_path = midi_to_mp3(new_piece.to_midi())

    command = f"""
        ffmpeg -y -f image2 -framerate 30 -i {str(scene_frames_dir)}/10%4d.png\
        -loglevel quiet -i {mp3_path} -map 0:v:0 -map 1:a:0\
        {movie_path}
    """
    print("Rendering a movie to file:", movie_path)
    subprocess.call(command, shell=True)

    return pieces
