import subprocess

import numpy as np

from fortepyan.audio.render import midi_to_mp3
from fortepyan.midi.structures import MidiPiece
from fortepyan.animation import evolution as evolution_animation


def diffuse_midi_piece(midi_piece: MidiPiece) -> list[MidiPiece]:
    def get_random(size):
        return np.random.random(size) - 0.5

    n_steps = 100
    midi_piece.source["diffusion_t_amplitude"] = 0
    midi_piece.source["diffusion_v_amplitude"] = 0
    diffused = [midi_piece]
    for it in range(n_steps):
        piece = diffused[-1]

        # TODO This is a poor mans linear beta schedule
        amplitude = it / 30
        v_amplitude = 8 * amplitude
        t_amplitude = 0.00 * amplitude
        s_noise = get_random(piece.size) * t_amplitude
        e_noise = get_random(piece.size) * t_amplitude
        v_noise = get_random(piece.size) * v_amplitude
        next_frame = piece.df.copy()
        next_frame.start += s_noise
        next_frame.end += e_noise
        next_frame.velocity += v_noise
        next_frame.velocity = next_frame.velocity.clip(0, 127)

        # Make a copy of the source info ...
        source = dict(piece.source)
        # ... and note the diffusion info
        source["diffusion_step"] = it
        source["diffusion_t_amplitude"] = t_amplitude
        source["diffusion_v_amplitude"] = v_amplitude
        next_piece = MidiPiece(
            df=next_frame,
            sustain=piece.sustain,
            source=source,
        )
        diffused.append(next_piece)

    # Move all pieces, so nothing starts before 0s
    start_shift = min([piece.df.start.min() for piece in diffused])
    for piece in diffused:
        piece.time_shift(-start_shift)

    return diffused


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
        sustain=pieces[0].sustain,
        source=source,
    )

    return new_piece


def animate_diffusion(
    piece: MidiPiece,
    movie_path: str = "tmp/tmp.mp4",
    cmap="PuBuGn",
):
    pieces = diffuse_midi_piece(piece)

    # We want to see the reverse diffusion as well
    pieces = 20 * pieces[:1] + pieces + pieces[::-1] + 20 * pieces[:1]
    evolved_piece = merge_diffused_pieces(pieces)

    scene = evolution_animation.EvolvingPianoRollScene(
        pieces,
        title_format="Diffusion Amplitude: {:.2f}",
        title_key="diffusion_v_amplitude",
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
