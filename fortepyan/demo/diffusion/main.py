import numpy as np

from fortepyan.midi.structures import MidiPiece


def diffuse_midi_piece(midi_piece: MidiPiece) -> list[MidiPiece]:
    def get_random(size):
        return np.random.random(size) - 0.5

    n_steps = 200
    diffused = [midi_piece]
    for it in range(n_steps):
        piece = diffused[-1]

        # TODO This is a poor mans linear beta schedule
        s_noise = get_random(piece.size) * 0.01 * it / 20
        e_noise = get_random(piece.size) * 0.01 * it / 20
        v_noise = get_random(piece.size) * 2 * it / 20
        next_frame = piece.df.copy()
        next_frame.start += s_noise
        next_frame.end += e_noise
        next_frame.velocity += v_noise
        next_frame.velocity = next_frame.velocity.clip(0, 127)

        # Make a copy of the source info ...
        source = dict(piece.source)
        # ... and note the diffusion step
        source["diffusion_step"] = it
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
        start_ids = (piece.df.start >= start) & (piece.df.start < finish)
        end_ids = (piece.df.end >= start) & (piece.df.end < finish)

        update_start_ids = start_ids & (~df.started)
        df.loc[update_start_ids, "start"] = piece.df[update_start_ids].start
        df.loc[update_start_ids, "velocity"] = piece.df[update_start_ids].velocity
        df.loc[update_start_ids, "started"] = True

        update_end_ids = end_ids
        df.loc[update_end_ids, "end"] = piece.df[update_end_ids].end

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
