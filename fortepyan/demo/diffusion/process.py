import numpy as np

from fortepyan.midi.structures import MidiPiece


def naive_diffusion(midi_piece: MidiPiece) -> list[MidiPiece]:
    def get_random(size):
        return np.random.random(size) - 0.5

    n_steps = 100
    midi_piece.source["diffusion_step"] = 0
    midi_piece.source["diffusion_t_amplitude"] = 0
    midi_piece.source["diffusion_v_amplitude"] = 0
    diffused = [midi_piece]
    for it in range(n_steps):
        piece = diffused[-1]

        # TODO This is a poor mans linear beta schedule
        amplitude = 1
        v_amplitude = 8 * amplitude
        t_amplitude = 0.03 * amplitude
        s_noise = get_random(piece.size) * t_amplitude
        d_noise = get_random(piece.size) * t_amplitude * 2
        v_noise = get_random(piece.size) * v_amplitude
        next_frame = piece.df.copy()
        next_frame.start += s_noise
        next_frame.duration += d_noise
        next_frame.velocity += v_noise
        next_frame.velocity = next_frame.velocity.clip(0, 127)

        # Make a copy of the source info ...
        source = dict(piece.source)
        # ... and note the diffusion info
        source["diffusion_step"] = it
        source["diffusion_t_amplitude"] = t_amplitude * it
        source["diffusion_v_amplitude"] = v_amplitude * it
        next_piece = MidiPiece(
            df=next_frame,
            source=source,
        )
        diffused.append(next_piece)

    # Move all pieces, so nothing starts before 0s
    start_shift = min([piece.df.start.min() for piece in diffused])
    for piece in diffused:
        piece.time_shift(-start_shift)

    return diffused


def scheduled_diffusion(
    midi_piece: MidiPiece,
    alpha_cumprod: np.array,
) -> list[MidiPiece]:
    midi_piece.source["diffusion_step"] = 0
    midi_piece.source["diffusion_t_amplitude"] = 0
    midi_piece.source["diffusion_v_amplitude"] = 0

    # Time in seconds
    max_t_shift = 0.03
    max_v_shift = 30
    noise = np.random.normal(size=midi_piece.size)

    diffused = [midi_piece]

    for it, alpha_cumulative in enumerate(alpha_cumprod):
        piece = diffused[-1]
        next_frame = midi_piece.df.copy()

        t_amplitude = max_t_shift * (1 - alpha_cumulative)
        start_noise = t_amplitude * noise
        next_frame.start += start_noise

        v_amplitude = max_v_shift * (1 - alpha_cumulative)
        v_noise = v_amplitude * noise
        next_frame.velocity = (next_frame.velocity + v_noise).clip(0, 127)

        # Make a copy of the source info ...
        source = dict(piece.source)
        # ... and note the diffusion info
        source["diffusion_step"] = it
        source["diffusion_t_amplitude"] = t_amplitude
        source["diffusion_v_amplitude"] = v_amplitude
        next_piece = MidiPiece(
            df=next_frame,
            source=source,
        )
        diffused.append(next_piece)

    return diffused
