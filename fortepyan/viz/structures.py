from dataclasses import field, dataclass

import numpy as np
from pretty_midi import note_number_to_name

from fortepyan.midi.structures import MidiPiece


@dataclass
class PianoRoll:
    midi_piece: MidiPiece
    current_time: float = None
    time_start: float = 0.0
    time_end: float = None

    roll: np.array = field(init=False)

    RESOLUTION: int = 30
    N_PITCHES: int = 128

    def __post_init__(self):
        self._build_image()
        self._prepare_ticks()

    @property
    def lowest_pitch(self) -> int:
        return self.midi_piece.df.pitch.min()

    @property
    def highest_pitch(self) -> int:
        return self.midi_piece.df.pitch.max()

    def _build_image(self):
        df = self.midi_piece.df_with_end
        if not self.time_end:
            # We don't really need a full second roundup
            self.time_end = np.ceil(df.end.max())

        if self.time_end < df.end.max():
            print("Warning, piano roll is not showing everything!")

        # duration = time_end - time_start
        self.duration = self.time_end
        n_time_steps = self.RESOLUTION * int(np.ceil(self.duration))
        pianoroll = np.zeros((self.N_PITCHES, n_time_steps), np.uint8)

        # Adjust velocity color intensity to be sure it's visible
        min_value = 20
        max_value = 160

        for it, row in df.iterrows():
            note_on = row.start * self.RESOLUTION
            note_on = np.round(note_on).astype(int)

            note_end = row.end * self.RESOLUTION
            note_end = np.round(note_end).astype(int)
            pitch_idx = int(row.pitch)

            # This note is sounding right now
            if self.current_time and note_on <= self.current_time * self.RESOLUTION < note_end:
                color_value = max_value
            else:
                color_value = min_value + row.velocity
            pianoroll[pitch_idx, note_on:note_end] = color_value

        # Could be a part of "prepare empty piano roll"
        for it in range(self.N_PITCHES):
            is_black = it % 12 in [1, 3, 6, 8, 10]
            if is_black:
                pianoroll[it, :] += min_value

        self.roll = pianoroll

    def _prepare_ticks(self):
        self.y_ticks = np.arange(0, 128, 12, dtype=float)

        # Adding new line shifts the label up a little and positions
        # it nicely at the height where the note actually is
        self.pitch_labels = [f"{note_number_to_name(it)}\n" for it in self.y_ticks]

        # Move the ticks to land between the notes
        # (each note is 1-width and ticks by default are centered, ergo: 0.5 shift)
        self.y_ticks -= 0.5

        # Prepare x ticks and labels
        n_ticks = min(30, self.duration)
        step = np.ceil(self.duration / n_ticks)
        x_ticks = np.arange(0, step * n_ticks, step)
        self.x_ticks = np.round(x_ticks)
        self.x_labels = [round(xt) for xt in self.x_ticks]
