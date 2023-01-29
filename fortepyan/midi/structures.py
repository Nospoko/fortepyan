from dataclasses import field, dataclass

import pretty_midi
import pandas as pd


@dataclass
class MidiFile:
    path: str
    df: pd.DataFrame = field(init=False)
    _midi: pretty_midi.PrettyMIDI = field(init=False, repr=False)

    def __rich_repr__(self):
        yield "MidiFile"
        yield self.path
        yield "notes", self.df.shape
        yield "sustain", self.sustain.shape
        yield "minutes", round(self.duration / 60, 2)

    @property
    def duration(self) -> float:
        return self._midi.get_end_time()

    @property
    def notes(self):
        return self._midi.instruments[0].notes

    @property
    def control_changes(self):
        return self._midi.instruments[0].control_changes

    def __post_init__(self):
        self._midi = pretty_midi.PrettyMIDI(self.path)
        df = pd.DataFrame(
            {
                "pitch": [note.pitch for note in self.notes],
                "velocity": [note.velocity for note in self.notes],
                "start": [note.start for note in self.notes],
                "end": [note.end for note in self.notes],
            }
        )
        self.df = df.sort_values("start", ignore_index=True)

        # Sustain CC is 64
        sf = pd.DataFrame(
            {
                "time": [cc.time for cc in self.control_changes if cc.number == 64],
                "value": [cc.value for cc in self.control_changes if cc.number == 64],
            }
        )
        self.sustain = sf
