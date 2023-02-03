from dataclasses import field, dataclass

import pretty_midi
import pandas as pd


@dataclass
class MidiPiece:
    df: pd.DataFrame
    sustain: pd.DataFrame
    source: dict

    def __rich_repr__(self):
        yield "MidiPiece"
        yield "notes", self.df.shape
        yield "sustain", self.sustain.shape
        yield "minutes", round(self.duration / 60, 2)

    def __getitem__(self, index: slice) -> "MidiPiece":
        if not isinstance(index, slice):
            raise TypeError("You can only get a part of MidiFile that has multiple notes: Index must be a slice")

        part = self.df[index].reset_index(drop=True)

        # +0.2 to make sure we get some sustain data at the end to ring out
        ids = (self.sustain.time >= part.start.min()) & (self.sustain.time <= part.end.max() + 0.2)
        sustain_part = self.sustain[ids].reset_index(drop=True)

        first_sound = part.start.min()
        sustain_part.time -= first_sound
        part.start -= first_sound
        part.end -= first_sound

        # Make sure the piece can always be track back to the original file exactly
        out_source = self.source
        out_source["start"] = self.source.get("start", 0) + index.start
        out_source["finish"] = self.source.get("finish", 0) + index.stop
        out_source["start_time"] = self.source.get("start_time", 0) + first_sound
        out = MidiPiece(df=part, sustain=sustain_part, source=out_source)

        return out

    @property
    def duration(self) -> float:
        duration = self.df.end.max() - self.df.start.min()
        return duration

    def to_midi(self, track_name: str = "piano"):
        track = pretty_midi.PrettyMIDI()
        piano = pretty_midi.Instrument(program=0, name=track_name)

        for it, row in self.df.iterrows():
            note = pretty_midi.Note(
                velocity=int(row.velocity),
                pitch=int(row.pitch),
                start=row.start,
                end=row.end,
            )
            piano.notes.append(note)

        cc = [pretty_midi.ControlChange(64, int(r.value), r.time) for _, r in self.sustain.iterrows()]
        piano.control_changes = cc

        track.instruments.append(piano)

        return track


@dataclass
class MidiFile:
    path: str
    df: pd.DataFrame = field(init=False)
    sustain: pd.DataFrame = field(init=False)
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

    def __getitem__(self, index: slice) -> MidiPiece:
        return self.piece[index]
        if not isinstance(index, slice):
            raise TypeError("You can only get a part of MidiFile that has multiple notes: Index must be a slice")

        part = self.df[index].reset_index(drop=True)

        # +0.2 to make sure we get some sustain data at the end to ring out
        ids = (self.sustain.time >= part.start.min()) & (self.sustain.time <= part.end.max() + 0.2)
        sustain_part = self.sustain[ids].reset_index(drop=True)

        first_sound = part.start.min()
        sustain_part.time -= first_sound
        part.start -= first_sound
        part.end -= first_sound

        source = {
            "type": "MidiFile",
            "path": self.path,
        }
        out = MidiPiece(df=part, sustain=sustain_part, source=source)

        return out

    @property
    def piece(self) -> MidiPiece:
        source = {
            "type": "MidiFile",
            "path": self.path,
        }
        out = MidiPiece(
            df=self.df,
            sustain=self.sustain,
            source=source,
        )
        return out
