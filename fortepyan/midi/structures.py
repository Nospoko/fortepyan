from dataclasses import field, dataclass

import numpy as np
import pretty_midi
import pandas as pd

from fortepyan.midi import tools as midi_tools


@dataclass
class MidiPiece:
    df: pd.DataFrame
    source: dict = None

    def __rich_repr__(self):
        yield "MidiPiece"
        yield "notes", self.df.shape
        yield "minutes", round(self.duration / 60, 2)

    def __post_init__(self):
        if not self.source:
            self.source = {
                "start": 0,
                "start_time": 0,
                "finish": self.size,
            }

    @property
    def size(self) -> int:
        return self.df.shape[0]

    def time_shift(self, shift_s: float):
        self.df.start += shift_s
        self.df.end += shift_s

    def trim(self, start: float, finish: float) -> "MidiPiece":
        """Trim the MidiPiece object between the specified start and finish time.

        This function takes two parameters, `start` and `finish`, which represent the start and end time in seconds,
        and returns a new MidiPiece object that contains only the notes within the specified time range.

        Args:
        - start (float): start time in seconds
        - finish (float): end time in seconds

        Returns:
        - MidiPiece: the trimmed MidiPiece object
        """
        # Filter the rows in the data frame that are within the specified start and end time
        ids = (self.df.start >= start) & (self.df.start <= finish)
        # Get the indices of the rows that meet the criteria
        idxs = np.where(ids)[0]

        # Get the start and end indices for the new MidiPiece object
        start = idxs[0]
        finish = idxs[-1] + 1
        # Slice the original MidiPiece object to create the trimmed MidiPiece object
        out = self[start:finish]
        # Return the trimmed MidiPiece object
        return out

    def __sanitize_get_index(self, index: slice) -> slice:
        if not isinstance(index, slice):
            raise TypeError("You can only get a part of MidiFile that has multiple notes: Index must be a slice")

        # If you wan piece[:stop]
        if not index.start:
            index = slice(0, index.stop)

        # If you want piece[start:]
        if not index.stop:
            index = slice(index.start, self.size)

        return index

    def __getitem__(self, index: slice) -> "MidiPiece":
        index = self.__sanitize_get_index(index)
        part = self.df[index].reset_index(drop=True)

        first_sound = part.start.min()
        part.start -= first_sound
        part.end -= first_sound

        # Make sure the piece can always be track back to the original file exactly
        out_source = dict(self.source)
        out_source["start"] = self.source.get("start", 0) + index.start
        out_source["finish"] = self.source.get("start", 0) + index.stop
        out_source["start_time"] = self.source.get("start_time", 0) + first_sound
        out = MidiPiece(df=part, source=out_source)

        return out

    @property
    def duration(self) -> float:
        duration = self.df.end.max() - self.df.start.min()
        return duration

    @property
    def end(self) -> float:
        return self.df_with_end.end.max()

    @property
    def df_with_end(self) -> pd.DataFrame:
        df = self.df.copy()
        df["end"] = df.start + df.duration
        return df

    def to_midi(self, track_name: str = "piano"):
        track = pretty_midi.PrettyMIDI()
        piano = pretty_midi.Instrument(program=0, name=track_name)

        for it, row in self.df_with_end.iterrows():
            note = pretty_midi.Note(
                velocity=int(row.velocity),
                pitch=int(row.pitch),
                start=row.start,
                end=row.end,
            )
            piano.notes.append(note)

        # cc = [pretty_midi.ControlChange(64, int(r.value), r.time) for _, r in self.sustain.iterrows()]
        # piano.control_changes = cc

        track.instruments.append(piano)

        return track

    @classmethod
    def from_huggingface(cls, record: dict) -> "MidiPiece":
        df = pd.DataFrame(record["notes"])
        df["duration"] = df.end - df.start

        source = {
            "composer": record.get("composer"),
            "title": record.get("title"),
            "midi_filename": record.get("midi_filename"),
            "record_id": record.get("record_id"),
            "user": record.get("user"),
        }
        that = cls(df=df, source=source)
        return that


@dataclass
class MidiFile:
    path: str
    apply_sustain: bool = True
    sustain_threshold: int = 62
    df: pd.DataFrame = field(init=False)
    raw_df: pd.DataFrame = field(init=False)
    sustain: pd.DataFrame = field(init=False)
    control_frame: pd.DataFrame = field(init=False, repr=False)
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
        # This is not great/foolproof, but we already have files
        # where the piano track is present on multiple "programs"/"instruments
        notes = sum([inst.notes for inst in self._midi.instruments], [])
        return notes

    @property
    def control_changes(self):
        # See the note for notes ^^
        ccs = sum([inst.control_changes for inst in self._midi.instruments], [])
        return ccs

    def __post_init__(self):
        # Read the MIDI object
        self._midi = pretty_midi.PrettyMIDI(self.path)

        # Extract CC data
        self.control_frame = pd.DataFrame(
            {
                "time": [cc.time for cc in self.control_changes],
                "value": [cc.value for cc in self.control_changes],
                "number": [cc.number for cc in self.control_changes],
            }
        )

        # Sustain CC is 64
        ids = self.control_frame.number == 64
        self.sustain = self.control_frame[ids].reset_index(drop=True)

        # Extract notes
        raw_df = pd.DataFrame(
            {
                "pitch": [note.pitch for note in self.notes],
                "velocity": [note.velocity for note in self.notes],
                "start": [note.start for note in self.notes],
                "end": [note.end for note in self.notes],
            }
        )
        self.raw_df = raw_df.sort_values("start", ignore_index=True)

        if self.apply_sustain:
            self.df = midi_tools.apply_sustain(
                df=self.raw_df,
                sustain=self.sustain,
                sustain_threshold=self.sustain_threshold,
            )
        else:
            self.df = self.raw_df

        self.df["duration"] = self.df.end - self.df.start

    def __getitem__(self, index: slice) -> MidiPiece:
        return self.piece[index]
        if not isinstance(index, slice):
            raise TypeError("You can only get a part of MidiFile that has multiple notes: Index must be a slice")

        part = self.df[index].reset_index(drop=True)
        first_sound = part.start.min()

        # TODO: When you start working with pedal data, add this to the Piece structure
        if not self.apply_sustain:
            # +0.2 to make sure we get some sustain data at the end to ring out
            ids = (self.sustain.time >= part.start.min()) & (self.sustain.time <= part.end.max() + 0.2)
            sustain_part = self.sustain[ids].reset_index(drop=True)
            sustain_part.time -= first_sound

        # Move the notes
        part.start -= first_sound
        part.end -= first_sound

        source = {
            "type": "MidiFile",
            "path": self.path,
        }
        out = MidiPiece(df=part, source=source)

        return out

    @property
    def piece(self) -> MidiPiece:
        source = {
            "type": "MidiFile",
            "path": self.path,
        }
        out = MidiPiece(
            df=self.df,
            source=source,
        )
        return out
