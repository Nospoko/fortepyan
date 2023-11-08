from warnings import showwarning
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
        # Ensure at least two of the three timing columns are present
        timing_columns = {"start", "end", "duration"}
        if sum(col in self.df.columns for col in timing_columns) < 2:
            raise ValueError("The DataFrame must have at least two of the following columns: 'start', 'end', 'duration'.")

        # Calculate the missing timing column if necessary
        if "start" not in self.df.columns:
            self.df["start"] = self.df["end"] - self.df["duration"]
        elif "end" not in self.df.columns:
            self.df["end"] = self.df["start"] + self.df["duration"]
        elif "duration" not in self.df.columns:
            self.df["duration"] = self.df["end"] - self.df["start"]

        # Convert timing columns to float to ensure consistency
        for col in timing_columns:
            self.df[col] = self.df[col].astype(float)

        # Check for the absolutely required columns: 'pitch' and 'velocity'
        if "pitch" not in self.df.columns:
            raise ValueError("The DataFrame is missing the required column: 'pitch'.")
        if "velocity" not in self.df.columns:
            raise ValueError("The DataFrame is missing the required column: 'velocity'.")

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

    def trim(self, start: float, finish: float, shift_time: bool = True, slice_type: str = "standard") -> "MidiPiece":
        """
        Trim the MidiPiece object based on a specified slicing type.

        Args:
        - start (float): Depending on `slice_type`, this is either the start time in seconds or the start index.
        - finish (float): Depending on `slice_type`, this is either the end time in seconds or the end index.
        - shift_time (bool, optional): If True, the trimmed piece's start time will be shifted to 0. Defaults to True.
        - slice_type (str, optional): Determines the slicing method ('standard', 'by_end', 'index'). Defaults to "standard".
            - "standard": Slices the MidiPiece to include notes that start within the [start, finish] time range.
            - "by_end": Slices the MidiPiece to include notes where the end time is within the [start, finish] time range.
            - "index": Slices the MidiPiece by note indices, where start and finish must be integer indices.

        Returns:
        - MidiPiece: The trimmed MidiPiece object.
        """
        if slice_type == "index":
            if not isinstance(start, int) or not isinstance(finish, int):
                raise ValueError("Using 'index' slice_type requires 'start' and 'finish' to be integers.")
            if start < 0 or finish >= self.size:
                raise IndexError("Index out of bounds.")
            if start > finish:
                raise ValueError("'start' must be smaller than 'finish'.")
            start_idx = start
            finish_idx = finish + 1
        else:
            if slice_type == "by_end":
                ids = (self.df.start >= start) & (self.df.end <= finish)
            elif slice_type == "standard":  # Standard slice type
                ids = (self.df.start >= start) & (self.df.start <= finish)
            else:
                # not implemented
                raise NotImplementedError(f"Slice type '{slice_type}' is not implemented.")
            idx = np.where(ids)[0]
            if len(idx) == 0:
                raise IndexError("No notes found in the specified range.")
            start_idx = idx[0]
            finish_idx = idx[-1] + 1

        slice_obj = slice(start_idx, finish_idx)

        out = self.__getitem__(slice_obj, shift_time)

        return out

    def __sanitize_get_index(self, index: slice) -> slice:
        if not isinstance(index, slice):
            raise TypeError("You can only get a part of MidiFile that has multiple notes: Index must be a slice")

        # If you want piece[:stop]
        if not index.start:
            index = slice(0, index.stop)

        # If you want piece[start:]
        if not index.stop:
            index = slice(index.start, self.size)

        return index

    def __getitem__(self, index: slice, shift_time: bool = True) -> "MidiPiece":
        index = self.__sanitize_get_index(index)
        part = self.df[index].reset_index(drop=True)

        if shift_time:
            # Shift the start and end times so that the first note starts at 0
            first_sound = part.start.min()
            part.start -= first_sound
            part.end -= first_sound
            # Adjust the source to reflect the new start time
            start_time_adjustment = first_sound
        else:
            # No adjustment to the start time
            start_time_adjustment = 0

        # Make sure the piece can always be tracked back to the original file exactly
        out_source = dict(self.source)
        out_source["start"] = self.source.get("start", 0) + index.start
        out_source["finish"] = self.source.get("start", 0) + index.stop
        out_source["start_time"] = self.source.get("start_time", 0) + start_time_adjustment
        out = MidiPiece(df=part, source=out_source)

        return out

    def __add__(self, other: "MidiPiece") -> "MidiPiece":
        if not isinstance(other, MidiPiece):
            raise TypeError("You can only add MidiPiece objects to other MidiPiece objects.")

        # Adjust the start/end times of the second piece
        other.df.start += self.end
        other.df.end += self.end

        # Concatenate the two pieces
        df = pd.concat([self.df, other.df], ignore_index=True)

        # make sure the other piece is not modified
        other.df.start -= self.end
        other.df.end -= self.end

        # make sure that start and end times are floats
        df.start = df.start.astype(float)
        df.end = df.end.astype(float)

        out = MidiPiece(df=df)

        # Show warning as the piece might not be musically valid.
        showwarning("The resulting piece may not be musically valid.", UserWarning, "fortepyan", lineno=1)

        return out

    def __len__(self) -> int:
        return self.size

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

        # Convert the DataFrame to a list of tuples to avoid pandas overhead in the loop
        note_data = self.df[["velocity", "pitch", "start", "end"]].to_records(index=False)

        # Now we can iterate through this array which is more efficient than DataFrame iterrows
        for velocity, pitch, start, end in note_data:
            note = pretty_midi.Note(velocity=int(velocity), pitch=int(pitch), start=start, end=end)
            piano.notes.append(note)

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

    @classmethod
    def from_file(cls, path: str) -> "MidiPiece":
        piece = MidiFile(str(path)).piece
        return piece


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
