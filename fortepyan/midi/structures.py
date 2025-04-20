import json
from typing import IO, Optional
from dataclasses import field, dataclass

import numpy as np
import pretty_midi
import pandas as pd

from fortepyan.midi import tools as midi_tools


@dataclass
class MidiPiece:
    """
    A data class representing a piece of MIDI music, encapsulated in a Pandas DataFrame.

    This class provides functionalities for managing MIDI data, including methods to manipulate and represent the MIDI piece.
    The data is primarily stored in a Pandas DataFrame (`df`) which contains columns like 'start', 'end', 'duration',
    'pitch', and 'velocity', essential for MIDI data representation. The class also includes source information for
    additional context.

    Attributes:
        df (pd.DataFrame): The DataFrame containing the MIDI data.
        source (dict, optional): Additional information about the MIDI piece's source. Defaults to None.

    Examples:
        Creating a MidiPiece instance:
        >>> midi_piece = MidiPiece(df=my_midi_df)

    """

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

    def time_shift(self, shift_s: float) -> "MidiPiece":
        """
        Shift the start and end times of all notes in the MidiPiece by a specified amount.

        This method adjusts the start and end times of every note in the MidiPiece by adding the specified shift amount
        `shift_s`. This can be used to move the entire piece forward or backward in time.

        Args:
            shift_s (float): The amount of time (in seconds) to shift the start and end times of the notes.
                        Positive values shift the piece forward, and negative values shift it backward.

        Examples:
            Shifting the entire piece forward by 2 seconds:
            >>> midi_piece.time_shift(2.0)

            Shifting the entire piece backward by 0.5 seconds:
            >>> midi_piece.time_shift(-0.5)

        Returns:
            - A new MidiPiece object after shifting
        """
        source = dict(self.source)
        source["time_shift"] = source.get("time_shift", 0) + shift_s

        new_piece = MidiPiece(
            df=self.df.copy(),
            source=source,
        )

        new_piece.df.start += shift_s
        new_piece.df.end += shift_s
        return new_piece

    def trim(
        self,
        start: float,
        finish: float,
        shift_time: bool = True,
        slice_type: str = "standard",
    ) -> "MidiPiece":
        """
        Trim a segment of a MIDI piece based on specified start and finish parameters,
        with options for different slicing types.

        This method modifies the MIDI piece by selecting a segment from it, based on the `start` and `finish` parameters.
        The segment can be selected through different methods determined by `slice_type`. If `shift_time` is True,
        the timing of notes in the trimmed segment will be shifted to start from zero.

        Args:
            start (float | int): The starting point of the segment.
                It's treated as a float for 'standard' or 'by_end' slicing types, and as an integer
                for 'index' slicing type.
            finish (float | int): The ending point of the segment. Similar to `start`, it's treated
                as a float or an integer depending on the `slice_type`.
            shift_time (bool, optional): Whether to shift note timings in the trimmed segment
                to start from zero. Default is True.
            slice_type (str, optional): The method of slicing. Can be 'standard',
                'by_end', or 'index'. Default is 'standard'. See note below.

        Returns:
            MidiPiece: A new `MidiPiece` object representing the trimmed segment of the original MIDI piece.

        Raises:
            ValueError: If `start` and `finish` are not integers when
                `slice_type` is 'index', or if `start` is larger than `finish`.
            IndexError: If the indices are out of bounds for 'index' slicing type,
                or if no notes are found in the specified range for other types.
            NotImplementedError: If the `slice_type` provided is not implemented.

        Examples:
            Trimming using standard slicing:
            >>> midi_piece.trim(start=1.0, finish=5.0)

            Trimming using index slicing:
            >>> midi_piece.trim(start=0, finish=10, slice_type="index")

            Trimming with time shift disabled:
            >>> midi_piece.trim(start=1.0, finish=5.0, shift_time=False)

            An example of a trimmed MIDI piece:
            ![Trimmed MIDI piece](../assets/random_midi_piece.png)

        Slice types:
            The `slice_type` parameter determines how the start and finish parameters are interpreted.
            It can be one of the following:

                'standard': Trims notes that start outside the [start, finish] range.

                'by_end': Trims notes that end after the finish parameter.

                'index': Trims notes based on their index in the DataFrame.
                    The start and finish parameters are treated as integers

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
        """
        Sanitize and adjust the provided slice index for the MIDI file object.

        This private method ensures that the slice provided is valid for slicing a MIDI file object. It adjusts the slice
        to handle scenarios where only one bound (start or stop) is provided. If no start is specified, it defaults to 0.
        If no stop is specified, it defaults to the size of the MIDI file.

        Parameters:
            index (slice): The slice object to be sanitized and adjusted. It must be a slice object.

        Returns:
            slice: The sanitized and possibly adjusted slice object.

        Raises:
            TypeError: If the provided index is not a slice object.

        Examples:
            - Getting a part of the MIDI file from the beginning up to a stop point:
                >>> midi_file.__sanitize_get_index(slice(None, 10))

            - Getting a part of the MIDI file from a start point to the end:
                >>> midi_file.__sanitize_get_index(slice(5, None))

        Note:
        - This method is intended for internal use within the class and should not be called directly from outside the class.
        """
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
        """
        Get a slice of the MIDI piece, optionally shifting the time of notes.

        This method returns a segment of the MIDI piece based on the provided index. It sanitizes the index using the
        `__sanitize_get_index` method. If `shift_time` is True, it shifts the start and end times of the notes in the
        segment so that the first note starts at time 0. The method also keeps track of the original piece's information
        in the sliced piece's source data.

        Args:
            index (slice): The slicing index to select a part of the MIDI piece. It must be a slice object.
            shift_time (bool, optional): If True, shifts the start and end times of notes so the first note starts at 0.
                                    Default is True.

        Returns:
            MidiPiece: A new `MidiPiece` object representing the sliced segment of the original MIDI piece.

        Raises:
            TypeError: If the provided index is not a slice object (handled in `__sanitize_get_index`).

        Examples:
            Getting a slice from the MIDI file with time shift:
                >>> midi_piece[0:10]

            Getting a slice without time shift:
                >>> midi_piece[5:15, shift_time=False]

        Note:
            The `__getitem__` method is a special method in Python used for indexing or slicing objects. In this class,
        it is used to get a slice of a MIDI piece.
        """
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
        """
        Combine this MidiPiece with another MidiPiece, adjusting the time stamps.

        This method overloads the `+` operator for MidiPiece objects. It concatenates the DataFrame of the current object
        with that of another MidiPiece, adjusting the start and end times of notes in the second piece so that they follow
        sequentially after the first piece. A warning is raised to inform the user that the resulting piece may not be
        musically valid.

        Parameters:
            other (MidiPiece): Another MidiPiece object to add to the current one.

        Returns:
            MidiPiece: A new MidiPiece object that represents the combination of the two MidiPieces.

        Raises:
            TypeError: If the object being added is not an instance of MidiPiece.

        Examples:
            Adding two MidiPiece objects:
            >>> combined_piece = midi_piece1 + midi_piece2

        Note:
            - The method ensures that the original MidiPiece objects are not modified during the addition.
            - A UserWarning is raised to indicate that the resulting piece might not be musically valid.
        """
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

        # TODO Think of another way to track this information
        # maybe add {"warnings": ["merged from multiple pieces"]} to .source?
        # Show warning as the piece might not be musically valid.
        # showwarning(
        #     message="The resulting piece may not be musically valid.",
        #     category=UserWarning,
        #     filename="fortepyan/midi/structures.py",
        #     lineno=280,
        # )

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

    def to_midi(self, instrument_name: str = "Piano") -> "MidiFile":
        """
        Converts the note data stored in this object into a MIDI track using the specified instrument.

        This function creates a MIDI track with notes defined by the object's data.
        It uses the MidiFile to construct the track and the notes within it.

        Args:
            instrument_name (str, optional):
                The name of the track's instrument. Defaults to "Piano".

        Returns:
            MidiFile:
                A MidiFile object representing the MIDI track created from the note data. This object can be
                further manipulated or directly written to a MIDI file.

        Examples:
            >>> track = my_object.to_midi("Violin")
            This would create a MIDI track using the notes in 'my_object' and name it "Violin".

        """
        return MidiFile.from_piece(self)

    @classmethod
    def from_huggingface(cls, record: dict) -> "MidiPiece":
        df = pd.DataFrame(record["notes"])
        df["duration"] = df.end - df.start

        source = json.loads(record["source"])
        that = cls(df=df, source=source)
        return that

    @classmethod
    def from_file(cls, path: str) -> "MidiPiece":
        piece = MidiFile(str(path)).piece
        return piece


@dataclass
class MidiFile:
    path: Optional[str] = None
    apply_sustain: bool = True
    sustain_threshold: int = 62
    df: pd.DataFrame = field(init=False)
    raw_df: pd.DataFrame = field(init=False)
    sustain: pd.DataFrame = field(init=False)
    control_frame: pd.DataFrame = field(init=False, repr=False)
    _midi: pretty_midi.PrettyMIDI = field(init=True, repr=False, default=None)

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

    def _load_midi_file(self):
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

    def __post_init__(self):
        if self.path:
            # Read the MIDI object
            self._midi = pretty_midi.PrettyMIDI(self.path)

        # Otherwise _midi had to be provided as an argument
        self._load_midi_file()

    def __getitem__(self, index: slice) -> MidiPiece:
        return self.piece[index]

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

    @classmethod
    def from_file(cls, midi_file: IO) -> "MidiFile":
        """
        Generic wrapper for the pretty_midi.PrettyMIDI interface.

        Args:
            midi_file (str or file): Path or file pointer to a MIDI file.

        Returns:
            MidiFile: A new `MidiFile` object containing the input file.
        """
        _midi = pretty_midi.PrettyMIDI(midi_file)

        midi_file = cls(_midi=_midi)
        return midi_file

    @classmethod
    def from_piece(cls, piece: MidiPiece) -> "MidiFile":
        _midi = pretty_midi.PrettyMIDI()

        # 0 is piano
        program = 0
        instrument_name = "fortepyan"
        instrument = pretty_midi.Instrument(program=program, name=instrument_name)

        # Convert the DataFrame to a list of tuples to avoid pandas overhead in the loop
        note_data = piece.df[["velocity", "pitch", "start", "end"]].to_records(index=False)
        # Now we can iterate through this array which is more efficient than DataFrame iterrows
        for velocity, pitch, start, end in note_data:
            note = pretty_midi.Note(
                velocity=int(velocity),
                pitch=int(pitch),
                start=start,
                end=end,
            )
            instrument.notes.append(note)

        _midi.instruments.append(instrument)

        midi_file = cls(_midi=_midi)

        return midi_file

    @classmethod
    def merge_files(cls, midi_files: list["MidiFile"], space: float = 0.0) -> "MidiFile":
        """
        Merges multiple MIDI files into a single MIDI file.

        This method combines the notes and control changes from the input list of
        `MidiFile` objects into a single MIDI track with an optional space between
        each file's content. All input files are assumed to have a piano track
        (`program=0`) as the first instrument.

        Args:
            midi_files (list[MidiFile]): List of `MidiFile` objects to be merged.
            space (float, optional): Time (in seconds) to insert between the end of
                one MIDI file and the start of the next. Defaults to 0.0.

        Returns:
            MidiFile: A new `MidiFile` object containing the merged tracks.

        Note:
            - Only the first instrument (assumed to be a piano track) from each file
              is processed.
            - The last control change time is considered to calculate the start offset
              for the next file. If there are no control changes, the last note end
              time is used.
        """

        _midi = pretty_midi.PrettyMIDI()

        # 0 is piano
        program = 0
        instrument_name = "fortepyan"
        instrument = pretty_midi.Instrument(program=program, name=instrument_name)

        start_offset = 0
        notes = []
        control_changes = []
        for midi_file in midi_files:
            piano_track = midi_file._midi.instruments[0]
            for note in piano_track.notes:
                new_note = pretty_midi.Note(
                    start=note.start + start_offset,
                    end=note.end + start_offset,
                    pitch=note.pitch,
                    velocity=note.velocity,
                )
                notes.append(new_note)

            for cc in piano_track.control_changes:
                new_cc = pretty_midi.ControlChange(
                    number=cc.number,
                    value=cc.value,
                    time=cc.time + start_offset,
                )
                control_changes.append(new_cc)

            # Events from the next file have to be shifted to start later
            last_cc_time = control_changes[-1].time if control_changes else 0
            start_offset = max(notes[-1].end, last_cc_time) + space

        instrument.notes = notes
        instrument.control_changes = control_changes
        _midi.instruments.append(instrument)

        midi_file = cls(_midi=_midi)

        return midi_file

    def write(self, filename):
        self._midi.write(filename)


def __repr__(self):
    return f"MidiFile({self.path})"


def __str__(self):
    return f"MidiFile({self.path})"
