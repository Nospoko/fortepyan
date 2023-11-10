from warnings import showwarning
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

    def time_shift(self, shift_s: float):
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

        Note:
            - This method modifies the MidiPiece object in place and does not return a new object.
        """
        self.df.start += shift_s
        self.df.end += shift_s

    def trim(self, start: float, finish: float, shift_time: bool = True, slice_type: str = "standard") -> "MidiPiece":
        """
        Trim a segment of a MIDI piece based on specified start and finish parameters, with options for different slicing types.

        This method modifies the MIDI piece by selecting a segment from it, based on the `start` and `finish` parameters.
        The segment can be selected through different methods determined by `slice_type`. If `shift_time` is True,
        the timing of notes in the trimmed segment will be shifted to start from zero.

        Args:
            start (float | int): The starting point of the segment. It's treated as a float for 'standard' or 'by_end' slicing types,
                                and as an integer for 'index' slicing type.
            finish (float | int): The ending point of the segment. Similar to `start`, it's treated as a float or an integer
                                depending on the `slice_type`.
            shift_time (bool, optional): Whether to shift note timings in the trimmed segment to start from zero. Default is True.
            slice_type (str, optional): The method of slicing. Can be 'standard', 'by_end', or 'index'. Default is 'standard'.

        Returns:
            MidiPiece: A new `MidiPiece` object representing the trimmed segment of the original MIDI piece.

        Raises:
            ValueError: If `start` and `finish` are not integers when `slice_type` is 'index', or if `start` is larger than `finish`.
            IndexError: If the indices are out of bounds for 'index' slicing type, or if no notes are found in the specified range for other types.
            NotImplementedError: If the `slice_type` provided is not implemented.

        Examples:
            Trimming using standard slicing:
            >>> midi_piece.trim(start=1.0, finish=5.0)

            Trimming using index slicing:
            >>> midi_piece.trim(start=0, finish=10, slice_type="index")

            Trimming with time shift disabled:
            >>> midi_piece.trim(start=1.0, finish=5.0, shift_time=False)

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

    def to_midi(self, instrument_name: str = "Acoustic Grand Piano") -> pretty_midi.PrettyMIDI:
        """
        Converts the note data stored in this object into a MIDI track using the specified instrument.

        This function creates a MIDI track with notes defined by the object's data. It uses the pretty_midi library
        to construct the track and the notes within it. The instrument used for the MIDI track can be specified,
        and defaults to "Acoustic Grand Piano" if not provided.

        Args:
            instrument_name (str, optional):
                The name of the instrument to be used for the MIDI track. This should be a valid instrument name
                that can be interpreted by the pretty_midi library. Defaults to "Acoustic Grand Piano". See the note below for more information.

        Returns:
            pretty_midi.PrettyMIDI:
                A PrettyMIDI object representing the MIDI track created from the note data. This object can be
                further manipulated or directly written to a MIDI file.

        Examples:
            >>> track = my_object.to_midi("Violin")
            This would create a MIDI track using the notes in 'my_object' with a Violin instrument.

        Note:
            - See [this wikipedia article](https://en.wikipedia.org/wiki/General_MIDI#Parameter_interpretations) for instrument names
        """
        track = pretty_midi.PrettyMIDI()
        program = pretty_midi.instrument_name_to_program(instrument_name)
        instrument = pretty_midi.Instrument(program=program, name=instrument_name)

        # Convert the DataFrame to a list of tuples to avoid pandas overhead in the loop
        note_data = self.df[["velocity", "pitch", "start", "end"]].to_records(index=False)

        # Now we can iterate through this array which is more efficient than DataFrame iterrows
        for velocity, pitch, start, end in note_data:
            note = pretty_midi.Note(velocity=int(velocity), pitch=int(pitch), start=start, end=end)
            instrument.notes.append(note)

        track.instruments.append(instrument)

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
