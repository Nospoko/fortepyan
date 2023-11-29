import math
import functools
import collections
from heapq import merge
from warnings import showwarning
from dataclasses import field, dataclass

import mido
import numpy as np
import pandas as pd

from fortepyan.midi import tools as midi_tools
from fortepyan.midi import containers as midi_containers
from fortepyan.midi.containers import key_name_to_key_number

# The largest we'd ever expect a tick to be
MAX_TICK = 1e10


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
            slice_type (str, optional): The method of slicing. Can be 'standard', 'by_end', or 'index'. Default is 'standard'. See note below.

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

            An example of a trimmed MIDI piece:
            ![Trimmed MIDI piece](../assets/random_midi_piece.png)

        Slice types:
            The `slice_type` parameter determines how the start and finish parameters are interpreted. It can be one of the following:

                'standard': Trims notes that start outside the [start, finish] range.

                'by_end': Trims notes that end after the finish parameter.

                'index': Trims notes based on their index in the DataFrame. The start and finish parameters are treated as integers

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

    def to_midi(self, instrument_name: str = "Piano") -> "MidiFile":
        """
        Converts the note data stored in this object into a MIDI track using the specified instrument.

        This function creates a MIDI track with notes defined by the object's data. It uses the MidiFile to construct the track and the notes within it.

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
        track = MidiFile()
        program = 0  # 0 is piano
        instrument = midi_containers.Instrument(program=program, name=instrument_name)

        # Convert the DataFrame to a list of tuples to avoid pandas overhead in the loop
        note_data = self.df[["velocity", "pitch", "start", "end"]].to_records(index=False)
        # Now we can iterate through this array which is more efficient than DataFrame iterrows
        for velocity, pitch, start, end in note_data:
            note = midi_containers.Note(velocity=int(velocity), pitch=int(pitch), start=start, end=end)
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
    path: str = None
    apply_sustain: bool = True
    sustain_threshold: int = 62
    resolution: int = 220
    initial_tempo: float = 120.0
    df: pd.DataFrame = field(init=False)
    raw_df: pd.DataFrame = field(init=False)
    sustain: pd.DataFrame = field(init=False)
    control_frame: pd.DataFrame = field(init=False, repr=False)
    instruments: list = field(init=False, repr=False)
    key_signature_changes: list = field(init=False, repr=False)
    time_signature_changes: list = field(init=False, repr=False)
    lyrics: list = field(init=False, repr=False)
    text_events: list = field(init=False, repr=False)
    __tick_to_time: np.ndarray = field(init=False, repr=False)

    def __rich_repr__(self):
        yield "MidiFile"
        yield self.path
        yield "notes", self.df.shape
        yield "sustain", self.sustain.shape
        yield "minutes", round(self.duration / 60, 2)

    @property
    def duration(self) -> float:
        return self.get_end_time()

    @property
    def notes(self):
        # This is not great/foolproof, but we already have files
        # where the piano track is present on multiple "programs"/"instruments
        notes = sum([inst.notes for inst in self.instruments], [])
        return notes

    @property
    def control_changes(self):
        # See the note for notes ^^
        ccs = sum([inst.control_changes for inst in self.instruments], [])
        return ccs

    def __post_init__(self):
        self._initialize_fields()
        if self.path:
            self._process_midi_file()
        else:
            self._setup_without_path()

    def _initialize_fields(self):
        self.instruments = []
        self.key_signature_changes = []
        self.time_signature_changes = []
        self.lyrics = []
        self.text_events = []
        self.control_frame = pd.DataFrame()
        self.sustain = pd.DataFrame()
        self.df = pd.DataFrame()
        self.raw_df = pd.DataFrame()

    def _setup_without_path(self):
        self._tick_scales = [(0, 60.0 / (self.initial_tempo * self.resolution))]
        self.__tick_to_time = [0]

    def _process_midi_file(self):
        midi_data = mido.MidiFile(filename=self.path)

        # Convert to absolute ticks
        for track in midi_data.tracks:
            tick = 0
            for event in track:
                event.time += tick
                tick = event.time

        # Store the resolution for later use
        self.resolution = midi_data.ticks_per_beat

        # Populate the list of tempo changes (tick scales)
        self._load_tempo_changes(midi_data)

        # Update the array which maps ticks to time
        max_tick = self.get_max_tick(midi_data)
        # If max_tick is too big, the MIDI file is probably corrupt
        # and creating the __tick_to_time array will thrash memory
        if max_tick > MAX_TICK:
            raise ValueError(("MIDI file has a largest tick of {}," " it is likely corrupt".format(max_tick)))

        # Create list that maps ticks to time in seconds
        self._update_tick_to_time(self.get_max_tick(midi_data))

        # Load the metadata
        self._load_metadata(midi_data)

        # Check that there are tempo, key and time change events
        # only on track 0
        if any(e.type in ("set_tempo", "key_signature", "time_signature") for track in midi_data.tracks[1:] for e in track):
            showwarning(
                "Tempo, Key or Time signature change events found on "
                "non-zero tracks.  This is not a valid type 0 or type 1 "
                "MIDI file.  Tempo, Key or Time Signature may be wrong.",
                RuntimeWarning,
                "fortepyan",
                lineno=1,
            )

        # Populate the list of instruments
        self._load_instruments(midi_data)

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

    def get_max_tick(self, midi_data):
        return max([max([e.time for e in t]) for t in midi_data.tracks]) + 1

    def __getitem__(self, index: slice) -> MidiPiece:
        return self.piece[index]
        # if not isinstance(index, slice):
        #     raise TypeError("You can only get a part of MidiFile that has multiple notes: Index must be a slice")

        # part = self.df[index].reset_index(drop=True)
        # first_sound = part.start.min()

        # # TODO: When you start working with pedal data, add this to the Piece structure
        # if not self.apply_sustain:
        #     # +0.2 to make sure we get some sustain data at the end to ring out
        #     ids = (self.sustain.time >= part.start.min()) & (self.sustain.time <= part.end.max() + 0.2)
        #     sustain_part = self.sustain[ids].reset_index(drop=True)
        #     sustain_part.time -= first_sound

        # # Move the notes
        # part.start -= first_sound
        # part.end -= first_sound

        # source = {
        #     "type": "MidiFile",
        #     "path": self.path,
        # }
        # out = MidiPiece(df=part, source=source)

        # return out

    def _load_tempo_changes(self, midi_data):
        """
        Populates `self._tick_scales` with tuples of
        `(tick, tick_scale)` loaded from `midi_data`.

        Parameters:
        midi_data (midi.FileReader): MIDI object from which data will be read.
        """

        # MIDI data is given in "ticks".
        # We need to convert this to clock seconds.
        # The conversion factor involves the BPM, which may change over time.
        # So, create a list of tuples, (time, tempo)
        # denoting a tempo change at a certain time.
        # By default, set the tempo to 120 bpm, starting at time 0
        self._tick_scales = [(0, 60.0 / (120.0 * self.resolution))]
        # For SMF file type 0, all events are on track 0.
        # For type 1, all tempo events should be on track 1.
        # Everyone ignores type 2. >>> :'(
        # So, just look at events on track 0
        for event in midi_data.tracks[0]:
            if event.type == "set_tempo":
                # Only allow one tempo change event at the beginning
                if event.time == 0:
                    bpm = 6e7 / event.tempo
                    self._tick_scales = [(0, 60.0 / (bpm * self.resolution))]
                else:
                    # Get time and BPM up to this point
                    _, last_tick_scale = self._tick_scales[-1]
                    tick_scale = 60.0 / ((6e7 / event.tempo) * self.resolution)
                    # Ignore repetition of BPM, which happens often
                    if tick_scale != last_tick_scale:
                        self._tick_scales.append((event.time, tick_scale))

    def _load_metadata(self, midi_data):
        """Populates ``self.time_signature_changes`` with ``TimeSignature``
        objects, ``self.key_signature_changes`` with ``KeySignature`` objects,
        ``self.lyrics`` with ``Lyric`` objects and ``self.text_events`` with
        ``Text`` objects.

        Parameters
        ----------
        midi_data : midi.FileReader
            MIDI object from which data will be read.
        """

        # Initialize empty lists for storing key signature changes, time
        # signature changes, and lyrics
        self.key_signature_changes = []
        self.time_signature_changes = []
        self.lyrics = []
        self.text_events = []

        for event in midi_data.tracks[0]:
            if event.type == "key_signature":
                key_obj = midi_containers.KeySignature(key_name_to_key_number(event.key), self.__tick_to_time[event.time])
                self.key_signature_changes.append(key_obj)

            elif event.type == "time_signature":
                ts_obj = midi_containers.TimeSignature(event.numerator, event.denominator, self.__tick_to_time[event.time])
                self.time_signature_changes.append(ts_obj)

        # We search for lyrics and text events on all tracks
        # Lists of lyrics and text events lists, for every track
        tracks_with_lyrics = []
        tracks_with_text_events = []
        for track in midi_data.tracks:
            # Track specific lists that get appended if not empty
            lyrics = []
            text_events = []
            for event in track:
                if event.type == "lyrics":
                    lyrics.append(midi_containers.Lyric(event.text, self.__tick_to_time[event.time]))
                elif event.type == "text":
                    text_events.append(midi_containers.Text(event.text, self.__tick_to_time[event.time]))

            if lyrics:
                tracks_with_lyrics.append(lyrics)
            if text_events:
                tracks_with_text_events.append(text_events)

        # We merge the already sorted lists for every track, based on time
        self.lyrics = list(merge(*tracks_with_lyrics, key=lambda x: x.time))
        self.text_events = list(merge(*tracks_with_text_events, key=lambda x: x.time))

    def _update_tick_to_time(self, max_tick):
        """
        Creates ``self.__tick_to_time``, a class member array which maps
        ticks to time starting from tick 0 and ending at ``max_tick``.
        """
        # If max_tick is smaller than the largest tick in self._tick_scales,
        # use this largest tick instead
        max_scale_tick = max(ts[0] for ts in self._tick_scales)
        max_tick = max_tick if max_tick > max_scale_tick else max_scale_tick
        # Allocate tick to time array - indexed by tick from 0 to max_tick
        self.__tick_to_time = np.zeros(max_tick + 1)
        # Keep track of the end time of the last tick in the previous interval
        last_end_time = 0
        # Cycle through intervals of different tempi
        for (start_tick, tick_scale), (end_tick, _) in zip(self._tick_scales[:-1], self._tick_scales[1:]):
            # Convert ticks in this interval to times
            ticks = np.arange(end_tick - start_tick + 1)
            self.__tick_to_time[start_tick : end_tick + 1] = last_end_time + tick_scale * ticks
            # Update the time of the last tick in this interval
            last_end_time = self.__tick_to_time[end_tick]
        # For the final interval, use the final tempo setting
        # and ticks from the final tempo setting until max_tick
        start_tick, tick_scale = self._tick_scales[-1]
        ticks = np.arange(max_tick + 1 - start_tick)
        self.__tick_to_time[start_tick:] = last_end_time + tick_scale * ticks

    def _load_instruments(self, midi_data):
        """Populates ``self.instruments`` using ``midi_data``.

        Parameters
        ----------
        midi_data : midi.FileReader
            MIDI object from which data will be read.
        """
        # MIDI files can contain a collection of tracks; each track can have
        # events occuring on one of sixteen channels, and events can correspond
        # to different instruments according to the most recently occurring
        # program number.  So, we need a way to keep track of which instrument
        # is playing on each track on each channel.  This dict will map from
        # program number, drum/not drum, channel, and track index to instrument
        # indices, which we will retrieve/populate using the __get_instrument
        # function below.
        instrument_map = collections.OrderedDict()
        # Store a similar mapping to instruments storing "straggler events",
        # e.g. events which appear before we want to initialize an Instrument
        stragglers = {}
        # This dict will map track indices to any track names encountered
        track_name_map = collections.defaultdict(str)

        def __get_instrument(program, channel, track, create_new):
            """Gets the Instrument corresponding to the given program number,
            drum/non-drum type, channel, and track index.  If no such
            instrument exists, one is created.

            """
            # If we have already created an instrument for this program
            # number/track/channel, return it
            if (program, channel, track) in instrument_map:
                return instrument_map[(program, channel, track)]
            # If there's a straggler instrument for this instrument and we
            # aren't being requested to create a new instrument
            if not create_new and (channel, track) in stragglers:
                return stragglers[(channel, track)]
            # If we are told to, create a new instrument and store it
            if create_new:
                is_drum = channel == 9
                instrument = midi_containers.Instrument(program, is_drum, track_name_map[track_idx])
                # If any events appeared for this instrument before now,
                # include them in the new instrument
                if (channel, track) in stragglers:
                    straggler = stragglers[(channel, track)]
                    instrument.control_changes = straggler.control_changes
                    instrument.pitch_bends = straggler.pitch_bends
                # Add the instrument to the instrument map
                instrument_map[(program, channel, track)] = instrument
            # Otherwise, create a "straggler" instrument which holds events
            # which appear before we actually want to create a proper new
            # instrument
            else:
                # Create a "straggler" instrument
                instrument = midi_containers.Instrument(program, track_name_map[track_idx])
                # Note that stragglers ignores program number, because we want
                # to store all events on a track which appear before the first
                # note-on, regardless of program
                stragglers[(channel, track)] = instrument
            return instrument

        for track_idx, track in enumerate(midi_data.tracks):
            # Keep track of last note on location:
            # key = (instrument, note),
            # value = (note-on tick, velocity)
            last_note_on = collections.defaultdict(list)
            # Keep track of which instrument is playing in each channel
            # initialize to program 0 for all channels
            current_instrument = np.zeros(16, dtype=np.int32)
            for event in track:
                # Look for track name events
                if event.type == "track_name":
                    # Set the track name for the current track
                    track_name_map[track_idx] = event.name
                # Look for program change events
                if event.type == "program_change":
                    # Update the instrument for this channel
                    current_instrument[event.channel] = event.program
                # Note ons are note on events with velocity > 0
                elif event.type == "note_on" and event.velocity > 0:
                    # Store this as the last note-on location
                    note_on_index = (event.channel, event.note)
                    last_note_on[note_on_index].append((event.time, event.velocity))
                # Note offs can also be note on events with 0 velocity
                elif event.type == "note_off" or (event.type == "note_on" and event.velocity == 0):
                    # Check that a note-on exists (ignore spurious note-offs)
                    key = (event.channel, event.note)
                    if key in last_note_on:
                        # Get the start/stop times and velocity of every note
                        # which was turned on with this instrument/drum/pitch.
                        # One note-off may close multiple note-on events from
                        # previous ticks. In case there's a note-off and then
                        # note-on at the same tick we keep the open note from
                        # this tick.
                        end_tick = event.time
                        open_notes = last_note_on[key]

                        notes_to_close = [(start_tick, velocity) for start_tick, velocity in open_notes if start_tick != end_tick]
                        notes_to_keep = [(start_tick, velocity) for start_tick, velocity in open_notes if start_tick == end_tick]

                        for start_tick, velocity in notes_to_close:
                            start_time = self.__tick_to_time[start_tick]
                            end_time = self.__tick_to_time[end_tick]
                            # Create the note event
                            note = midi_containers.Note(velocity, event.note, start_time, end_time)
                            # Get the program and drum type for the current
                            # instrument
                            program = current_instrument[event.channel]
                            # Retrieve the Instrument instance for the current
                            # instrument
                            # Create a new instrument if none exists
                            instrument = __get_instrument(program, event.channel, track_idx, 1)
                            # Add the note event
                            instrument.notes.append(note)

                        if len(notes_to_close) > 0 and len(notes_to_keep) > 0:
                            # Note-on on the same tick but we already closed
                            # some previous notes -> it will continue, keep it.
                            last_note_on[key] = notes_to_keep
                        else:
                            # Remove the last note on for this instrument
                            del last_note_on[key]
                # Store control changes
                elif event.type == "control_change":
                    control_change = midi_containers.ControlChange(event.control, event.value, self.__tick_to_time[event.time])
                    # Get the program for the current inst
                    program = current_instrument[event.channel]
                    # Retrieve the Instrument instance for the current inst
                    # Don't create a new instrument if none exists
                    instrument = __get_instrument(program, event.channel, track_idx, 0)
                    # Add the control change event
                    instrument.control_changes.append(control_change)
        # Initialize list of instruments from instrument_map
        self.instruments = [i for i in instrument_map.values()]

    def tick_to_time(self, tick):
        """
        Converts from an absolute tick to time in seconds using
        ``self.__tick_to_time``.

        Parameters:
            tick (int):
                Absolute tick to convert.

        Returns:
            time (float):
                Time in seconds of tick.

        """
        # Check that the tick isn't too big
        if tick >= MAX_TICK:
            raise IndexError("Supplied tick is too large.")
        # If we haven't compute the mapping for a tick this large, compute it
        if tick >= len(self.__tick_to_time):
            self._update_tick_to_time(tick)
        # Ticks should be integers
        if not isinstance(tick, int):
            showwarning("ticks should be integers", RuntimeWarning, "fortepyan", lineno=1)
        # Otherwise just return the time
        return self.__tick_to_time[int(tick)]

    def get_tempo_changes(self):
        """Return arrays of tempo changes in quarter notes-per-minute and their
        times.
        """
        # Pre-allocate return arrays
        tempo_change_times = np.zeros(len(self._tick_scales))
        tempi = np.zeros(len(self._tick_scales))
        for n, (tick, tick_scale) in enumerate(self._tick_scales):
            # Convert tick of this tempo change to time in seconds
            tempo_change_times[n] = self.tick_to_time(tick)
            # Convert tick scale to a tempo
            tempi[n] = 60.0 / (tick_scale * self.resolution)
        return tempo_change_times, tempi

    def get_end_time(self):
        """
        Returns the time of the end of the MIDI object (time of the last
        event in all instruments/meta-events).

        Returns:
            end_time (float):
                Time, in seconds, where this MIDI file ends.

        """
        # Get end times from all instruments, and times of all meta-events
        meta_events = [self.time_signature_changes, self.key_signature_changes, self.lyrics, self.text_events]
        times = (
            [i.get_end_time() for i in self.instruments]
            + [e.time for m in meta_events for e in m]
            + self.get_tempo_changes()[0].tolist()
        )
        # If there are no events, return 0
        if len(times) == 0:
            return 0.0
        else:
            return max(times)

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

    def time_to_tick(self, time):
        """
        Converts from a time in seconds to absolute tick using
        `self._tick_scales`.

        Parameters:
            time (float):
                Time, in seconds.

        Returns:
            tick (int)
                Absolute tick corresponding to the supplied time.

        """
        # Find the index of the ticktime which is smaller than time
        tick = np.searchsorted(self.__tick_to_time, time, side="left")
        # If the closest tick was the final tick in self.__tick_to_time...
        if tick == len(self.__tick_to_time):
            # start from time at end of __tick_to_time
            tick -= 1
            # Add on ticks assuming the final tick_scale amount
            _, final_tick_scale = self._tick_scales[-1]
            tick += (time - self.__tick_to_time[tick]) / final_tick_scale
            # Re-round/quantize
            return int(round(tick))
        # If the tick is not 0 and the previous ticktime in a is closer to time
        if tick and (math.fabs(time - self.__tick_to_time[tick - 1]) < math.fabs(time - self.__tick_to_time[tick])):
            # Decrement index by 1
            return tick - 1
        else:
            return tick

    def write(self, filename: str):
        """
        Write the MIDI data out to a .mid file.

        Parameters:
            filename (str): Path to write .mid file to.

        """

        def event_compare(event1, event2):
            """
            Compares two events for sorting.

            Events are sorted by tick time ascending. Events with the same tick
            time ares sorted by event type. Some events are sorted by
            additional values. For example, Note On events are sorted by pitch
            then velocity, ensuring that a Note Off (Note On with velocity 0)
            will never follow a Note On with the same pitch.

            Parameters:
                event1, event2 (mido.Message):
                    Two events to be compared.
            """
            # Construct a dictionary which will map event names to numeric
            # values which produce the correct sorting. Each dictionary value
            # is a function which accepts an event and returns a score.
            # The spacing for these scores is 256, which is larger than the
            # largest value a MIDI value can take.
            secondary_sort = {
                "set_tempo": lambda e: (1 * 256 * 256),
                "time_signature": lambda e: (2 * 256 * 256),
                "key_signature": lambda e: (3 * 256 * 256),
                "lyrics": lambda e: (4 * 256 * 256),
                "text_events": lambda e: (5 * 256 * 256),
                "program_change": lambda e: (6 * 256 * 256),
                "pitchwheel": lambda e: ((7 * 256 * 256) + e.pitch),
                "control_change": lambda e: ((8 * 256 * 256) + (e.control * 256) + e.value),
                "note_off": lambda e: ((9 * 256 * 256) + (e.note * 256)),
                "note_on": lambda e: ((10 * 256 * 256) + (e.note * 256) + e.velocity),
                "end_of_track": lambda e: (11 * 256 * 256),
            }
            # If the events have the same tick, and both events have types
            # which appear in the secondary_sort dictionary, use the dictionary
            # to determine their ordering.
            if event1.time == event2.time and event1.type in secondary_sort and event2.type in secondary_sort:
                return secondary_sort[event1.type](event1) - secondary_sort[event2.type](event2)
            # Otherwise, just return the difference of their ticks.
            return event1.time - event2.time

        # Initialize output MIDI object
        mid = mido.MidiFile(ticks_per_beat=self.resolution)
        # Create track 0 with timing information
        timing_track = mido.MidiTrack()
        # Add a default time signature only if there is not one at time 0.
        add_ts = True
        if self.time_signature_changes:
            add_ts = min([ts.time for ts in self.time_signature_changes]) > 0.0
        if add_ts:
            # Add time signature event with default values (4/4)
            timing_track.append(mido.MetaMessage("time_signature", time=0, numerator=4, denominator=4))

        # Add in each tempo change event
        for tick, tick_scale in self._tick_scales:
            timing_track.append(
                mido.MetaMessage(
                    "set_tempo",
                    time=tick,
                    # Convert from microseconds per quarter note to BPM
                    tempo=int(6e7 / (60.0 / (tick_scale * self.resolution))),
                )
            )
        # Add in each time signature
        for ts in self.time_signature_changes:
            timing_track.append(
                mido.MetaMessage(
                    "time_signature", time=self.time_to_tick(ts.time), numerator=ts.numerator, denominator=ts.denominator
                )
            )
        # Add in each key signature
        # Mido accepts key changes in a different format than pretty_midi, this
        # list maps key number to mido key name
        key_number_to_mido_key_name = [
            "C",
            "Db",
            "D",
            "Eb",
            "E",
            "F",
            "F#",
            "G",
            "Ab",
            "A",
            "Bb",
            "B",
            "Cm",
            "C#m",
            "Dm",
            "D#m",
            "Em",
            "Fm",
            "F#m",
            "Gm",
            "G#m",
            "Am",
            "Bbm",
            "Bm",
        ]
        for ks in self.key_signature_changes:
            timing_track.append(
                mido.MetaMessage("key_signature", time=self.time_to_tick(ks.time), key=key_number_to_mido_key_name[ks.key_number])
            )
        # Add in all lyrics events
        for lyr in self.lyrics:
            timing_track.append(mido.MetaMessage("lyrics", time=self.time_to_tick(lyr.time), text=lyr.text))
        # Add text events
        for tex in self.text_events:
            timing_track.append(mido.MetaMessage("text", time=self.time_to_tick(tex.time), text=tex.text))
        # Sort the (absolute-tick-timed) events.
        timing_track.sort(key=functools.cmp_to_key(event_compare))
        # Add in an end of track event
        timing_track.append(mido.MetaMessage("end_of_track", time=timing_track[-1].time + 1))
        mid.tracks.append(timing_track)
        # Create a list of possible channels to assign - this seems to matter
        # for some synths.
        channels = list(range(16))
        # Don't assign the drum channel by mistake!
        channels.remove(9)
        for n, instrument in enumerate(self.instruments):
            # Initialize track for this instrument
            track = mido.MidiTrack()
            # Add track name event if instrument has a name
            if instrument.name:
                track.append(mido.MetaMessage("track_name", time=0, name=instrument.name))
            # If it's a drum event, we need to set channel to 9
            if instrument.is_drum:
                channel = 9
            # Otherwise, choose a channel from the possible channel list
            else:
                channel = channels[n % len(channels)]
            # Set the program number
            track.append(mido.Message("program_change", time=0, program=instrument.program, channel=channel))
            # Add all note events
            for note in instrument.notes:
                # Construct the note-on event
                track.append(
                    mido.Message(
                        "note_on", time=self.time_to_tick(note.start), channel=channel, note=note.pitch, velocity=note.velocity
                    )
                )
                # Also need a note-off event (note on with velocity 0)
                track.append(
                    mido.Message("note_on", time=self.time_to_tick(note.end), channel=channel, note=note.pitch, velocity=0)
                )
            # Add all pitch bend events
            for bend in instrument.pitch_bends:
                track.append(mido.Message("pitchwheel", time=self.time_to_tick(bend.time), channel=channel, pitch=bend.pitch))
            # Add all control change events
            for control_change in instrument.control_changes:
                track.append(
                    mido.Message(
                        "control_change",
                        time=self.time_to_tick(control_change.time),
                        channel=channel,
                        control=control_change.number,
                        value=control_change.value,
                    )
                )
            # Sort all the events using the event_compare comparator.
            track = sorted(track, key=functools.cmp_to_key(event_compare))

            # If there's a note off event and a note on event with the same
            # tick and pitch, put the note off event first
            for n, (event1, event2) in enumerate(zip(track[:-1], track[1:])):
                if (
                    event1.time == event2.time
                    and event1.type == "note_on"
                    and event2.type == "note_on"
                    and event1.note == event2.note
                    and event1.velocity != 0
                    and event2.velocity == 0
                ):
                    track[n] = event2
                    track[n + 1] = event1
            # Finally, add in an end of track event
            track.append(mido.MetaMessage("end_of_track", time=track[-1].time + 1))
            # Add to the list of output tracks
            mid.tracks.append(track)
        # Turn ticks to relative time from absolute
        for track in mid.tracks:
            tick = 0
            for event in track:
                event.time -= tick
                tick += event.time

        # Write it out to a file,
        mid.save(filename=filename)


def __repr__(self):
    return f"MidiFile({self.path})"


def __str__(self):
    return f"MidiFile({self.path})"
