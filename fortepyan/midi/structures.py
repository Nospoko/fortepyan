import collections
from warnings import showwarning
from dataclasses import field, dataclass

import mido
import numpy as np
import pretty_midi
import pandas as pd

from fortepyan.midi import tools as midi_tools

# The largest we'd ever expect a tick to be
MAX_TICK = 1e7


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

            An example of a trimmed MIDI piece:
            ![Trimmed MIDI piece](../assets/random_midi_piece.png)

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
    resolution: int = field(init=False)
    df: pd.DataFrame = field(init=False)
    raw_df: pd.DataFrame = field(init=False)
    sustain: pd.DataFrame = field(init=False)
    control_frame: pd.DataFrame = field(init=False, repr=False)
    instruments: list = field(init=False, repr=False)

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
        # Read the MIDI object
        midi_data = mido.MidiFile(filename=self.path)

        # Convert tick values in midi_data to absolute, a useful thing.
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
        max_tick = max([max([e.time for e in t]) for t in midi_data.tracks]) + 1
        # If max_tick is too big, the MIDI file is probably corrupt
        # and creating the __tick_to_time array will thrash memory
        if max_tick > MAX_TICK:
            raise ValueError(("MIDI file has a largest tick of {}," " it is likely corrupt".format(max_tick)))

        # Create list that maps ticks to time in seconds
        self._update_tick_to_time(max_tick)

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

        # TODO: change according to the new structure
        # MIDIFILE FUNCTIONS BELOW
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
                instrument = Instrument(program, is_drum, track_name_map[track_idx])
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
                instrument = Instrument(program, track_name_map[track_idx])
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
                            note = Note(velocity, event.note, start_time, end_time)
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
                    control_change = ControlChange(event.control, event.value, self.__tick_to_time[event.time])
                    # Get the program for the current inst
                    program = current_instrument[event.channel]
                    # Retrieve the Instrument instance for the current inst
                    # Don't create a new instrument if none exists
                    instrument = __get_instrument(program, event.channel, track_idx, 0)
                    # Add the control change event
                    instrument.control_changes.append(control_change)
        # Initialize list of instruments from instrument_map
        self.instruments = [i for i in instrument_map.values()]

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
        """Returns the time of the end of the MIDI object (time of the last
        event in all instruments/meta-events).

        Returns
        -------
        end_time : float
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


# Container classes from PrettyMIDI
class Instrument(object):
    """Object to hold event information for a single instrument.

    Parameters:
        program (int): MIDI program number (instrument index), in ``[0, 127]``.
        is_drum (bool, optinal): Is the instrument a drum instrument (channel 9)?
        name (str, optional): Name of the instrument.

    Notes:
        It's a container class used to store notes, and control changes. Adapted from [pretty_midi](https://github.com/craffel/pretty-midi).

    """

    def __init__(self, program, is_drum=False, name=""):
        self.program = program
        self.is_drum = is_drum
        self.name = name
        self.notes = []
        self.control_changes = []


class Note(object):
    """A note event.

    Parameters:
        velocity (int): Note velocity.
        pitch (int): Note pitch, as a MIDI note number.
        start (float): Note on time, absolute, in seconds.
        end (float): Note off time, absolute, in seconds.

    Notes:
        It's a container class used to store a note. Adapted from [pretty_midi](https://github.com/craffel/pretty-midi).

    """

    def __init__(self, velocity, pitch, start, end):
        if end < start:
            raise ValueError("Note end time must be greater than start time")

        self.velocity = velocity
        self.pitch = pitch
        self.start = start
        self.end = end

    def get_duration(self):
        """
        Get the duration of the note in seconds.
        """
        return self.end - self.start

    @property
    def duration(self):
        return self.get_duration()

    def __repr__(self):
        return "Note(start={:f}, end={:f}, pitch={}, velocity={})".format(self.start, self.end, self.pitch, self.velocity)


class ControlChange(object):
    """
    A control change event.

    Parameters:
        number (int): The control change number, in ``[0, 127]``.
        value (int): The value of the control change, in ``[0, 127]``.
        time (float): Time where the control change occurs.

    Notes:
        It's a container class used to store a control change. Adapted from [pretty_midi](https://github.com/craffel/pretty-midi).
    """

    def __init__(self, number, value, time):
        self.number = number
        self.value = value
        self.time = time

    def __repr__(self):
        return "ControlChange(number={:d}, value={:d}, " "time={:f})".format(self.number, self.value, self.time)
