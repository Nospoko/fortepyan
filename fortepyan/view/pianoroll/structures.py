from typing import Union
from warnings import showwarning
from dataclasses import field, dataclass

import matplotlib
import numpy as np
from cmcrameri import cm
from matplotlib.colors import ListedColormap

from fortepyan.midi.structures import MidiPiece
from fortepyan.midi.tools import note_number_to_name


@dataclass
class PianoRoll:
    """
    Represents a piano roll visualization of a MIDI piece.

    The PianoRoll class provides a visual representation of MIDI data as a traditional piano roll,
    which is often used in music software. This representation includes the ability to mark the current time,
    set start and end times for the visualization, and dynamically build the piano roll image based on the MIDI data.

    Attributes:
        midi_piece (MidiPiece): The MIDI piece to be visualized.
        current_time (float, optional): The current time position in the MIDI piece.
        time_start (float): The start time for the piano roll visualization.
        time_end (float, optional): The end time for the piano roll visualization.
        roll (np.array): The numpy array representing the piano roll image.
        RESOLUTION (int): The resolution of the piano roll image.
        N_PITCHES (int): The number of pitches to be represented in the piano roll.

    Methods:
        __post_init__(): Initializes the piano roll image and tick preparations.
        lowest_pitch(): Returns the lowest pitch present in the MIDI piece.
        highest_pitch(): Returns the highest pitch present in the MIDI piece.
        _build_image(): Builds the piano roll image from the MIDI data.
        _prepare_ticks(): Prepares the tick marks and labels for the piano roll visualization.
    """

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


@dataclass
class DualPianoRoll(PianoRoll):
    """
    Extends the PianoRoll class to represent a dual-layer piano roll visualization.

    The DualPianoRoll class enhances the basic piano roll visualization by allowing for the
    representation of additional information, such as masking in machine learning contexts, through
    the use of dual color mapping.

    Attributes:
        base_cmap (Union[str, ListedColormap]): The colormap for the base layer of the piano roll.
        marked_cmap (Union[str, ListedColormap]): The colormap for the marked layer of the piano roll.
        mark_key (str): The key used to determine markings in the MIDI data.

    Methods:
        __post_init__(): Initializes the dual-layer piano roll with specified colormaps.
        _build_image(): Builds the dual-layer piano roll image from the MIDI data, applying color mappings.
    """

    base_cmap: Union[str, ListedColormap] = cm.devon_r
    marked_cmap: Union[str, ListedColormap] = "RdPu"
    mark_key: str = "mask"

    def __post_init__(self):
        # Strings are for the standard set of colormaps
        # ListedColormap is for custom solutions (e.g.: cmcrameri)
        if isinstance(self.base_cmap, ListedColormap):
            self.base_colormap = self.base_cmap
        else:
            self.base_colormap = matplotlib.colormaps[self.base_cmap]

        if isinstance(self.marked_cmap, matplotlib.colors.ListedColormap):
            self.marked_colormap = self.marked_cmap
        else:
            self.marked_colormap = matplotlib.colormaps[self.marked_cmap]
        super().__post_init__()

    def _build_image(self):
        df = self.midi_piece.df_with_end
        if not self.time_end:
            # We don't really need a full second roundup
            self.time_end = np.ceil(df.end.max())

        if self.time_end < df.end.max():
            showwarning("Warning, piano roll is not showing everything!", UserWarning, "pianoroll.py", 164)

        # duration = time_end - time_start
        self.duration = self.time_end
        n_time_steps = self.RESOLUTION * int(np.ceil(self.duration))

        # Adjust velocity color intensity to be sure it's visible
        min_value = 20
        max_value = 160

        # Canvas to draw on
        background = np.zeros((self.N_PITCHES, n_time_steps), np.uint8)

        # Draw black keys with the base colormap
        for it in range(self.N_PITCHES):
            is_black = it % 12 in [1, 3, 6, 8, 10]
            if is_black:
                background[it, :] += min_value
        # This makes the array RGB
        background = self.base_colormap(background)

        # Draw notes
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

            # Colormaps are up to 255, but velocity is up to 127
            color_value += 90

            cmap = self.marked_colormap if row[self.mark_key] else self.base_colormap
            background[pitch_idx, note_on:note_end] = cmap(color_value)

            # pianoroll[pitch_idx, note_on:note_end] = color_value

        self.roll = background


@dataclass
class FigureResolution:
    """
    Represents the resolution configuration for a figure.

    Attributes:
        w_pixels (int): The width of the figure in pixels.
        h_pixels (int): The height of the figure in pixels.
        dpi (int): The dots per inch (resolution) of the figure.

    Properties:
        w_inches (float): The width of the figure in inches, calculated from pixels and dpi.
        h_inches (float): The height of the figure in inches, calculated from pixels and dpi.
        figsize (tuple[float, float]): The size of the figure as a tuple of width and height in inches.
    """

    w_pixels: int = 1920 // 2
    h_pixels: int = 1080 // 2
    dpi: int = 72

    @property
    def w_inches(self) -> float:
        return self.w_pixels / self.dpi

    @property
    def h_inches(self) -> float:
        return self.h_pixels / self.dpi

    @property
    def figsize(self) -> tuple[float, float]:
        return self.w_inches, self.h_inches
