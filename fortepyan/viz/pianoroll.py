from dataclasses import field, dataclass

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pretty_midi import note_number_to_name

from .. import MidiPiece

# These could be properties of the PianoRoll only
# not global configs
N_PITCHES = 128
RESOLUTION = 30


@dataclass
class PianoRoll:
    roll: np.array
    lowest_pitch: int
    highest_pitch: int
    duration: float
    max_value: int

    # Plot elements
    y_ticks: np.array = field(init=False)
    pitch_labels: list[str] = field(init=False)
    x_ticks: np.array = field(init=False)
    x_labels: list[str] = field(init=False)

    def __post_init__(self):
        # "Octave" mode for y-ticks
        self.y_ticks = np.arange(0, 128, 12, dtype=float)

        # Adding new line shifts the label up a little and positions
        # it nicely at the height where the note actually is
        self.pitch_labels = [f"{note_number_to_name(it)}\n" for it in self.y_ticks]

        # Move the ticks to land between the notes
        # (each note is 1-width and ticks by default are centered, ergo: 0.5 shift)
        self.y_ticks -= 0.5

        # Prepare x ticks and labels
        n_ticks = min(30, self.duration)
        step = np.ceil(self.duration / n_ticks) * RESOLUTION
        x_ticks = np.arange(0, step * n_ticks, step)
        self.x_ticks = np.round(x_ticks)
        self.x_labels = [round(xt / RESOLUTION) for xt in self.x_ticks]


def draw_pianoroll_with_velocities(midi_piece: MidiPiece, cmap: str = "GnBu"):
    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=[16, 9],
        gridspec_kw={
            "height_ratios": [4, 1],
            "hspace": 0,
        },
    )
    piece = sanitize_midi_piece(midi_piece)
    draw_piano_roll(ax=axes[0], midi_piece=piece, cmap=cmap)
    draw_velocities(ax=axes[1], midi_piece=piece, cmap=cmap)

    sanitize_xticks(axes[1], piece)

    return fig


def prepare_piano_roll(
    df: pd.DataFrame,
    time_indicator: float = None,
    time_start: float = 0,
    time_end: float = None,
) -> PianoRoll:
    if not time_end:
        # We don't really need a full second roundup
        time_end = np.ceil(df.end.max())

    if time_end <= df.end.max():
        print("Warning, piano roll is not showing everythin!")

    # duration = time_end - time_start
    duration = time_end
    n_time_steps = RESOLUTION * int(duration)
    pianoroll = np.zeros((N_PITCHES, n_time_steps), np.uint8)

    # Adjust velocity color intensity to be sure it's visible
    min_value = 20
    max_value = 160

    for it, row in df.iterrows():
        note_on = row.start * RESOLUTION
        note_on = np.round(note_on).astype(int)

        note_end = row.end * RESOLUTION
        note_end = np.round(note_end).astype(int)
        pitch_idx = int(row.pitch)

        if time_indicator and note_on <= time_indicator * RESOLUTION < note_end:
            color_value = max_value
        else:
            color_value = min_value + row.velocity
        pianoroll[pitch_idx, note_on:note_end] = color_value

    # Could be a part of "prepare empty piano roll"
    for it in range(N_PITCHES):
        is_black = it % 12 in [1, 3, 6, 8, 10]
        if is_black:
            pianoroll[it, :] += min_value

    pianoroll = PianoRoll(
        roll=pianoroll,
        lowest_pitch=df.pitch.min(),
        highest_pitch=df.pitch.max(),
        duration=duration,
        max_value=max_value,
    )

    return pianoroll


def sanitize_midi_piece(piece: MidiPiece) -> MidiPiece:
    # 20 minutes?
    duration_threshold = 1200
    if piece.duration > duration_threshold:
        # TODO Logger
        print("Warning: playtime to long! Showing after trim")
        piece = piece.trim(0, duration_threshold)

    return piece


def sanitize_midi_frame(mf: pd.DataFrame) -> pd.DataFrame:
    # Do not modify input data
    df = mf.copy()

    # Make it start at 0.0
    df.end -= df.start.min()
    df.start -= df.start.min()
    duration_in = df.end.max()

    # 20 minutes?
    duration_threshold = 1200
    if duration_in > duration_threshold:
        # TODO Logger
        print("Warning: playtime to long! Showing after trim")
        ids = df.end <= duration_threshold
        df = df[ids]

    return df


def draw_piano_roll(
    ax: plt.Axes,
    midi_piece: MidiPiece,
    time: float = 0.0,
    cmap: str = "GnBu",
) -> plt.Axes:
    """
    Draws a pianoroll onto an ax.

    Parameters:
        ax: Matplotlib axis
        midi_piece: MidiPiece with piano performance
        time (Optional[float]): Use for dynamic visualization - will highlight notes
            played at this *time* value
        cmap (str): colormap recognizable by Matplotlib

    Returns:
        ax: Matplotlib axis with pianoroll.
    """
    piece = sanitize_midi_piece(midi_piece)
    piano_roll = prepare_piano_roll(piece.df, time_indicator=time)

    ax.imshow(
        piano_roll.roll,
        aspect="auto",
        vmin=0,
        vmax=138,
        origin="lower",
        interpolation="none",
        cmap=cmap,
    )

    ax.set_yticks(piano_roll.y_ticks)
    ax.set_yticklabels(piano_roll.pitch_labels, fontsize=15)

    # Show keyboard range where the music is
    y_min = piano_roll.lowest_pitch - 1
    y_max = piano_roll.highest_pitch + 1
    ax.set_ylim(y_min, y_max)

    ax.set_xticks(piano_roll.x_ticks)
    ax.set_xticklabels(piano_roll.x_labels, rotation=60)
    ax.set_xlabel("Time [s]")
    ax.set_xlim(0, piano_roll.duration * RESOLUTION)
    ax.grid()

    # Vertical position indicator
    ax.axvline(time * RESOLUTION, color="k", lw=0.5)

    return ax


def draw_velocities(ax: plt.Axes, midi_piece: MidiPiece, cmap: str = "GnBu") -> plt.Axes:
    piece = sanitize_midi_piece(midi_piece)
    df = piece.df
    colormap = matplotlib.cm.get_cmap(cmap)
    color = colormap(125 / 127)

    ax.plot(df.start, df.velocity, "o", ms=7, color=color)
    ax.plot(df.start, df.velocity, ".", color="white")
    ax.vlines(
        df.start,
        ymin=0,
        ymax=df.velocity,
        lw=2,
        alpha=0.777,
        colors=color,
    )
    ax.set_ylim(0, 128)

    return ax


def sanitize_xticks(ax: plt.Axes, piece: MidiPiece):
    """Sanitize the x-axis of a Matplotlib plot for easier readability.

    This function takes two parameters, `ax` and `piece`, which represent the Matplotlib axes object and the midi piece
    that the plot is based on, respectively. The function sets the x-axis tick positions, labels, and limits,
    and adds a grid to the plot to make it easier to read.

    Args:
    - ax (plt.Axes): Matplotlib axes object
    - piece (MidiPiece): `MidiPiece` object used to create the plot
    """
    # Calculate the number of seconds in the plot
    n_seconds = np.ceil(piece.duration)
    # Set the maximum number of x-axis ticks to 30
    n_ticks = min(30, n_seconds)
    # Calculate the step size for the x-axis tick positions
    step = np.ceil(n_seconds / n_ticks)
    # Calculate the x-axis tick positions
    x_ticks = np.arange(0, step * n_ticks, step)
    # Round the x-axis tick positions to the nearest integer
    x_ticks = np.round(x_ticks)
    # Set the x-axis tick labels to the same values as the tick positions
    labels = [xt for xt in x_ticks]

    # Set the x-axis tick positions and labels, and add a label to the x-axis
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(labels, rotation=60, fontsize=15)
    ax.set_xlabel("Time [s]")
    # Set the x-axis limits to the range of the data
    ax.set_xlim(0, n_seconds)
    # Add a grid to the plot
    ax.grid()
