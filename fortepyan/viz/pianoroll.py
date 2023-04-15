import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from fortepyan.viz.structures import PianoRoll
from fortepyan.midi.structures import MidiPiece


def draw_pianoroll_with_velocities(
    midi_piece: MidiPiece,
    time_end: float = None,
    title: str = None,
    cmap: str = "GnBu",
):
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
    piano_roll = PianoRoll(piece, time_end=time_end)
    draw_piano_roll(ax=axes[0], piano_roll=piano_roll, cmap=cmap)
    v_ax = axes[1]
    draw_velocities(ax=v_ax, piano_roll=piano_roll, cmap=cmap)

    if title:
        axes[0].set_title(title, fontsize=20)

    # Set the x-axis tick positions and labels, and add a label to the x-axis
    v_ax.set_xticks(piano_roll.x_ticks)
    v_ax.set_xticklabels(piano_roll.x_labels, rotation=60, fontsize=15)
    v_ax.set_xlabel("Time [s]")
    # Set the x-axis limits to the range of the data
    v_ax.set_xlim(0, piano_roll.duration)

    return fig


def sanitize_midi_piece(piece: MidiPiece) -> MidiPiece:
    # 20 minutes?
    duration_threshold = 1200
    if piece.duration > duration_threshold:
        # TODO Logger
        print("Warning: playtime too long! Showing after trim")
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
    piano_roll: PianoRoll,
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

    ax.set_xticks(piano_roll.x_ticks * piano_roll.RESOLUTION)
    ax.set_xticklabels(piano_roll.x_labels, rotation=60)
    ax.set_xlabel("Time [s]")
    ax.set_xlim(0, piano_roll.duration * piano_roll.RESOLUTION)
    ax.grid()

    # Vertical position indicator
    if piano_roll.current_time:
        ax.axvline(piano_roll.current_time * piano_roll.RESOLUTION, color="k", lw=0.5)

    return ax


def draw_velocities(
    ax: plt.Axes,
    piano_roll: PianoRoll,
    cmap: str = "GnBu",
) -> plt.Axes:
    df = piano_roll.midi_piece.df
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
    # Add a grid to the plot
    ax.grid()

    # Vertical position indicator
    if piano_roll.current_time:
        ax.axvline(piano_roll.current_time, color="k", lw=0.5)

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
