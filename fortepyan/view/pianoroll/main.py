from warnings import showwarning

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from fortepyan.midi.structures import MidiPiece
from fortepyan.view.pianoroll.structures import PianoRoll, FigureResolution


def draw_pianoroll_with_velocities(
    midi_piece: MidiPiece,
    time_end: float = None,
    title: str = None,
    cmap: str = "GnBu",
    figres: FigureResolution = None,
):
    """
    Draws a pianoroll representation of a MIDI piece with an additional plot for velocities.

    This function creates a two-part plot with the upper part displaying the pianoroll and the lower part showing
    the velocities of the notes. Customizable aspects include the end time for the plot, the title, color mapping,
    and figure resolution.

    Args:
        midi_piece (MidiPiece): The MIDI piece to be visualized.
        time_end (float, optional): End time for the plot. Defaults to None, meaning full duration is used.
        title (str, optional): Title for the plot. Defaults to None.
        cmap (str): Color map for the pianoroll and velocities. Defaults to "GnBu".
        figres (FigureResolution, optional): Custom figure resolution settings. Defaults to None,
                                              which initiates a default `FigureResolution`.

    Returns:
        fig (plt.Figure): A matplotlib figure object with the pianoroll and velocity plots.
    """
    if not figres:
        figres = FigureResolution()

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=figres.figsize,
        dpi=figres.dpi,
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
    """
    Trims a MIDI piece to a maximum duration threshold for manageability.

    If the duration of the MIDI piece exceeds a predefined threshold, it trims the piece to fit within this limit.
    This function is useful to avoid excessively long playtimes which might be impractical for visualization or analysis.

    Args:
        piece (MidiPiece): The MIDI piece to be sanitized.

    Returns:
        MidiPiece: The sanitized MIDI piece, trimmed if necessary.
    """
    duration_threshold = 1200
    if piece.duration > duration_threshold:
        # TODO Logger
        showwarning("playtime too long! Showing after trim", RuntimeWarning, filename="", lineno=0)
        piece = piece.trim(
            0, duration_threshold, slice_type="by_end", shift_time=False
        )  # Added "by_end" to make sure a very long note doesn't cause an error

    return piece


def draw_piano_roll(
    ax: plt.Axes,
    piano_roll: PianoRoll,
    time: float = 0.0,
    cmap: str = "GnBu",
) -> plt.Axes:
    """
    Draws a piano roll visualization on a Matplotlib axis.

    This function visualizes the piano roll of a MIDI piece on a given Matplotlib axis. It includes options to highlight
    notes played at a specific time and to customize the color mapping.

    Args:
        ax (plt.Axes): The Matplotlib axis on which to draw the piano roll.
        piano_roll (PianoRoll): The PianoRoll object representing the MIDI piece.
        time (float, optional): The specific time at which to highlight notes. Defaults to 0.0.
        cmap (str): The color map to use for the visualization. Defaults to "GnBu".

    Returns:
        plt.Axes: The modified Matplotlib axis with the piano roll visualization.
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
    """
    Draws a velocity plot for a MIDI piece on a Matplotlib axis.

    This function visualizes the velocities of notes in a MIDI piece using a scatter and line plot. The color and
    style of the plot can be customized using a colormap.

    Args:
        ax (plt.Axes): The Matplotlib axis on which to draw the velocity plot.
        piano_roll (PianoRoll): The PianoRoll object representing the MIDI piece.
        cmap (str): The color map to use for the visualization. Defaults to "GnBu".

    Returns:
        plt.Axes: The modified Matplotlib axis with the velocity plot.
    """
    df = piano_roll.midi_piece.df
    colormap = matplotlib.colormaps.get_cmap(cmap)
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
    """
    Adjusts the x-axis ticks and labels for a MIDI piece plot for improved readability.

    This function computes and sets appropriate tick marks and labels on the x-axis of a Matplotlib plot based on the
    duration of a MIDI piece. It ensures the plot is easy to read and interpret by adjusting the frequency and format
    of the x-axis ticks.

    Args:
        ax (plt.Axes): The Matplotlib axes object to be modified.
        piece (MidiPiece): The MIDI piece based on which the axis ticks and labels are adjusted.
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
