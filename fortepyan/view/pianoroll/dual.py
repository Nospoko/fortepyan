from typing import Union

from cmcrameri import cm
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from fortepyan.midi.structures import MidiPiece
from fortepyan.view.pianoroll import main as pianoroll_view
from fortepyan.view.pianoroll.structures import DualPianoRoll, FigureResolution


def draw_dual_pianoroll(
    piece: MidiPiece,
    title: str = None,
    figres: FigureResolution = None,
    base_cmap: Union[str, ListedColormap] = cm.devon_r,
    marked_cmap: Union[str, ListedColormap] = "RdPu",
):
    if not figres:
        figres = FigureResolution()

    piano_roll = DualPianoRoll(
        midi_piece=piece,
        base_cmap=base_cmap,
        marked_cmap=marked_cmap,
    )

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

    pianoroll_view.draw_piano_roll(axes[0], piano_roll)
    v_ax = axes[1]
    draw_velocities(ax=v_ax, piano_roll=piano_roll)

    if title:
        axes[0].set_title(title, fontsize=20)

    # Set the x-axis tick positions and labels, and add a label to the x-axis
    v_ax.set_xticks(piano_roll.x_ticks)
    v_ax.set_xticklabels(piano_roll.x_labels, rotation=60, fontsize=15)
    v_ax.set_xlabel("Time [s]")
    # Set the x-axis limits to the range of the data
    v_ax.set_xlim(0, piano_roll.duration)

    return fig


def draw_velocities(
    ax: plt.Axes,
    piano_roll: DualPianoRoll,
) -> plt.Axes:
    df = piano_roll.midi_piece.df
    base_color = piano_roll.base_colormap(125 / 127)
    marked_color = piano_roll.marked_colormap(125 / 127)

    ids = df[piano_roll.mark_key]
    for jds, color in zip([~ids, ids], [base_color, marked_color]):
        ax.plot(df[jds].start, df[jds].velocity, "o", ms=7, color=color)
        # This could be 0-value color :thinking:
        ax.plot(df[jds].start, df[jds].velocity, ".", color="white")

        ax.vlines(
            df[jds].start,
            ymin=0,
            ymax=df[jds].velocity,
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
