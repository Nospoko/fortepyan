from dataclasses import dataclass

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pretty_midi import note_number_to_name

# These could be properties of the PianoRoll only
# not global configs
N_PITCHES = 128
RESOLUTION = 30


@dataclass
class PianoRoll:
    roll: np.array
    lowest_pitch: int
    highest_pitch: int
    n_seconds: int
    max_value: int


def draw_pianoroll_with_velocities(mf: pd.DataFrame, cmap: str = "GnBu"):
    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=[16, 9],
        gridspec_kw={
            "height_ratios": [4, 1],
            "hspace": 0,
        },
    )
    df = sanitize_midi_frame(mf)
    draw_piano_roll(ax=axes[0], midi_frame=df, cmap=cmap)
    draw_velocities(ax=axes[1], midi_frame=df, cmap=cmap)

    sanitize_xticks(axes[1], df)

    return fig


def prepare_piano_roll(df: pd.DataFrame, time: float = None) -> PianoRoll:
    n_seconds = np.ceil(df.end.max())
    n_time_steps = RESOLUTION * int(n_seconds)
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

        if time and note_on <= time * RESOLUTION < note_end:
            color_value = 160
        else:
            color_value = min_value + row.velocity
        pianoroll[pitch_idx, note_on:note_end] = color_value

    for it in range(N_PITCHES):
        is_black = it % 12 in [1, 3, 6, 8, 10]
        if is_black:
            pianoroll[it, :] += min_value

    lowest_pitch = df.pitch.min()
    highest_pitch = df.pitch.max()
    pianoroll = PianoRoll(
        roll=pianoroll,
        lowest_pitch=lowest_pitch,
        highest_pitch=highest_pitch,
        n_seconds=n_seconds,
        max_value=max_value,
    )

    return pianoroll


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
    midi_frame: pd.DataFrame,
    time: float = 0.0,
    cmap: str = "GnBu",
) -> plt.Axes:
    """
    Draws a pianoroll onto an ax.

    Parameters:
        ax: Matplotlib axis
        midi_frame: Dataframe with piano key strokes
        time (Optional[float]): Use for dynamic visualization - will highlight notes
            played at this *time* value
        cmap (str): colormap recognizable by Matplotlib

    Returns:
        ax: Matplotlib axis with pianoroll.
    """
    df = sanitize_midi_frame(midi_frame)
    piano_roll = prepare_piano_roll(df, time=time)

    ax.imshow(
        piano_roll.roll,
        aspect="auto",
        vmin=0,
        vmax=138,
        origin="lower",
        interpolation="none",
        cmap=cmap,
    )

    # "Octave" mode for y-ticks
    y_ticks = np.arange(0, 128, 12, dtype=float)

    # Adding new line shifts the label up a little and positions
    # it nicely at the height where the note actually is
    pitch_labels = [f"{note_number_to_name(it)}\n" for it in y_ticks]

    # Move the ticks to land between the notes
    # (each note is 1-width and ticks by default are centered, ergo: 0.5 shift)
    y_ticks -= 0.5
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(pitch_labels)

    # Show keyboard range where the music is
    y_min = piano_roll.lowest_pitch - 1
    y_max = piano_roll.highest_pitch + 1
    ax.set_ylim(y_min, y_max)

    # Prepare x ticks and labels
    n_ticks = min(30, piano_roll.n_seconds)
    step = np.ceil(piano_roll.n_seconds / n_ticks) * RESOLUTION
    x_ticks = np.arange(0, step * n_ticks, step)
    x_ticks = np.round(x_ticks)
    labels = [round(xt / RESOLUTION) for xt in x_ticks]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(labels, rotation=60)
    ax.set_xlabel("Time [s]")
    ax.set_xlim(0, piano_roll.n_seconds * RESOLUTION)
    ax.grid()

    # Vertical position indicator
    ax.axvline(time * RESOLUTION, color="k", lw=0.5)

    return ax


def draw_velocities(ax: plt.Axes, midi_frame: pd.DataFrame, cmap: str = "GnBu") -> plt.Axes:
    df = sanitize_midi_frame(midi_frame)
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


def sanitize_xticks(ax: plt.Axes, df: pd.DataFrame):
    # Prepare x ticks and labels
    n_seconds = np.ceil(df.end.max())
    n_ticks = min(30, n_seconds)
    step = np.ceil(n_seconds / n_ticks)
    x_ticks = np.arange(0, step * n_ticks, step)
    x_ticks = np.round(x_ticks)
    labels = [xt for xt in x_ticks]

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(labels, rotation=60)
    ax.set_xlabel("Time [s]")
    ax.set_xlim(0, n_seconds)
    ax.grid()
