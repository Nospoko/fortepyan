import tempfile
from pathlib import Path
from typing import Union

import pandas as pd
from cmcrameri import cm
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from fortepyan.midi.structures import MidiPiece
from fortepyan.view.pianoroll import dual as dual_roll
from fortepyan.view.pianoroll import main as roll_view
from fortepyan.view.animation.pianoroll import PianoRollScene
from fortepyan.view.pianoroll.structures import DualPianoRoll, FigureResolution


class DualRollScene(PianoRollScene):
    def __init__(
        self,
        piece: MidiPiece,
        title: str,
        base_cmap: Union[str, ListedColormap] = cm.devon_r,
        marked_cmap: Union[str, ListedColormap] = "RdPu",
        figres: FigureResolution = None,
    ):
        self.axes = []
        self.content_dir = Path(tempfile.mkdtemp())

        self.frame_paths = []

        self.piece = piece
        self.title = title
        self.base_cmap = base_cmap
        self.marked_cmap = marked_cmap

        if not figres:
            figres = FigureResolution()

        f, axes = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=figres.figsize,
            dpi=figres.dpi,
            gridspec_kw={
                "height_ratios": [4, 1],
                "hspace": 0,
            },
        )

        self.figure = f
        self.roll_ax = axes[0]
        self.velocity_ax = axes[1]
        self.axes = [self.roll_ax, self.velocity_ax]

    def draw_all_axes(self, time: float) -> None:
        self.draw_piano_roll(time)
        self.draw_velocities(time)

    def draw_piano_roll(self, time: float) -> None:
        piano_roll = DualPianoRoll(
            midi_piece=self.piece,
            base_cmap=self.base_cmap,
            marked_cmap=self.marked_cmap,
            current_time=time,
        )
        roll_view.draw_piano_roll(
            ax=self.roll_ax,
            piano_roll=piano_roll,
            time=time,
        )
        self.roll_ax.set_title(self.title, fontsize=20)

    def draw_velocities(self, time: float) -> None:
        piano_roll = DualPianoRoll(self.piece)
        dual_roll.draw_velocities(
            ax=self.velocity_ax,
            piano_roll=piano_roll,
        )

        # Set the x-axis tick positions and labels, and add a label to the x-axis
        self.velocity_ax.set_xticks(piano_roll.x_ticks)
        self.velocity_ax.set_xticklabels(piano_roll.x_labels, rotation=60, fontsize=15)
        self.velocity_ax.set_xlabel("Time [s]")
        # Set the x-axis limits to the range of the data
        self.velocity_ax.set_xlim(0, piano_roll.duration)

    def animate_part(self, part: pd.DataFrame):
        for it, row in part.iterrows():
            time = row.time
            frame_counter = int(row.counter)
            self.draw(time)
            savepath = self.content_dir / f"{100000 + frame_counter}.png"
            self.save_frame(savepath)
            self.frame_paths.append(savepath)

    def draw(self, time: float) -> None:
        self.clean_figure()
        self.figure.tight_layout()
        self.draw_all_axes(time)
