import tempfile
from pathlib import Path
import multiprocessing as mp

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from .. import MidiPiece, roll


class PianoRollScene:
    def __init__(self, piece: MidiPiece, title: str, cmap: str = "GnBu"):
        self.axes = []
        self.content_dir = Path(tempfile.mkdtemp())

        self.frame_paths = []

        self.piece = piece
        self.title = title
        self.cmap = cmap

        f, axes = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=[16, 9],
            gridspec_kw={
                "height_ratios": [4, 1],
                "hspace": 0,
                # 'wspace': 0
            },
        )

        self.figure = f
        self.roll_ax = axes[0]
        self.velocity_ax = axes[1]
        self.axes = [self.roll_ax, self.velocity_ax]

    def draw_all_axes(self, time: float) -> None:
        self.draw_piano_roll(time)
        self.draw_velocities(time)
        roll.sanitize_xticks(self.velocity_ax, self.piece.df)

    def draw_piano_roll(self, time: float) -> None:
        roll.draw_piano_roll(
            ax=self.roll_ax,
            midi_frame=self.piece.df,
            cmap=self.cmap,
            time=time,
        )
        self.roll_ax.set_title(self.title)

    def draw_velocities(self, time: float) -> None:
        roll.draw_velocities(
            ax=self.velocity_ax,
            midi_frame=self.piece.df,
            cmap=self.cmap,
        )

    def save_frame(self, savepath="tmp/tmp.png"):
        self.figure.tight_layout()

        self.figure.savefig(savepath)

        self.clean_figure()

    def clean_figure(self):
        for ax in self.axes:
            ax.clear()

    def render(self, framerate: int = 30) -> None:
        max_time = np.ceil(self.piece.df.end.max()).astype(int)

        n_frames = max_time * framerate
        times = np.linspace(0, max_time - 1 / framerate, n_frames)
        df = pd.DataFrame({"time": times, "counter": range(n_frames)})

        # One big part
        self.animate_part(df)

        return self.content_dir

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

    def render_mp(self, framerate: int = 30) -> None:
        max_time = np.ceil(self.piece.df.end.max()).astype(int)

        n_frames = max_time * framerate
        times = np.linspace(0, max_time - 1 / framerate, n_frames)
        df = pd.DataFrame({"time": times, "counter": range(n_frames)})
        step = 50
        parts = [df[sta : sta + step] for sta in range(0, n_frames, step)]
        with mp.Pool(mp.cpu_count()) as pool:
            pool.map(self.animate_part, parts)

        return self.content_dir
