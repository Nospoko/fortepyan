import tempfile
from pathlib import Path
import multiprocessing as mp

import pandas as pd
from matplotlib import pyplot as plt

from fortepyan.viz import pianoroll as roll
from fortepyan.viz.structures import PianoRoll
from fortepyan.midi.structures import MidiPiece


class MutePianoRollEvolution:
    def __init__(
        self,
        pieces: list[MidiPiece],
        title_format: str = "{}",
        cmap: str = "GnBu",
    ):
        self.axes = []
        self.content_dir = Path(tempfile.mkdtemp())

        self.frame_paths = []

        self.pieces = pieces
        self.time_end = max([p.df.end.max() for p in pieces])
        self.title_format = title_format
        self.cmap = cmap

        f, axes = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=[16, 9],
            gridspec_kw={
                "height_ratios": [4, 1],
                "hspace": 0,
            },
        )

        self.figure = f
        self.roll_ax = axes[0]
        self.velocity_ax = axes[1]
        self.axes = [self.roll_ax, self.velocity_ax]

    def clean_figure(self):
        for ax in self.axes:
            ax.clear()

    def save_frame(self, savepath="tmp/tmp.png"):
        self.figure.tight_layout()

        self.figure.savefig(savepath)

        self.clean_figure()

    def draw(self, step: int) -> None:
        self.clean_figure()
        self.figure.tight_layout()
        self.draw_all_axes(step)

    def draw_all_axes(self, step: int) -> None:
        piece = self.pieces[step]
        piano_roll = PianoRoll(piece, time_end=self.time_end)

        # Piano roll
        roll.draw_piano_roll(
            ax=self.roll_ax,
            piano_roll=piano_roll,
            cmap=self.cmap,
        )
        title = self.title_format.format(step)
        self.roll_ax.set_title(title, fontsize=20)

        # Velocities
        roll.draw_velocities(
            ax=self.velocity_ax,
            piano_roll=piano_roll,
            cmap=self.cmap,
        )

    def prepare_animation_steps(self) -> pd.DataFrame:
        """
        Prepare the data required for the animation.

        Returns:
            pd.DataFrame: DataFrame containing time and counter for each frame.
        """
        steps = list(range(len(self.pieces)))
        df = pd.DataFrame({"step": steps, "counter": steps})

        return df

    def render(self) -> None:
        """
        Render the animation using a single process.
        """
        df = self.prepare_animation_steps()

        # Call the animate_part function with the entire DataFrame as an argument
        self.animate_part(df)

        # Return the directory containing the generated animation content
        return self.content_dir

    def animate_part(self, part: pd.DataFrame):
        for it, row in part.iterrows():
            step = row.step
            frame_counter = int(row.counter)
            self.draw(step=step)
            savepath = self.content_dir / f"{100000 + frame_counter}.png"
            self.save_frame(savepath)
            self.frame_paths.append(savepath)

    def render_mp(self) -> None:
        """
        Render the animation using multi-processing to speed up the process.
        """
        df = self.prepare_animation_steps()

        # Step size for dividing the DataFrame into parts
        step = 50

        # Divide the DataFrame into parts based on the step size
        parts = [df[sta : sta + step] for sta in range(0, df.shape[0], step)]

        # Create a pool of processes using all available CPU cores
        with mp.Pool(mp.cpu_count()) as pool:
            # Map the animate_part function to each part of the DataFrame
            pool.map(self.animate_part, parts)

        # Return the directory containing the generated animation content
        return self.content_dir
