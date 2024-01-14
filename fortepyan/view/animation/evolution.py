import tempfile
from pathlib import Path
import multiprocessing as mp

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from fortepyan.midi.structures import MidiPiece
from fortepyan.view.pianoroll import main as roll
from fortepyan.view.pianoroll.structures import PianoRoll


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

        # Set the x-axis tick positions and labels, and add a label to the x-axis
        self.velocity_ax.set_xticks(piano_roll.x_ticks)
        self.velocity_ax.set_xticklabels(piano_roll.x_labels, rotation=60, fontsize=15)
        self.velocity_ax.set_xlabel("Time [s]")
        # Set the x-axis limits to the range of the data
        self.velocity_ax.set_xlim(0, piano_roll.duration)

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


class EvolvingPianoRollScene:
    def __init__(
        self,
        pieces: list[MidiPiece],
        title_format: str = "{}",
        title_key: str = None,
        cmap: str = "GnBu",
    ):
        self.axes = []
        self.content_dir = Path(tempfile.mkdtemp())

        self.frame_paths = []

        self.pieces = pieces
        n_steps = len(pieces)
        self.duration = max([piece.df_with_end.end.max() for piece in pieces])
        self.time_per_step = self.duration / n_steps

        self.cmap = cmap
        self.title_key = title_key
        self.title_format = title_format

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

    def draw_all_axes(self, time: float) -> None:
        piece_id = int(np.floor(time / self.time_per_step))
        piece_id = min(piece_id, len(self.pieces) - 1)
        piece = self.pieces[piece_id]
        piano_roll = PianoRoll(
            midi_piece=piece,
            current_time=time,
            time_end=self.duration,
        )

        title_info = piece.source[self.title_key]
        title = self.title_format.format(title_info)
        self.roll_ax.set_title(title, fontsize=20)

        self.draw_piano_roll(piano_roll, time)
        self.draw_velocities(piano_roll, time)

        # Set the x-axis tick positions and labels, and add a label to the x-axis
        self.velocity_ax.set_xticks(piano_roll.x_ticks)
        self.velocity_ax.set_xticklabels(piano_roll.x_labels, rotation=60, fontsize=15)
        self.velocity_ax.set_xlabel("Time [s]")
        # Set the x-axis limits to the range of the data
        self.velocity_ax.set_xlim(0, piano_roll.duration)

    def draw_piano_roll(self, piano_roll: PianoRoll, time: float) -> None:
        roll.draw_piano_roll(
            ax=self.roll_ax,
            piano_roll=piano_roll,
            cmap=self.cmap,
            time=time,
        )

    def draw_velocities(self, piano_roll: PianoRoll, time: float) -> None:
        roll.draw_velocities(
            ax=self.velocity_ax,
            piano_roll=piano_roll,
            cmap=self.cmap,
        )

    def save_frame(self, savepath="tmp/tmp.png"):
        self.figure.tight_layout()

        self.figure.savefig(savepath)

        self.clean_figure()

    def clean_figure(self):
        for ax in self.axes:
            ax.clear()

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

    def prepare_animation_steps(self, framerate: int = 30) -> pd.DataFrame:
        """
        Prepare the data required for the animation.

        Parameters:
            framerate (int): Framerate for the animation (default is 30).

        Returns:
            pd.DataFrame: DataFrame containing time and counter for each frame.
        """
        # Calculate the maximum time required for the animation
        max_time = np.ceil(self.duration).astype(int)
        # Calculate the number of frames required for the animation
        n_frames = max_time * framerate
        # Create an array of times that will be used to create the animation
        times = np.linspace(0, max_time - 1 / framerate, n_frames)
        # Create a DataFrame to store the time and counter for each frame
        df = pd.DataFrame({"time": times, "counter": range(n_frames)})

        return df

    def render(self, framerate: int = 30) -> None:
        """
        Render the animation using a single process.

        Parameters:
            framerate (int): Framerate for the animation (default is 30).
        """
        df = self.prepare_animation_steps(framerate)

        # Call the animate_part function with the entire DataFrame as an argument
        self.animate_part(df)

        # Return the directory containing the generated animation content
        return self.content_dir

    def render_mp(self, framerate: int = 30) -> None:
        """
        Render the animation using multi-processing to speed up the process.

        Parameters:
            framerate (int): Framerate for the animation (default is 30).
        """
        df = self.prepare_animation_steps(framerate)

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


class EvolvingPianoRollSceneWithChart:
    def __init__(
        self,
        pieces: list[MidiPiece],
        chart_data: list[float],
        title: str,
        cmap: str = "GnBu",
    ):
        self.axes = []
        self.content_dir = Path(tempfile.mkdtemp())

        self.frame_paths = []

        self.chart_data = chart_data

        self.pieces = pieces
        n_steps = len(pieces)
        self.duration = max([piece.df_with_end.end.max() for piece in pieces])
        self.time_per_step = self.duration / n_steps

        self.cmap = cmap
        self.title = title

        f, axes = plt.subplots(
            nrows=3,
            ncols=1,
            figsize=[16, 9],
            gridspec_kw={
                "height_ratios": [4, 1, 1],
                "hspace": 0,
            },
        )

        self.figure = f
        self.roll_ax = axes[0]
        self.velocity_ax = axes[1]
        self.chart_ax = axes[2]
        self.axes = [self.roll_ax, self.velocity_ax, self.chart_ax]

    def draw_all_axes(self, time: float) -> None:
        piece_id = int(np.floor(time / self.time_per_step))
        piece_id = min(piece_id, len(self.pieces) - 1)
        piece = self.pieces[piece_id]
        piano_roll = PianoRoll(
            midi_piece=piece,
            current_time=time,
            time_end=self.duration,
        )

        self.roll_ax.set_title(self.title, fontsize=30)

        self.draw_piano_roll(piano_roll, time)
        self.draw_velocities(piano_roll, time)
        self.draw_chart(time)

        # Set the x-axis tick positions and labels, and add a label to the x-axis
        self.chart_ax.set_xticks(piano_roll.x_ticks)
        self.chart_ax.set_xticklabels(piano_roll.x_labels, rotation=60, fontsize=15)
        self.chart_ax.set_xlabel("Time [s]")
        # Set the x-axis limits to the range of the data
        self.chart_ax.set_xlim(0, piano_roll.duration)

    def draw_chart(self, time: float):
        x = np.linspace(0, self.duration, len(self.chart_data))
        self.chart_ax.plot(x, self.chart_data, label="Diffusion Amplitude")
        self.chart_ax.grid()
        self.chart_ax.legend(loc="upper left", fontsize=16)
        self.chart_ax.axvline(time, color="k", lw=0.5)

    def draw_piano_roll(self, piano_roll: PianoRoll, time: float) -> None:
        roll.draw_piano_roll(
            ax=self.roll_ax,
            piano_roll=piano_roll,
            cmap=self.cmap,
            time=time,
        )

    def draw_velocities(self, piano_roll: PianoRoll, time: float) -> None:
        roll.draw_velocities(
            ax=self.velocity_ax,
            piano_roll=piano_roll,
            cmap=self.cmap,
        )

        # Set the x-axis tick positions and labels, and add a label to the x-axis
        self.velocity_ax.set_xticks(piano_roll.x_ticks)
        self.velocity_ax.set_xticklabels(piano_roll.x_labels, rotation=60, fontsize=15)
        self.velocity_ax.set_xlabel("Time [s]")
        # Set the x-axis limits to the range of the data
        self.velocity_ax.set_xlim(0, piano_roll.duration)

    def save_frame(self, savepath="tmp/tmp.png"):
        self.figure.tight_layout()

        self.figure.savefig(savepath)

        self.clean_figure()

    def clean_figure(self):
        for ax in self.axes:
            ax.clear()

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

    def prepare_animation_steps(self, framerate: int = 30) -> pd.DataFrame:
        """
        Prepare the data required for the animation.

        Parameters:
            framerate (int): Framerate for the animation (default is 30).

        Returns:
            pd.DataFrame: DataFrame containing time and counter for each frame.
        """
        # Calculate the maximum time required for the animation
        max_time = np.ceil(self.duration).astype(int)
        # Calculate the number of frames required for the animation
        n_frames = max_time * framerate
        # Create an array of times that will be used to create the animation
        times = np.linspace(0, max_time - 1 / framerate, n_frames)
        # Create a DataFrame to store the time and counter for each frame
        df = pd.DataFrame({"time": times, "counter": range(n_frames)})

        return df

    def render(self, framerate: int = 30) -> None:
        """
        Render the animation using a single process.

        Parameters:
            framerate (int): Framerate for the animation (default is 30).
        """
        df = self.prepare_animation_steps(framerate)

        # Call the animate_part function with the entire DataFrame as an argument
        self.animate_part(df)

        # Return the directory containing the generated animation content
        return self.content_dir

    def render_mp(self, framerate: int = 30) -> None:
        """
        Render the animation using multi-processing to speed up the process.

        Parameters:
            framerate (int): Framerate for the animation (default is 30).
        """
        df = self.prepare_animation_steps(framerate)

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
