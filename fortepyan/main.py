import numpy as np
import pretty_midi
import pandas as pd
import matplotlib.patches as patches
from matplotlib import pyplot as plt

from fortepyan.midi.tools import note_number_to_name


def process_midi_file(path: str):
    pm = pretty_midi.PrettyMIDI(path)
    pitches = [note.pitch for note in pm.instruments[0].notes]
    pitch_distro = pd.Series(pitches).value_counts().to_dict()

    all_pitches = list(range(21, 108))

    hist = {p: pitch_distro.get(p, 0) for p in all_pitches}
    white_mods = [0, 2, 4, 5, 7, 9, 11]

    white_hist = {}
    black_hist = {}
    for pitch, count in hist.items():
        if pitch % 12 in white_mods:
            white_hist[pitch] = count
        else:
            black_hist[pitch] = count

    ax = draw_histograms(pitches, white_hist, black_hist)

    return ax


def draw_histograms(pitches, white, black):
    all_pitches = list(range(20, 108))
    white_mods = [0, 2, 4, 5, 7, 9, 11]

    key_length = max(max(white.values()), max(black.values())) / 3
    print(key_length)
    fig, ax = plt.subplots(figsize=[14, 3])
    for pitch in all_pitches:
        if pitch % 12 in white_mods:
            patch = patches.Rectangle((pitch - 0.5, -key_length), 1, key_length, edgecolor="k", facecolor="white", alpha=0.9)
        else:
            patch = patches.Rectangle((pitch - 0.5, -key_length), 1, key_length, edgecolor="k", facecolor="black", alpha=0.9)
        ax.add_patch(patch)

    ax.bar(white.keys(), white.values(), color="teal", edgecolor="k")
    ax.bar(black.keys(), black.values(), color="teal", edgecolor="k")

    x_ticks = np.arange(0, 128, 12, dtype=float)
    pitch_labels = [f"{note_number_to_name(it)}" for it in x_ticks]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(pitch_labels)

    yticks = ax.get_yticks()
    yticks = yticks[yticks >= 0]
    ax.set_yticks(yticks)

    ax.set_xlim(min(pitches) - 0.5, max(pitches) + 0.5)
    ax.set_title("Your favorite keys")

    ax.set_xlabel("Pitch Distribution", fontsize=14)

    return ax
