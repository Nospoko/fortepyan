from matplotlib import pyplot as plt

import fortepyan as ff
from fortepyan import MidiPiece
from fortepyan.view.pianoroll.structures import PianoRoll


def draw_vairant(
    variant: dict,
    piece: MidiPiece,
    n: int,
):
    howmany = variant["n_variants"]
    left_shift = variant["left_shift"]
    right_shift = variant["right_shift"]
    idxs = variant["idxs"]
    print("Variants:", howmany)
    print("Shifts:", "left:", left_shift, "right:", right_shift)

    fig, axes = plt.subplots(nrows=howmany, ncols=1, figsize=[10, 2 * howmany])
    for ax, it in zip(axes, idxs):
        p = piece[it - left_shift : it + n + right_shift]
        pr = PianoRoll(p)
        ff.view.pianoroll.draw_piano_roll(ax=ax, piano_roll=pr)
        ax.set_title(f"Index: {it}")
