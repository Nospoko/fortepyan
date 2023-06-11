import string
from dataclasses import field, dataclass

import pandas as pd

from fortepyan import MidiPiece

PITCH_MAP = {20 + it: string.printable.strip()[it] for it in range(90)}


@dataclass
class NgramContainer:
    n: int
    piece: MidiPiece
    df: pd.DataFrame = field(init=False)
    pitch_sequence: list[str] = field(init=False)

    def __post_init__(self):
        df = self.piece.df.copy()

        # There are <90 pitches, and 94 printable characters
        # We convert pitches to unique chars - this makes
        # Fuzzy-wuzzy Levenshtein string matching easier to follow
        df["pitch_char"] = df.pitch.map(PITCH_MAP)

        howmany = self.piece.size - self.n + 1

        pitch_chars = df.pitch_char.values

        ngrams = ["".join(pitch_chars[it : it + self.n]) for it in range(howmany)]
        df["gram_duration"] = df.start.shift(-self.n) - df.start

        df = df[:howmany].reset_index(drop=True)
        df["ngram"] = ngrams

        self.df = df
        self.pitch_sequence = "".join(pitch_chars)

    def __rich_repr__(self):
        yield "NgramContainer"
        yield "N", self.n
        yield "df", self.df.shape
        yield "pitch", len(self.pitch_sequence)
        yield "duration", self.piece.duration

    @property
    def top_grams(self) -> list[str]:
        # Select sequence seeds with multiple occurances
        gram_counts = self.df.ngram.value_counts()
        ids = gram_counts >= 2
        top_grams = gram_counts[ids].index.tolist()
        return top_grams
