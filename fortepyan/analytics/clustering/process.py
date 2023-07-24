import Levenshtein
import numpy as np
import pandas as pd

from fortepyan import MidiPiece

from .structures import NgramContainer


def run(piece: MidiPiece, n: int = 16) -> pd.DataFrame:
    ngrams = NgramContainer(piece=piece, n=n)

    # If a sequence starts at *idx*, it's very likely that we'll get
    # another set of sequences starting at *idx + m* (with small m)
    # This step tries to filter out those duplicates
    duplication_threshold = 2 / 3 * n
    selected_grams = remove_duplicate_ngrams(
        ngrams=ngrams.top_grams,
        threshold=duplication_threshold,
    )

    # Each ngram seed is used to find longer sequences that are
    # simillar within some range of some metric, see *calculate_group_shifts*
    variants = []
    for gram in selected_grams:
        variant = get_gram_variants(
            gram=gram,
            ngrams=ngrams,
        )

        # We only care for sequences that were played more than once
        # Singular occurances should not get there, but apparently
        # the above logic is not air tight :see_no_evil:
        if variant["n_variants"] > 1:
            variants.append(variant)

    df = pd.DataFrame(variants)
    if not df.empty:
        df = remove_duplicate_variants(df, n)
        df = df.sort_values("n_variants", ascending=False, ignore_index=True)

    return df


def remove_duplicate_variants(df: pd.DataFrame, n: int) -> pd.DataFrame:
    # Initialize an empty list to hold unique variants
    unique_variants = []

    for it, row in df.iterrows():
        # Get the variant indices of the current row
        idxs = row.idxs

        # Assume the current row is not similar to any existing unique variants
        is_similar = False

        # Iterate over each existing unique variant
        for variant in unique_variants:
            # Calculate the absolute differences between the current variant and the existing variant
            diffs = np.abs(idxs[:, np.newaxis] - variant.idxs).flatten()

            # Check which differences are less than the threshold n
            ids = diffs < n

            # If all the relevant differences are less than n, the two variants are considered similar
            if ids.sum() == row.n_variants:
                # Mark the current row as similar to the existing variant and break out of the loop
                is_similar = True
                break

        # If the current row is not similar to any existing unique variants, add it to the list
        if not is_similar:
            unique_variants.append(row)

    df = pd.DataFrame(unique_variants).reset_index(drop=True)

    return df


def remove_duplicate_ngrams(ngrams: list[str], threshold: int) -> list[str]:
    unique_ngrams = []

    # Iterate through each n-gram in the input list
    for ngram in ngrams:
        # Check if the current n-gram is similar to any selected n-gram
        # by comparing their Levenshtein distances
        is_similar = any([Levenshtein.distance(ngram, existing_ngram) <= threshold for existing_ngram in unique_ngrams])

        # If the current n-gram is not similar to any existing unique n-grams,
        # add it to the unique_ngrams list
        if not is_similar:
            unique_ngrams.append(ngram)

    return unique_ngrams


def get_gram_variants(
    gram: str,
    ngrams: NgramContainer,
) -> dict:
    idxs = np.where(ngrams.df.ngram == gram)[0]
    idxs = filter_overlaping_sequences(idxs, ngrams.n)

    # Fuzzy-wuzzy extension of the seeds, *threshold* is a measure
    # of deviation between two sequences that are being compared
    # "if the sequence is extended further, next *threshold* notes
    # are going to be different between both sequences
    threshold = 4
    left_shifts, right_shifts = calculate_group_shifts(
        pitch_sequence=ngrams.pitch_sequence,
        idxs=idxs,
        threshold=threshold,
        n=ngrams.n,
    )

    # Use those thresholds to find groups of similar fragments
    # based on the same ngram seed
    variant = select_variant(
        idxs=idxs,
        left_shifts=left_shifts,
        right_shifts=right_shifts,
    )

    return variant.to_dict()


def filter_overlaping_sequences(idxs: list[int], n: int) -> list[int]:
    # Initialize a list to store the indexes to keep
    keep = []

    # Iterate over the indexes and check for overlaps
    for it, idx in enumerate(idxs):
        # Check if this index overlaps with any of the previous ones
        is_overlapping = any(idx - prev_idx < n for prev_idx in idxs[:it])

        # Only keep the index if it does not overlap with any previous sequence
        if not is_overlapping:
            keep.append(idx)

    keep = np.array(keep)
    return keep


def calculate_group_shifts(
    pitch_sequence: list[str],
    idxs: list[int],
    threshold: int,
    n: int,
) -> tuple[list[int], list[int]]:
    left_shifts, right_shifts = [], []
    for it in idxs:
        lefts, rights = [], []
        for jt in idxs:
            left, right = expand_sequences(pitch_sequence, it, jt, n, 100)

            left_shift = get_shift_limit(left, threshold)
            lefts.append(left_shift)

            right_shift = get_shift_limit(right, threshold)
            rights.append(right_shift)

        left_shifts.append(lefts)
        right_shifts.append(rights)

    left_shifts = np.array(left_shifts)
    right_shifts = np.array(right_shifts)
    return left_shifts, right_shifts


def expand_sequences(
    pitch_sequence: list[str],
    it: int,
    jt: int,
    n: int,
    distance: int = 40,
) -> tuple[list[int], list[int]]:
    left = pitch_sequence[it : it + n]
    right = pitch_sequence[jt : jt + n]

    left_scores = []
    for shift in range(distance):
        left = pitch_sequence[it - shift : it + n]
        right = pitch_sequence[jt - shift : jt + n]
        d = Levenshtein.distance(left, right)
        left_scores.append(d)

    righ_scores = []
    for shift in range(distance):
        left = pitch_sequence[it : it + n + shift]
        right = pitch_sequence[jt : jt + n + shift]
        d = Levenshtein.distance(left, right)
        righ_scores.append(d)

    return left_scores, righ_scores


def select_variant(
    idxs: list[int],
    left_shifts: np.array,
    right_shifts: np.array,
) -> dict:
    # For every left/right expansion combination ...
    scores = []
    for it in range(10):
        for jt in range(10):
            left_shift = jt * 5
            right_shift = it * 5
            ids = right_shifts >= right_shift
            jds = left_shifts >= left_shift

            # ... find the one that includes most versions of this fragment
            kds = ids & jds
            top_row = kds.sum(1).argmax()
            score = {
                "left_shift": left_shift,
                "right_shift": right_shift,
                "row": top_row,
                "n_variants": kds[top_row].sum(),
                "expansion": left_shift + right_shift,
                "idxs": idxs[kds[top_row]],
            }
            scores.append(score)

    score = pd.DataFrame(scores)

    # Set the target length to be half the length of idxs, rounded up to the nearest integer,
    # but ensure that it is at least 3 and no more than the length of idxs
    target = min(max(len(idxs) * 0.5, 3), len(idxs))

    # Find the variant of this fragment with the best expansion
    vds = score.n_variants >= target
    idx = score[vds].expansion.argmax()
    selected = score[vds].iloc[idx]

    return selected


def get_shift_limit(shifts: list[int], threshold: int = 5) -> int:
    kernel = np.ones(threshold)
    convolved = np.convolve(np.diff(shifts), kernel, mode="valid")

    hits = convolved == threshold
    if hits.any():
        acceptable_shift = np.where(hits)[0][0]
    else:
        acceptable_shift = len(shifts)

    return acceptable_shift
