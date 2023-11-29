import numpy as np
import pandas as pd


def apply_sustain(
    df: pd.DataFrame,
    sustain: pd.DataFrame,
    sustain_threshold: int = 64,
) -> pd.DataFrame:
    """
    Apply sustain pedal effects to the notes in a DataFrame.

    This function uses a second DataFrame containing sustain pedal events to extend
    the duration of notes in the original DataFrame. It modifies the end times of notes
    that are held during the time when the sustain pedal is pressed down.

    Args:
        df (pd.DataFrame):
            The DataFrame containing musical note data. Expected to have columns
            'start', 'end', and 'pitch', where 'start' and 'end' represent the
            start and end times of the notes.
        sustain (pd.DataFrame):
            The DataFrame containing sustain pedal events. Expected to have columns
            'time' and 'value', where 'time' is the timestamp of the pedal event and
            'value' is the intensity of the pedal press.
        sustain_threshold (int, optional):
            The threshold value above which the sustain pedal is considered to be pressed
            down. Defaults to 64.

    Returns:
        df (pd.DataFrame):
            The modified DataFrame with updated end times for notes affected by the
            sustain pedal.

    Notes:
        - The sustain effect is applied by extending the end time of notes to either the
          start of the next note with the same pitch or the time when the sustain pedal is
          released, whichever comes first.
    """
    # Mark sustain pedal as down or up based on threshold value
    sustain["is_down"] = sustain.value >= sustain_threshold

    # Group sustain pedal events by continuous down or up states
    ids = sustain.is_down
    sustain["down_index"] = (ids != ids.shift(1)).cumsum()
    groups = sustain[sustain.is_down].groupby("down_index")

    # Iterate over each group of sustain pedal down events
    for _, gdf in groups:
        # Get start and end times for current sustain pedal down event
        pedal_down = gdf.time.min()
        pedal_up = gdf.time.max()

        # Select notes affected by current sustain pedal down event
        # ids = (df.end >= pedal_down) & (df.end < pedal_up)
        # affeced_notes = df[ids]

        # Modify end times of selected notes based on sustain pedal duration
        df = sustain_notes(
            df=df,
            pedal_down=pedal_down,
            pedal_up=pedal_up,
        )

    # Keep duration consistent
    df["duration"] = df.end - df.start

    return df


def sustain_notes(
    df: pd.DataFrame,
    pedal_down: float,
    pedal_up: float,
) -> pd.DataFrame:
    """
    Extend the end times of notes affected by a sustain pedal down event.

    This helper function is called by `apply_sustain` to process each group of sustain
    pedal down events. It extends the end times of notes that are playing during the
    sustain pedal down event.

    Args:
        df (pd.DataFrame):
            The DataFrame containing musical note data. Expected to have columns
            'start', 'end', and 'pitch'.
        pedal_down (float):
            The start time of the sustain pedal down event.
        pedal_up (float):
            The end time of the sustain pedal down event, indicating when the pedal is released.

    Returns:
        df (pd.DataFrame):
            The DataFrame with updated end times for the notes affected by the sustain pedal.
    """
    end_times = []

    # Select notes affected by current sustain pedal down event
    ids = (df.end >= pedal_down) & (df.end < pedal_up)
    affected_notes = df[ids]

    for it, row in affected_notes.iterrows():
        # Get the rows in the DataFrame that correspond to the same pitch
        # as the current row, and that start after the current row
        jds = (df.pitch == row.pitch) & (df.start > row.start)

        if jds.any():
            # If there are any such rows, set the end time
            # to be the start time of the next note of the same pitch
            end_time = min(df[jds].start.min(), pedal_up)
        else:
            # If there are no such rows, set the end time to be the end of the sustain
            # or to be the release of the note, whichever is later
            end_time = max(row.end, pedal_up)

        end_times.append(end_time)

    df.loc[ids, "end"] = end_times

    return df


def note_number_to_name(note_number):
    """
    Convert a MIDI note number to its name, in the format
    ``'(note)(accidental)(octave number)'`` (e.g. ``'C#4'``).

    Parameters:
        note_number (int):
            MIDI note number.  If not an int, it will be rounded.

    Returns:
        note_name (str):
            Name of the supplied MIDI note number.

    """

    # Note names within one octave
    semis = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    # Ensure the note is an int
    note_number = int(np.round(note_number))

    # Get the semitone and the octave, and concatenate to create the name
    return semis[note_number % 12] + str(note_number // 12 - 1)
