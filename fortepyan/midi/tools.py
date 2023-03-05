import pandas as pd


def apply_sustain(
    df: pd.DataFrame,
    sustain: pd.DataFrame,
    sustain_threshold: int = 64,
) -> pd.DataFrame:
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
        # df.loc[ids, "end"] = modified_end_times

    return df


def sustain_notes(
    df: pd.DataFrame,
    pedal_down: float,
    pedal_up: float,
) -> pd.DataFrame:
    end_times = []

    # Select notes affected by current sustain pedal down event
    ids = (df.end >= pedal_down) & (df.end < pedal_up)
    affeced_notes = df[ids]

    for it, row in affeced_notes.iterrows():
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
