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
        ids = (df.start >= pedal_down) & (df.start < pedal_up)
        affeced_notes = df[ids]

        # Modify end times of selected notes based on sustain pedal duration
        modified_end_times = sustain_notes(affeced_notes, pedal_down, pedal_up)
        df.loc[ids, "end"] = modified_end_times

    return df


def sustain_notes(df: pd.DataFrame, pedal_down: float, pedal_up: float) -> list[float]:
    end_times = []
    for it, row in df.iterrows():
        # Get the rows in the DataFrame that correspond to the same pitch
        # as the current row, and that start after the current row
        ids = (df.pitch == row.pitch) & (df.start > row.start)

        if ids.any():
            # If there are any such rows, set the end time
            # to be the start time of the next note of the same pitch
            end_time = df[ids].start.min()
        else:
            # If there are no such rows, set the end time to be the end of the sustain
            end_time = pedal_up

        end_times.append(end_time)

    return end_times
