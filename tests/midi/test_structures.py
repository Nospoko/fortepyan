import pytest
import pandas as pd

from fortepyan.midi.structures import MidiPiece


# Define a single comprehensive fixture
@pytest.fixture
def sample_df():
    df = pd.DataFrame(
        {
            "start": [0.0, 1, 2, 3, 4],
            "end": [1, 2, 3, 4, 5.5],
            "duration": [1, 1, 1, 1, 1.5],
            "pitch": [60, 62, 64, 65, 67],
            "velocity": [80, 80, 80, 80, 80],
        }
    )
    return df


@pytest.fixture
def sample_midi_piece():
    df = pd.DataFrame(
        {
            "start": [0, 1, 2, 3, 4],
            "end": [1, 2, 3, 4, 5.5],
            "duration": [1, 1, 1, 1, 1.5],
            "pitch": [60, 62, 64, 65, 67],
            "velocity": [80, 80, 80, 80, 80],
        }
    )
    return MidiPiece(df)


def test_with_start_end_duration(sample_df):
    piece = MidiPiece(df=sample_df)
    assert piece.df.shape[0] == 5


def test_with_start_end(sample_df):
    df_mod = sample_df.drop(columns=["duration"])
    piece = MidiPiece(df=df_mod)
    assert "duration" in piece.df.columns


def test_with_start_duration(sample_df):
    df_mod = sample_df.drop(columns=["end"])
    piece = MidiPiece(df=df_mod)
    assert "end" in piece.df.columns


def test_with_end_duration(sample_df):
    df_mod = sample_df.drop(columns=["start"])
    piece = MidiPiece(df=df_mod)
    assert "start" in piece.df.columns


def test_missing_velocity(sample_df):
    df_mod = sample_df.drop(columns=["velocity"])
    with pytest.raises(ValueError):
        MidiPiece(df=df_mod)


def test_missing_pitch(sample_df):
    df_mod = sample_df.drop(columns=["pitch"])
    with pytest.raises(ValueError):
        MidiPiece(df=df_mod)


def test_midi_piece_duration_calculation(sample_df):
    piece = MidiPiece(df=sample_df)
    assert piece.duration == 5.5


def test_trim_within_bounds(sample_midi_piece):
    # Test currently works as in the original code.
    # We might want to change this behavior so that
    # we do not treat the trimed piece as a new piece
    trimmed_piece = sample_midi_piece.trim(2, 3)
    assert len(trimmed_piece.df) == 2, "Trimmed MidiPiece should contain 2 notes."
    assert trimmed_piece.df["start"].iloc[0] == 0, "New first note should start at 0 seconds."
    assert trimmed_piece.df["pitch"].iloc[0] == 64, "New first note should have pitch 64."
    assert trimmed_piece.df["end"].iloc[-1] == 2, "New last note should end at 2 seconds."


def test_trim_at_boundaries(sample_midi_piece):
    trimmed_piece = sample_midi_piece.trim(0, 5.5)
    assert trimmed_piece.size == sample_midi_piece.size, "Trimming at boundaries should not change the size."


def test_trim_out_of_bounds(sample_midi_piece):
    with pytest.raises(IndexError):
        _ = sample_midi_piece.trim(5.5, 8)  # Out of bounds, should raise an error


def test_trim_with_invalid_range(sample_midi_piece):
    # Assuming the behavior is to raise an error with invalid range
    with pytest.raises(IndexError):
        _ = sample_midi_piece.trim(4, 2)  # Invalid range, start is greater than finish


def test_source_update_after_trimming(sample_midi_piece):
    trimmed_piece = sample_midi_piece.trim(1, 3)
    assert trimmed_piece.source["start_time"] == 1, "Source start_time should be updated to reflect trimming."


def test_to_midi(sample_midi_piece):
    # Create the MIDI track
    midi_track = sample_midi_piece.to_midi()
    # Set the expected end time according to the sample MIDI piece
    expected_end_time = 5.5
    # Get the end time of the MIDI track
    midi_end_time = midi_track.get_end_time()

    assert midi_end_time == expected_end_time, f"MIDI end time {midi_end_time} does not match expected {expected_end_time}"


def test_add_two_midi_pieces(sample_midi_piece):
    # Create a second MidiPiece to add to the sample one
    df2 = pd.DataFrame(
        {
            "start": [0, 1, 2],
            "end": [1, 2, 3],
            "duration": [1, 1, 1],
            "pitch": [70, 72, 74],
            "velocity": [80, 80, 80],
        }
    )
    midi_piece2 = MidiPiece(df=df2)

    # Add the two pieces together
    combined_piece = sample_midi_piece + midi_piece2

    # Check that the resulting piece has the correct number of notes
    assert len(combined_piece) == len(sample_midi_piece) + len(midi_piece2)

    # Check if duration has been adjusted
    assert combined_piece.duration == sample_midi_piece.duration + midi_piece2.duration


def test_add_non_midi_piece(sample_midi_piece):
    # Try to add a non-MidiPiece object to a MidiPiece
    with pytest.raises(TypeError):
        _ = sample_midi_piece + "not a MidiPiece"


def test_add_does_not_modify_originals(sample_midi_piece):
    # Create a second MidiPiece to add to the sample one
    df2 = pd.DataFrame(
        {
            "start": [0, 1, 2],
            "end": [1, 2, 3],
            "duration": [1, 1, 1],
            "pitch": [70, 72, 74],
            "velocity": [80, 80, 80],
        }
    )
    midi_piece2 = MidiPiece(df=df2)

    # Store the original dataframes for comparison
    original_df1 = sample_midi_piece.df.copy()
    original_df2 = midi_piece2.df.copy()

    # Add the two pieces together
    _ = sample_midi_piece + midi_piece2

    # Check that the original pieces have not been modified
    pd.testing.assert_frame_equal(sample_midi_piece.df, original_df1)
    pd.testing.assert_frame_equal(midi_piece2.df, original_df2)
