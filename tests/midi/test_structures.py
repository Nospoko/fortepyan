import pytest
import pandas as pd

from fortepyan.midi.structures import MidiPiece


# Define a single comprehensive fixture
@pytest.fixture
def sample_df():
    df = pd.DataFrame(
        {
            "start": [0, 1, 2, 3, 4],
            "end": [1, 2, 3, 4, 5],
            "duration": [1, 1, 1, 1, 1],
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
            "end": [1, 2, 3, 4, 5],
            "duration": [1, 1, 1, 1, 1],
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
    assert piece.duration == 5


def test_trim_within_bounds(sample_midi_piece):
    # Test currently works as in the original code.
    # We might want to change this behavior so that
    # we do not treat the trimed piece as a new piece
    trimmed_piece = sample_midi_piece.trim(2, 3)
    assert len(trimmed_piece.df) == 2, "Trimmed MidiPiece should contain 2 notes."
    assert trimmed_piece.df["start"].iloc[0] == 0, "New first note should start at 0 seconds."
    assert trimmed_piece.df["end"].iloc[-1] == 2, "New last note should end at 2 seconds."


def test_trim_at_boundaries(sample_midi_piece):
    trimmed_piece = sample_midi_piece.trim(0, 5)
    assert trimmed_piece.size == sample_midi_piece.size, "Trimming at boundaries should not change the size."


def test_trim_out_of_bounds(sample_midi_piece):
    # Assuming the behavior is to return an empty MidiPiece or raise an error
    with pytest.raises(IndexError):
        _ = sample_midi_piece.trim(6, 7)  # Out of bounds, should raise an error


def test_trim_with_invalid_range(sample_midi_piece):
    # Assuming the behavior is to raise an error with invalid range
    with pytest.raises(IndexError):
        _ = sample_midi_piece.trim(4, 2)  # Invalid range, start is greater than finish


def test_source_update_after_trimming(sample_midi_piece):
    trimmed_piece = sample_midi_piece.trim(1, 3)
    assert trimmed_piece.source["start_time"] == 1, "Source start_time should be updated to reflect trimming."
