import pytest
import pandas as pd

from fortepyan.midi.structures import MidiFile, MidiPiece

# constants
TEST_MIDI_PATH = "tests/resources/test_midi.mid"


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


def test_midi_file_merge():
    mfa = MidiFile(path=TEST_MIDI_PATH)
    mfb = MidiFile(path=TEST_MIDI_PATH)

    mf_merged = MidiFile.merge_files([mfa, mfb])

    assert len(mf_merged.notes) == len(mfa.notes) + len(mfb.notes)
    assert mf_merged.duration == mfa.duration + mfb.duration


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


def test_trim_within_bounds_with_shift(sample_midi_piece):
    # Test currently works as in the original code.
    # We might want to change this behavior so that
    # we do not treat the trimed piece as a new piece
    trimmed_piece = sample_midi_piece.trim(2, 3)
    assert len(trimmed_piece.df) == 2, "Trimmed MidiPiece should contain 2 notes."
    assert trimmed_piece.df["start"].iloc[0] == 0, "New first note should start at 0 seconds."
    assert trimmed_piece.df["pitch"].iloc[0] == 64, "New first note should have pitch 64."
    assert trimmed_piece.df["end"].iloc[-1] == 2, "New last note should end at 2 seconds."


def test_trim_index_slice_type(sample_midi_piece):
    trimmed_piece = sample_midi_piece.trim(1, 3, slice_type="index")
    assert len(trimmed_piece) == 3, "Trimmed MidiPiece should contain 3 notes."
    assert trimmed_piece.df["start"].iloc[0] == 0, "New first note should start at 0 seconds."
    assert trimmed_piece.df["pitch"].iloc[0] == 62, "New first note should have pitch 62."
    assert trimmed_piece.df["end"].iloc[-1] == 3, "New last note should end at 3 seconds."


def test_trim_by_end_slice_type(sample_midi_piece):
    trimmed_piece = sample_midi_piece.trim(1, 5, slice_type="by_end")
    assert len(trimmed_piece.df) == 3, "Trimmed MidiPiece should contain 3 notes."
    assert trimmed_piece.df["start"].iloc[0] == 0, "New first note should start at 0 seconds."
    assert trimmed_piece.df["pitch"].iloc[0] == 62, "New first note should have pitch 62."
    assert trimmed_piece.df["end"].iloc[-1] == 3, "New last note should end at 2 seconds."
    assert trimmed_piece.df["pitch"].iloc[-1] == 65, "New last note should have pitch 65."


def test_trim_with_invalid_slice_type(sample_midi_piece):
    with pytest.raises(NotImplementedError):
        _ = sample_midi_piece.trim(1, 3, slice_type="invalid")  # Invalid slice type, should raise an error


def test_trim_within_bounds_no_shift(sample_midi_piece):
    # This test should not shift the start times
    trimmed_piece = sample_midi_piece.trim(2, 3, shift_time=False)
    assert len(trimmed_piece.df) == 2, "Trimmed MidiPiece should contain 2 notes."
    # Since we're not shifting, the start should not be 0 but the actual start time
    assert trimmed_piece.df["start"].iloc[0] == 2, "First note should retain its original start time."
    assert trimmed_piece.df["pitch"].iloc[0] == 64, "First note should have pitch 64."
    assert trimmed_piece.df["end"].iloc[-1] == 4, "Last note should end at 4 seconds."


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
    midi_file = sample_midi_piece.to_midi()
    # Set the expected end time according to the sample MIDI piece
    expected_end_time = sample_midi_piece.duration
    # Get the end time of the MIDI track
    midi_end_time = midi_file.duration

    assert midi_end_time == expected_end_time, f"MIDI end time {midi_end_time} does not match expected {expected_end_time}"
    assert midi_file.df.shape == sample_midi_piece.df.shape


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


# === Tests for MidiFile ===
# TODO: fill tests with assertions based on test_midi.mid


def test_midi_file_initialization():
    """
    Test the initialization of the MidiFile class.
    """
    midi_file = MidiFile(path=TEST_MIDI_PATH)

    assert midi_file.path == TEST_MIDI_PATH
    assert midi_file.apply_sustain is True
    assert midi_file.sustain_threshold == 62


def test_midi_file_duration_property():
    """
    Test the 'duration' property.
    """
    midi_file = MidiFile(path=TEST_MIDI_PATH)
    assert isinstance(midi_file.duration, float)


def test_midi_file_notes_property():
    """
    Test the 'notes' property.
    """
    midi_file = MidiFile(path=TEST_MIDI_PATH)
    notes = midi_file.notes
    assert isinstance(notes, list)


def test_midi_file_control_changes_property():
    """
    Test the 'control_changes' property.
    """
    midi_file = MidiFile(path=TEST_MIDI_PATH)
    ccs = midi_file.control_changes
    assert isinstance(ccs, list)


@pytest.mark.parametrize(
    "index, expected_type",
    [
        (slice(0, 10), MidiPiece),
        # Add more test cases
    ],
)
def test_midi_file_getitem(index, expected_type):
    """
    Test the '__getitem__' method.
    """
    midi_file = MidiFile(path=TEST_MIDI_PATH)
    result = midi_file[index]
    assert isinstance(result, expected_type)


def test_midi_file_duration():
    """
    Test the 'get_end_time' method.
    """
    midi_file = MidiFile(path=TEST_MIDI_PATH)
    end_time = midi_file.duration
    assert isinstance(end_time, float)


# Add more tests for other methods and properties as needed
