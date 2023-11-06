import pytest
import pandas as pd

from fortepyan.midi.structures import MidiPiece


# Define a single comprehensive fixture
@pytest.fixture
def df_full():
    return pd.DataFrame(
        {
            "pitch": [60, 62],
            "start": [0, 1],
            "end": [1, 2],
            "duration": [1, 1],
            "velocity": [100, 100],
        }
    )


# Tests using the fixture and modifying it according to the test's need
def test_with_start_end_duration(df_full):
    piece = MidiPiece(df=df_full)
    assert piece.df.shape[0] == 2


def test_with_start_end(df_full):
    df_mod = df_full.drop(columns=["duration"])
    piece = MidiPiece(df=df_mod)
    assert "duration" in piece.df.columns


def test_with_start_duration(df_full):
    df_mod = df_full.drop(columns=["end"])
    piece = MidiPiece(df=df_mod)
    assert "end" in piece.df.columns


def test_with_end_duration(df_full):
    df_mod = df_full.drop(columns=["start"])
    piece = MidiPiece(df=df_mod)
    assert "start" in piece.df.columns


def test_missing_velocity(df_full):
    df_mod = df_full.drop(columns=["velocity"])
    with pytest.raises(ValueError):
        MidiPiece(df=df_mod)


def test_missing_pitch(df_full):
    df_mod = df_full.drop(columns=["pitch"])
    with pytest.raises(ValueError):
        MidiPiece(df=df_mod)
