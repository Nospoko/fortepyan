import pytest
import pandas as pd

from fortepyan.midi.structures import MidiPiece
from fortepyan.view.pianoroll.main import sanitize_midi_piece


@pytest.fixture
def midi_piece():
    path = "tests/resources/test_midi.mid"
    return MidiPiece.from_file(path)


@pytest.fixture
def midi_piece_long():
    path = "tests/resources/test_midi.mid"
    piece = MidiPiece.from_file(path)
    # Add a very long note to check that the trimming works
    df = pd.DataFrame(
        {
            "start": [0],
            "end": [1200],
            "duration": [1200],
            "pitch": [60],
            "velocity": [80],
        }
    )
    piece = piece + MidiPiece(df)
    return piece


def test_sanitize_midi_piece(midi_piece):
    sanitized_piece = sanitize_midi_piece(midi_piece)
    assert isinstance(sanitized_piece, MidiPiece)
    assert sanitized_piece.duration < 1200


def test_sanitize_midi_piece_long(midi_piece_long):
    sanitized_piece = sanitize_midi_piece(midi_piece_long)
    assert isinstance(sanitized_piece, MidiPiece)
    assert sanitized_piece.duration < 1200
