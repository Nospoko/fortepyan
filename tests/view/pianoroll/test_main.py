import pytest
import pandas as pd
from matplotlib import pyplot as plt

from fortepyan.midi.structures import MidiPiece
from fortepyan.view.pianoroll.main import sanitize_midi_piece, draw_pianoroll_with_velocities


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


def test_draw_pianoroll_with_velocities(midi_piece):
    fig = draw_pianoroll_with_velocities(midi_piece)
    assert isinstance(fig, plt.Figure)

    # Accessing the axes of the figure
    ax1, ax2 = fig.axes
    assert ax1.get_title() == ""
    xticks = ax1.get_xticks()
    assert len(xticks) == 13  # Number of ticks with default resolution in the test midi file

    # Verify label
    assert ax1.get_xlabel() == "Time [s]"

    yticks = ax2.get_yticks()
    assert len(yticks) == 4  # 0 50 100 150
    yticks = ax1.get_yticks()
    assert len(yticks) == 11


def test_draw_pianoroll_with_velocities_long(midi_piece_long):
    fig = draw_pianoroll_with_velocities(midi_piece_long)
    assert isinstance(fig, plt.Figure)

    # Accessing the axes of the figure
    ax1, ax2 = fig.axes
    assert ax1.get_title() == ""
    xticks = ax1.get_xticks()
    assert len(xticks) == 13  # Number of ticks with default resolution in the long test midi file

    # Verify label
    assert ax1.get_xlabel() == "Time [s]"

    yticks = ax2.get_yticks()
    assert len(yticks) == 4  # 0 50 100 150
    yticks = ax1.get_yticks()
    assert len(yticks) == 11
