import pytest
import numpy as np
import pandas as pd

from fortepyan.midi.structures import MidiFile
from fortepyan.midi.tools import apply_sustain


@pytest.fixture
def expected_sustain_output():
    df = pd.read_csv("tests/resources/expected_sustain_output.csv")
    return df


@pytest.fixture
def testing_midi_file():
    return MidiFile("tests/resources/test_midi.mid", apply_sustain=False)


def test_apply_sustain(testing_midi_file, expected_sustain_output):
    applied_sustain = apply_sustain(
        df=testing_midi_file.raw_df,
        sustain=testing_midi_file.sustain,
        sustain_threshold=testing_midi_file.sustain_threshold,
    )

    # Flatten the dataframes or compare column-wise
    for column in expected_sustain_output.columns:
        assert np.all(np.isclose(applied_sustain[column].values, expected_sustain_output[column].values, atol=1e-10))
