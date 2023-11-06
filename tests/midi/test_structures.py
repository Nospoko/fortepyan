import unittest

import pandas as pd

from fortepyan.midi.structures import MidiPiece


class TestMidiPiece(unittest.TestCase):
    def test_with_start_end_duration(self):
        # All three columns provided
        df = pd.DataFrame({"pitch": [60, 62], "start": [0, 1], "end": [1, 2], "duration": [1, 1], "velocity": [100, 100]})
        piece = MidiPiece(df=df)
        self.assertEqual(piece.df.shape[0], 2)

    def test_with_start_end(self):
        # Only start and end provided
        df = pd.DataFrame({"pitch": [60, 62], "start": [0, 1], "end": [1, 2], "velocity": [100, 100]})
        piece = MidiPiece(df=df)
        self.assertTrue("duration" in piece.df.columns)

    def test_with_start_duration(self):
        # Only start and duration provided
        df = pd.DataFrame({"pitch": [60, 62], "start": [0, 1], "duration": [1, 1], "velocity": [100, 100]})
        piece = MidiPiece(df=df)
        self.assertTrue("end" in piece.df.columns)

    def test_with_end_duration(self):
        # Only end and duration provided
        df = pd.DataFrame({"pitch": [60, 62], "end": [1, 2], "duration": [1, 1], "velocity": [100, 100]})
        piece = MidiPiece(df=df)
        self.assertTrue("start" in piece.df.columns)

    def test_missing_velocity(self):
        # Missing velocity
        df = pd.DataFrame({"pitch": [60, 62], "start": [0, 1], "end": [1, 2], "duration": [1, 1]})
        with self.assertRaises(ValueError):
            MidiPiece(df=df)

    def test_missing_pitch(self):
        # Missing pitch
        df = pd.DataFrame({"start": [0, 1], "end": [1, 2], "duration": [1, 1], "velocity": [100, 100]})
        with self.assertRaises(ValueError):
            MidiPiece(df=df)


if __name__ == "__main__":
    unittest.main()
