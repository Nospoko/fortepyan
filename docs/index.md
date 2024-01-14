# Welcome to Fortepyan

![GitHub CI](https://github.com/Nospoko/Fortepyan/actions/workflows/ci_tests.yaml/badge.svg?branch=master) [![Python 3.9](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads) [![PyPI version](https://img.shields.io/pypi/v/fortepyan.svg)](https://pypi.org/project/fortepyan/) [![PyPI download month](https://img.shields.io/pypi/dm/fortepyan.svg)](https://pypi.org/project/fortepyan/)

Fortepyan is a Python package designed for programmers who work with MIDI data. Its standout feature is the ability to visualize MIDI data in a pianoroll format, providing an intuitive and interactive way to work with musical data.

Creating new midi files is as simple as creating a pandas dataframe. All you need is a couple of lines of code to create a new midi file from scratch.
```python
import fortepyan as ff
import pandas as pd
import random

# Create a dataframe with random values
df = pd.DataFrame(
    {
    "start": [i for i in range(0, 100)],
    "duration": [1 for i in range(0, 100)],
    "pitch": [random.randint(40, 90) for i in range(0, 100)],
    "velocity": [random.randint(40, 90) for i in range(0, 100)],
    }
)

# Create a MidiPiece object
midi_piece = ff.MidiPiece(df)
```
You can also load existing midi files and work with them, visualize them, or even save a mp3/wav file with the audio of the midi file.

```python
# Load a midi file
midi_file_path = "path/to/midi/file.mid"
midi_piece = ff.MidiPiece.from_file(midi_file_path)

# Visualize the midi file as a pianoroll
ff.view.draw_pianoroll_with_velocities(midi_piece)

# Save the audio of the midi file as a mp3 file
ff.audio.midi_to_mp3(midi_piece, "path/to/mp3/file.mp3")
```

Installing Fortepyan is as simple as: `pip install fortepyan`
