# Fortepyan :musical_keyboard:

![GitHub CI](https://github.com/Nospoko/Fortepyan/actions/workflows/ci_tests.yaml/badge.svg?branch=master) [![Python 3.9](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads) [![PyPI version](https://img.shields.io/pypi/v/fortepyan.svg)](https://pypi.org/project/fortepyan/) [![PyPI download month](https://img.shields.io/pypi/dm/fortepyan.svg)](https://pypi.org/project/fortepyan/)

**fortepyan** is a glorified pandas wrapper for midi files with piano music.

The main class to operate with is `MidiPiece`, which gives you access to the notes dataframe, and some useful utilities:

```python
from fortepyan import MidiPiece

piece = MidiPiece.from_file("foo.mid")

piece.df
#        pitch  velocity        start          end  duration
# 0         52        70     0.000000     0.538542  0.538542
# 1         52        62     0.660417     0.951562  0.291146
# 2         57        62    20.769792    21.207812  0.438021
# 3         67        65    62.172917    62.510937  0.338021
# 4         69        56    62.179167    62.232292  0.053125
```

Using with HuggingFace datasets:


```python
from datasets import load_datasets

dataset = load_dataset("epr-labs/maestro-sustain-v2", split="train")
piece = MidiPiece.from_huggingface(dataset[312])
piece.source
# {
#     'composer': 'Franz Liszt',
#     'title': 'Dante Sonata',
#     'split': 'train',
#     'year': 2009,
#     'midi_filename': '2009/MIDI-Unprocessed_11_R1_2009_06-09_ORIG_MID--AUDIO_11_R1_2009_11_R1_2009_09_WAV.midi',
#     'dataset': 'maestro'
# }
```

### Usage

```python
import fortepyan as ff

piece = ff.MidiPiece.from_file("mymidi.mid")

ff.view.draw_pianoroll_with_velocities(piece)
ff.view.make_piano_roll_video(piece, "tmp.mp4")
```

## Contributing

### Development

Pre-commit hooks with forced python formatting ([black](https://github.com/psf/black), [flake8](https://flake8.pycqa.org/en/latest/), and [isort](https://pycqa.github.io/isort/)):

```sh
pip install pre-commit
pre-commit install
```

Whenever you execute `git commit` the files altered / added within the commit will be checked and corrected. `black` and `isort` can modify files locally - if that happens you have to `git add` them again.
You might also be prompted to introduce some fixes manually.

To run the hooks against all files without running `git commit`:

```sh
pre-commit run --all-files
```

Package release:
```sh
# from the root directory with clean working tree
# replace patch with one of: [major, minor, patch]
./scripts/release/start_release.sh patch

# Make any additional changes to the release commit
./scripts/release/finish_release.sh
```

#### Tests

Recommended way to run and monitor tests is using pytest-watch:

```sh
ptw
```
