# Fortepyan :musical_keyboard:
![GitHub CI](https://github.com/Nospoko/Fortepyan/actions/workflows/ci_tests.yaml/badge.svg?branch=master) [![Python 3.9](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads) [![PyPI version](https://img.shields.io/pypi/v/fortepyan.svg)](https://pypi.org/project/fortepyan/) [![PyPI download month](https://img.shields.io/pypi/dm/fortepyan.svg)](https://pypi.org/project/fortepyan/)

### Usage

```python
import fortepyan as ff

piece = ff.MidiPiece.from_file("mymidi.mid")

ff.view.draw_pianoroll_with_velocities(piece)
ff.view.make_piano_roll_video(piece, "tmp.mp4")
```

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
