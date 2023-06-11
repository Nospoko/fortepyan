# fortepyan :musical_keyboard:

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
# use patch/minor/major
bumpver update --patch

python -m build
twine upload -r pypi dist/*
```
