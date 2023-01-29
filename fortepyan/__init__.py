import pretty_midi

from .viz import pianoroll as roll
from .midi.structures import MidiFile


__all__ = ['roll', 'MidiFile']

# Pretty MIDI will throw an unwanted error for large files with high PPQ
# This is a workaround
# https://github.com/craffel/pretty-midi/issues/112
pretty_midi.pretty_midi.MAX_TICK = 1e10
