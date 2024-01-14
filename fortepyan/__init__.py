import pretty_midi

from fortepyan import view
from fortepyan.midi.structures import MidiFile, MidiPiece

__all__ = ["view", "MidiFile", "MidiPiece"]
__version__ = "0.2.7"

# Pretty MIDI will throw an unwanted error for large files with high PPQ
# This is a workaround
# https://github.com/craffel/pretty-midi/issues/112
pretty_midi.pretty_midi.MAX_TICK = 1e10
