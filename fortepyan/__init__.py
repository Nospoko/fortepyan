import pretty_midi

from .viz import pianoroll as roll
from .midi.structures import MidiFile, MidiPiece
from .audio.render import midi_to_mp3, midi_to_wav

__all__ = ["roll", "MidiFile", "MidiPiece", "midi_to_wav", "midi_to_mp3"]

# Pretty MIDI will throw an unwanted error for large files with high PPQ
# This is a workaround
# https://github.com/craffel/pretty-midi/issues/112
pretty_midi.pretty_midi.MAX_TICK = 1e10
