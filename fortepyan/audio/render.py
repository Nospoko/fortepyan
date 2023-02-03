import os
import tempfile

import pretty_midi
from pydub import AudioSegment
from midi2audio import FluidSynth

synth = FluidSynth(sound_font="tmp/GUGSv1.471.sf2")


def midi_to_wav(midi: pretty_midi.PrettyMIDI, wavpath: str):
    tmp_midi_path = tempfile.mkstemp(suffix=".mid")[1]

    # Add an silent event to make sure the final notes
    # have time to ring out
    end_time = midi.get_end_time() + 0.2
    pedal_off = pretty_midi.ControlChange(64, 0, end_time)
    midi.instruments[0].control_changes.append(pedal_off)

    midi.write(tmp_midi_path)
    synth.midi_to_audio(tmp_midi_path, wavpath)

    os.remove(tmp_midi_path)


def midi_to_mp3(midi: pretty_midi.PrettyMIDI, mp3_path: str):
    tmp_wav_path = tempfile.mkstemp(suffix=".wav")[1]
    midi_to_wav(midi, tmp_wav_path)

    AudioSegment.from_wav(tmp_wav_path).export(mp3_path, format="mp3")

    os.remove(tmp_wav_path)
