import tempfile

import pretty_midi
from pydub import AudioSegment
from midi2audio import FluidSynth

from fortepyan.audio import soundfont


def midi_to_wav(midi: pretty_midi.PrettyMIDI, wavpath: str):
    # This will be deleted
    tmp_midi_path = tempfile.mkstemp(suffix=".mid")[1]

    # Add an silent event to make sure the final notes
    # have time to ring out
    end_time = midi.get_end_time() + 0.2
    pedal_off = pretty_midi.ControlChange(64, 0, end_time)
    midi.instruments[0].control_changes.append(pedal_off)

    midi.write(tmp_midi_path)

    sound_font_path = soundfont.download_if_needed()
    synth = FluidSynth(sound_font=sound_font_path)
    synth.midi_to_audio(tmp_midi_path, wavpath)


def midi_to_mp3(midi: pretty_midi.PrettyMIDI, mp3_path: str = None):
    # This will be deleted
    tmp_wav_path = tempfile.mkstemp(suffix=".wav")[1]
    midi_to_wav(midi=midi, wavpath=tmp_wav_path)

    # Wav to mp3
    if not mp3_path:
        mp3_path = tempfile.mkstemp(suffix=".mp3")[1]
    print("Rendering audio to file:", mp3_path)
    AudioSegment.from_wav(tmp_wav_path).export(mp3_path, format="mp3")

    return mp3_path
