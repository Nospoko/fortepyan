import tempfile
from typing import Union

from pydub import AudioSegment
from midi2audio import FluidSynth

from fortepyan.audio import soundfont
from fortepyan.midi import structures
from fortepyan.midi import containers as midi_containers


def midi_to_wav(midi: Union[structures.MidiFile, structures.MidiPiece], wavpath: str):
    """
    Converts a MIDI file to a WAV file.

    This function takes a MIDI object, either as a `ff.MidiFile` object or a `MidiPiece` object,
    and converts it to a WAV file. The conversion uses the FluidSynth synthesizer with a downloaded sound font.
    The function also adds a silent event at the end of the MIDI sequence to ensure that the final notes have time
    to ring out properly.

    Args:
        midi (Union[MidiFile, MidiPiece]):
            The MIDI file to convert. Can be either a `MidiFile` object or a `MidiPiece` object.
        wavpath (str):
            The path where the converted WAV file will be saved.

    Note:
        If a `MidiPiece` object is provided, it is first converted to a `MidiFile` object
        before proceeding with the WAV conversion.


    Examples:
        >>> some_midi = ff.MidiPiece(midi_df)
        >>> midi_to_wav(some_midi, "test.wav")
    """
    if isinstance(midi, structures.MidiPiece):
        midi = midi.to_midi()
    # This will be deleted
    tmp_midi_path = tempfile.mkstemp(suffix=".mid")[1]

    # Add an silent event to make sure the final notes
    # have time to ring out
    end_time = midi.get_end_time() + 0.2
    pedal_off = midi_containers.ControlChange(64, 0, end_time)
    midi.instruments[0].control_changes.append(pedal_off)

    midi.write(tmp_midi_path)

    sound_font_path = soundfont.download_if_needed()
    synth = FluidSynth(sound_font=sound_font_path)
    synth.midi_to_audio(tmp_midi_path, wavpath)


def midi_to_mp3(midi: Union[structures.MidiFile, structures.MidiPiece], mp3_path: str = None):
    """
    Converts a MIDI file to an MP3 file.

    This function takes a MIDI object, either as a `MidiFile` object or a `MidiPiece` object,
    and first converts it to a WAV file. It then converts this WAV file to an MP3 file.

    Args:
        midi (Union[MidiFile, MidiPiece]):
            The MIDI file to convert. Can be either a `ff.MidiFile` object or a `MidiPiece` object.
        mp3_path (str, optional):
            The path where the converted MP3 file will be saved. If not specified, a temporary file is created.

    Returns:
        mp3_path (str): The path to the created MP3 file.

    Note:
        If a `MidiPiece` object is provided, it is first converted to a `MidiFile` object
        before proceeding with the WAV and then MP3 conversion.


    Examples:
        >>> some_midi = ff.MidiPiece(midi_df)
        >>> mp3_path = midi_to_mp3(some_midi, "test.mp3")
        >>> print("MP3 file created at:", mp3_path)
    """
    if isinstance(midi, structures.MidiPiece):
        midi = midi.to_midi()

    # This will be deleted
    tmp_wav_path = tempfile.mkstemp(suffix=".wav")[1]
    midi_to_wav(midi=midi, wavpath=tmp_wav_path)

    # Wav to mp3
    if not mp3_path:
        mp3_path = tempfile.mkstemp(suffix=".mp3")[1]

    print("Rendering audio to file:", mp3_path)
    AudioSegment.from_wav(tmp_wav_path).export(mp3_path, format="mp3")

    return mp3_path
