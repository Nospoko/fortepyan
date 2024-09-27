import streamlit as st
from streamlit_pianoroll import from_fortepyan

from fortepyan import MidiFile


def main():
    st.write("# Test MidiFile merging")
    uploaded_files = st.file_uploader(
        label="Upload one or many MIDI files",
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.write("Waiting for files")
        return

    midi_files = []
    for uploaded_file in uploaded_files:
        midi_file = MidiFile.from_file(uploaded_file)
        midi_files.append(midi_file)

    merge_spacing = st.number_input(
        label="merge spacing [s] (time interval inserted between files)",
        min_value=0.0,
        max_value=30.0,
        value=5.0,
    )
    merged_midi_file = MidiFile.merge_files(
        midi_files=midi_files,
        space=merge_spacing,
    )
    st.write("Duration after merge:", merged_midi_file.duration)
    st.write("Number of notes after merge:", merged_midi_file.piece.size)

    from_fortepyan(merged_midi_file.piece)


if __name__ == "__main__":
    main()
