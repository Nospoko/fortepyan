import streamlit as st
import fortepyan as ff
from fortepyan import MidiFile

from matplotlib import pyplot as plt

import streamlit_pianoroll


def main():
    st.title("Sustain pedal notes elongation review")

    uploaded_file = st.file_uploader(
        label="Upload one MIDI file",
    )

    if not uploaded_file:
        st.warning("Waiting for files")
        return

    midi_file = MidiFile.from_file(uploaded_file)
    streamlit_pianoroll.from_fortepyan(midi_file.piece)

    fig = ff.view.draw_pianoroll_with_velocities(midi_file.piece)
    st.pyplot(fig)

    c_df = midi_file.control_frame

    fig, ax = plt.subplots(figsize=[8, 3])

    ax.plot(c_df.time, c_df.value, "--o")

    st.pyplot(fig)


if __name__ == "__main__":
    main()
