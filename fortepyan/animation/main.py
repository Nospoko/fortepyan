import subprocess

from fortepyan.audio.render import midi_to_mp3
from fortepyan.midi.structures import MidiPiece
from fortepyan.animation import pianoroll as pianoroll_animation


def make_piano_roll_video(
    piece: MidiPiece,
    movie_path: str,
    title: str = "animation",
    cmap: str = "PuBuGn",
):
    scene = pianoroll_animation.PianoRollScene(piece, title=title, cmap=cmap)
    mp3_path = midi_to_mp3(piece.to_midi())

    scene_frames_dir = scene.render_mp()
    command = f"""
        ffmpeg -y -f image2 -framerate 30 -i {str(scene_frames_dir)}/10%4d.png\
        -loglevel quiet -i {mp3_path} -map 0:v:0 -map 1:a:0\
        {movie_path}
    """
    print("Rendering a movie to file:", movie_path)
    subprocess.call(command, shell=True)
