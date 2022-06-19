import glob
import os
from pathlib import Path
import sys
from omnizart.cli import silence_tensorflow
from omnizart.constants.midi import SOUNDFONT_PATH
from omnizart.remote import download_large_file_from_google_drive

from omni_utils import ensure_path_exists, synth_midi

# from omnizart.music.app import MusicTranscription
from omni_music_app import MusicTranscription


def transcribe(
    input_audio,
    model_path="../omnizart/omnizart/checkpoints/music/music_note_stream/",
    output="./",
):
    """Transcribe a single audio and output as a MIDI file.

    This will output a MIDI file with the same name as the given audio, except the
    extension will be replaced with '.mid'.

    Supported modes are: Piano, Stream, Pop

    \b
    Example Usage
    $ omnizart music transcribe \
        # example.wav \
        --model-path path/to/model \
        --output example.mid
    """
    silence_tensorflow()
    app = MusicTranscription()

    return app.transcribe(input_audio, model_path, output=output)


def synth(input_midi, output_path="./", sf2_path=None):
    """Synthesize the MIDI into wav file.

    If --sf2-path is not specified, will use the default soundfont file same as used by MuseScore."
    """
    f_name, _ = os.path.splitext(os.path.basename(input_midi))
    out_name = f"{f_name}_synth.wav"
    if os.path.isdir(output_path):
        # Specifies only directory without file name.
        # Use the default file name.
        ensure_path_exists(output_path)
        output_file = os.path.join(output_path, out_name)
    else:
        # Already specified the output file name.
        f_dir = os.path.dirname(os.path.abspath(output_path))
        ensure_path_exists(f_dir)
        output_file = output_path
    print(f"Output file as: {output_file}")

    if sf2_path is None:
        if not os.path.exists(SOUNDFONT_PATH):
            # Download the default soundfont file.
            print("Downloading default sondfont file...")
            download_large_file_from_google_drive(
                url="16RM-dWKcNtjpBoo7DFSONpplPEg5ruvO",
                file_length=31277462,
                save_path=os.path.dirname(SOUNDFONT_PATH),
                save_name=os.path.basename(SOUNDFONT_PATH),
            )
        sf2_path = SOUNDFONT_PATH

    print("Synthesizing MIDI...")
    synth_midi(input_midi, output_path=output_file, sf2_path=sf2_path)
    print("Synthesize finished")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == []:
        filename = "test.wav"
        midis = transcribe(filename)
    else:
        midis = transcribe(args[0], args[1], args[2])

    midis = glob.glob(f"{filename}_*.mid")
    for midi in midis:
        synth(midi)
