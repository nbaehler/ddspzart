#/bin/bash
deactivate
source deactivate
source ../.omni_venv/bin/activate
omnizart music transcribe 1788.wav
omnizart synth 1788.mid

source ../.ddsp_venv/bin/activate
python timbre_transfer.py 1788.wav Violin

# omnizart drum transcribe sample.wav
# omnizart chord transcribe sample.wav
# omnizart music transcribe sample.wav
# omnizart synth sample.mid

# # Vocals (singing voice) / accompaniment separation (2 stems)
# # Vocals / drums / bass / other separation (4 stems)
# # Vocals / drums / bass / piano / other separation (5 stems)
# spleeter separate -p spleeter:2stems -o output audio_example.mp3
