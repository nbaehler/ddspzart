import glob

from omni_transcribe import transcribe, synth
from ddsp_timbre_transfer import timbre_transfer

import tensorflow as tf
gpus = tf.config.list_physical_devices(device_type = 'GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.config.set_visible_devices([], 'GPU')

filename = "1788"
midi = transcribe(f"./{filename}.wav")
filenames = glob.glob(f"./{filename}_*.mid")

for filename in filenames:
    synth(f"./{filename}")
    timbre_transfer(filename, "Violin")