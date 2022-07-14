# Copyright 2021 Google LLC. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Ignore a bunch of deprecation warnings
from tensorflow.python.ops.numpy_ops import np_config
from scipy.io.wavfile import write
import tensorflow.compat.v2 as tf
import pickle
import numpy as np
import matplotlib.pyplot as plt
import librosa
import gin
from ddsp.training.postprocessing import detect_notes, fit_quantile_transform
from ddsp_colab_utils import (
    auto_tune,
    get_tuning_factor,
    specplot,
    DEFAULT_SAMPLE_RATE,
    audio_bytes_to_np,
)
import ddsp.training
import ddsp
import time
import os
import sys
import warnings

warnings.filterwarnings("ignore")


# gpus = tf.config.list_physical_devices(device_type = 'GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)
# tf.config.set_visible_devices([], 'GPU')


np_config.enable_numpy_behavior()


def timbre_transfer(in_file, model):
    # Helper Functions
    sample_rate = DEFAULT_SAMPLE_RATE
    normalize_db = None

    workspace = os.path.dirname(os.path.abspath(__file__))

    input_file = os.path.join(workspace, in_file)

    with open(input_file, "rb") as wavfile:
        audio_bytes = wavfile.read()

    audio = audio_bytes_to_np(
        audio_bytes, sample_rate=sample_rate, normalize_db=normalize_db
    )

    if len(audio.shape) == 1:
        audio = audio[np.newaxis, :]
    print("\nExtracting audio features...")

    # # Plot
    # specplot(audio)
    # plt.title("Original")
    # plt.savefig(os.path.join(workspace, "original_spectrum.png"))

    # Setup the session.
    ddsp.spectral_ops.reset_crepe()

    # Compute features.
    start_time = time.time()
    audio_features = ddsp.training.metrics.compute_audio_features(audio)
    audio_features["loudness_db"] = audio_features["loudness_db"].astype(
        np.float32)
    audio_features_mod = None
    print("Audio features took %.1f seconds" % (time.time() - start_time))

    # TRIM = -15
    # # Plot Features.
    # fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(6, 8))
    # ax[0].plot(audio_features["loudness_db"][:TRIM])
    # ax[0].set_ylabel("loudness_db")

    # ax[1].plot(librosa.hz_to_midi(audio_features["f0_hz"][:TRIM]))
    # ax[1].set_ylabel("f0 [midi]")

    # ax[2].plot(audio_features["f0_confidence"][:TRIM])
    # ax[2].set_ylabel("f0 confidence")
    # _ = ax[2].set_xlabel("Time step [frame]")

    # fig.savefig(os.path.join(workspace, "features.png"))

    # @param ['Violin', 'Flute', 'Flute2', 'Trumpet', 'Tenor_Saxophone']
    if model in {"Violin", "Flute", "Flute2", "Trumpet", "Tenor_Saxophone"}:
        # Pretrained models.
        PRETRAINED_DIR = os.path.join(
            workspace,
            "ddsp_pretrained",
        )

        # Copy over from gs:// for faster loading.
        if not os.path.exists(PRETRAINED_DIR):
            # os.system(f"rm -rf {PRETRAINED_DIR}")
            os.system(f"mkdir {PRETRAINED_DIR}")

        GCS_CKPT_DIR = "gs://ddsp/models/timbre_transfer_colab/2021-07-08"
        model_dir = os.path.join(GCS_CKPT_DIR, f"solo_{model.lower()}_ckpt")

        PRETRAINED_DIR = os.path.join(PRETRAINED_DIR, f"solo_{model.lower()}")
        if not os.path.exists(PRETRAINED_DIR):
            os.system(f"mkdir {PRETRAINED_DIR}")
            os.system(f"gsutil cp {model_dir}/* {PRETRAINED_DIR}")
        model_dir = PRETRAINED_DIR
    else:
        # User models.
        USER_DIR = os.path.join(
            workspace,
            "ddsp_user",
        )

        model_dir = os.path.join(USER_DIR, f"{model.lower()}")

    gin_file = os.path.join(model_dir, "operative_config-0.gin")

    # Load the dataset statistics.
    DATASET_STATS = None
    dataset_stats_file = os.path.join(model_dir, "dataset_statistics.pkl")
    print(f"Loading dataset statistics from {dataset_stats_file}")
    try:
        if tf.io.gfile.exists(dataset_stats_file):
            with tf.io.gfile.GFile(dataset_stats_file, "rb") as f:
                DATASET_STATS = pickle.load(f)
    except Exception as err:
        print(f"Loading dataset statistics from pickle failed: {err}.")

    # Parse gin config,
    with gin.unlock_config():
        gin.parse_config_file(gin_file, skip_unknown=True)

    # Assumes only one checkpoint in the folder, 'ckpt-[iter]`.
    ckpt_files = [f for f in tf.io.gfile.listdir(model_dir) if "ckpt" in f]
    ckpt_name = ckpt_files[0].split(".")[0]
    ckpt = os.path.join(model_dir, ckpt_name)

    # Ensure dimensions and sampling rates are equal
    time_steps_train = gin.query_parameter("F0LoudnessPreprocessor.time_steps")
    n_samples_train = gin.query_parameter("Harmonic.n_samples")
    hop_size = int(n_samples_train / time_steps_train)

    time_steps = int(audio.shape[1] / hop_size)
    n_samples = time_steps * hop_size

    print("===Trained model===")
    print("Time Steps", time_steps_train)
    print("Samples", n_samples_train)
    print("Hop Size", hop_size)
    print("\n===Resynthesis===")
    print("Time Steps", time_steps)
    print("Samples", n_samples)
    print("")

    gin_params = [
        f"Harmonic.n_samples = {n_samples}",
        f"FilteredNoise.n_samples = {n_samples}",
        f"F0LoudnessPreprocessor.time_steps = {time_steps}",
        "oscillator_bank.use_angular_cumsum = True",
    ]

    with gin.unlock_config():
        gin.parse_config(gin_params)

    # Trim all input vectors to correct lengths
    for key in ["f0_hz", "f0_confidence", "loudness_db"]:
        audio_features[key] = audio_features[key][:time_steps]
    audio_features["audio"] = audio_features["audio"][:, :n_samples]

    # Set up the model just to predict audio given new conditioning
    model = ddsp.training.models.Autoencoder()
    model.restore(ckpt)

    # Build model by running a batch through it.
    start_time = time.time()
    _ = model(audio_features, training=False)
    print("Restoring model took %.1f seconds" % (time.time() - start_time))

    # Note Detection
    # You can leave this at 1.0 for most cases
    threshold = 1  # @param {type:"slider", min: 0.0, max:2.0, step:0.01}

    # Automatic
    ADJUST = True  # @param{type:"boolean"}

    # Quiet parts without notes detected (dB)
    quiet = 20  # @param {type:"slider", min: 0, max:60, step:1}

    # Force pitch to nearest note (amount)
    autotune = 0  # @param {type:"slider", min: 0.0, max:1.0, step:0.1}

    # Manual
    # Shift the pitch (octaves)
    pitch_shift = 0  # @param {type:"slider", min:-2, max:2, step:1}

    # Adjust the overall loudness (dB)
    loudness_shift = 0  # @param {type:"slider", min:-20, max:20, step:1}

    audio_features = {
        k: v.numpy() if tf.is_tensor(v) else v for k, v in audio_features.items()
    }

    audio_features_mod = {
        k: v.numpy() if tf.is_tensor(v) else v.copy() for k, v in audio_features.items()
    }

    # Helper functions.
    def shift_ld(audio_features, ld_shift=0.0):
        """Shift loudness by a number of octaves."""
        audio_features["loudness_db"] += ld_shift
        return audio_features

    def shift_f0(audio_features, pitch_shift=0.0):
        """Shift f0 by a number of octaves."""
        audio_features["f0_hz"] *= 2.0 ** (pitch_shift)
        audio_features["f0_hz"] = np.clip(
            audio_features["f0_hz"], 0.0, librosa.midi_to_hz(110.0)
        )
        return audio_features

    mask_on = None

    if ADJUST and DATASET_STATS is not None:
        # Detect sections that are "on".
        mask_on, note_on_value = detect_notes(
            audio_features["loudness_db"], audio_features["f0_confidence"], threshold
        )

        if np.any(mask_on):
            # Shift the pitch register.
            target_mean_pitch = DATASET_STATS["mean_pitch"]
            pitch = ddsp.core.hz_to_midi(audio_features["f0_hz"])
            mean_pitch = np.mean(pitch[mask_on])
            p_diff = target_mean_pitch - mean_pitch
            p_diff_octave = p_diff / 12.0
            round_fn = np.floor if p_diff_octave > 1.5 else np.ceil
            p_diff_octave = round_fn(p_diff_octave)
            audio_features_mod = shift_f0(audio_features_mod, p_diff_octave)

            # Quantile shift the note_on parts.
            _, loudness_norm = fit_quantile_transform(
                audio_features["loudness_db"],
                mask_on,
                inv_quantile=DATASET_STATS["quantile_transform"],
            )

            # Turn down the note_off parts.
            mask_off = np.logical_not(mask_on)
            loudness_norm[mask_off] -= quiet * (
                1.0 - note_on_value[mask_off][:, np.newaxis]
            )
            loudness_norm = np.reshape(
                loudness_norm, audio_features["loudness_db"].shape
            )

            audio_features_mod["loudness_db"] = loudness_norm

            # Auto-tune.
            if autotune:
                f0_midi = np.array(ddsp.core.hz_to_midi(
                    audio_features_mod["f0_hz"]))
                tuning_factor = get_tuning_factor(
                    f0_midi, audio_features_mod["f0_confidence"], mask_on
                )
                f0_midi_at = auto_tune(
                    f0_midi, tuning_factor, mask_on, amount=autotune)
                audio_features_mod["f0_hz"] = ddsp.core.midi_to_hz(f0_midi_at)

        else:
            print("\nSkipping auto-adjust (no notes detected or ADJUST box empty).")

    else:
        print(
            "\nSkipping auto-adjust (box not checked or no dataset statistics found)."
        )

    # Manual Shifts.
    audio_features_mod = shift_ld(audio_features_mod, loudness_shift)
    audio_features_mod = shift_f0(audio_features_mod, pitch_shift)

    # # Plot Features.
    # has_mask = int(mask_on is not None)
    # n_plots = 3 if has_mask else 2
    # fig, axes = plt.subplots(
    #     nrows=n_plots, ncols=1, sharex=True, figsize=(2 * n_plots, 8)
    # )

    # if has_mask:
    #     ax = axes[0]
    #     ax.plot(np.ones_like(mask_on[:TRIM]) * threshold, "k:")
    #     ax.plot(note_on_value[:TRIM])
    #     ax.plot(mask_on[:TRIM])
    #     ax.set_ylabel("Note-on Mask")
    #     ax.set_xlabel("Time step [frame]")
    #     ax.legend(["Threshold", "Likelihood", "Mask"])

    # ax = axes[0 + has_mask]
    # ax.plot(audio_features["loudness_db"][:TRIM])
    # ax.plot(audio_features_mod["loudness_db"][:TRIM])
    # ax.set_ylabel("loudness_db")
    # ax.legend(["Original", "Adjusted"])

    # ax = axes[1 + has_mask]
    # ax.plot(librosa.hz_to_midi(audio_features["f0_hz"][:TRIM]))
    # ax.plot(librosa.hz_to_midi(audio_features_mod["f0_hz"][:TRIM]))
    # ax.set_ylabel("f0 [midi]")
    # _ = ax.legend(["Original", "Adjusted"])

    # fig.savefig(os.path.join(workspace, "features_processed.png"))

    # Resynthesize Audio
    af = audio_features if audio_features_mod is None else audio_features_mod

    # Run a batch of predictions.
    start_time = time.time()
    outputs = model(af, training=False)
    audio_gen = model.get_audio_from_outputs(outputs)
    print("Prediction took %.1f seconds" % (time.time() - start_time))

    # specplot(audio_gen)
    # _ = plt.title("Resynthesis")
    # plt.savefig(os.path.join(workspace, "generated_spectrum.png"))

    audio_gen = audio_gen.numpy()[0]

    out_path = in_file.replace(".wav", "_generated.wav")

    write(os.path.join(workspace, out_path), DEFAULT_SAMPLE_RATE, audio_gen)

    return out_path


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == []:
        timbre_transfer("data/test.wav", "Violin")
    else:
        timbre_transfer(args[0], args[1])
