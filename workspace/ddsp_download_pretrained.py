import os

# Pretrained models.
PRETRAINED_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "ddsp_pretrained"
)

# Copy over from gs:// for faster loading.
os.system(f"rm -rf {PRETRAINED_DIR}")
os.system(f"mkdir {PRETRAINED_DIR}")

GCS_CKPT_DIR = "gs://ddsp/models/timbre_transfer_colab/2021-07-08"

for model in {"Violin", "Flute", "Flute2", "Trumpet", "Tenor_Saxophone"}:
    model_dir = os.path.join(GCS_CKPT_DIR, f"solo_{model.lower()}_ckpt")
    save_dir = os.path.join(PRETRAINED_DIR, f"solo_{model.lower()}")
    os.system(f"mkdir {save_dir}")
    os.system(f"gsutil cp {model_dir}/* {save_dir}")

print("Done!")
