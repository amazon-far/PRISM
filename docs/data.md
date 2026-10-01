# Dataset preparation

Download [Amazon-FAR/far-prism-data](https://huggingface.co/datasets/Amazon-FAR/far-prism-data)
from Hugging Face. The current release contains **129 successful teacher-rollout
trajectories** (35 box, 32 bin, 34 barrel, 28 ball), precomputed policy commands,
matching contact sidecars, G1 assets and reference-replay videos. This is a
filtered subset of the original 137 trajectories; the raw-video, reconstruction
and teacher-training sections are still under review.

Run the data script from the repository root (requires Python 3.11, `curl` and
standard Linux utilities). It downloads the single **317 MB** archive, checks
its SHA256, extracts and verifies every file, then checks the object assets and
prepares eight training shards:

```bash
bash download_data.sh
```

The script pins dataset revision `2498b1dbfda63a8000d28aa1b5d504cf9ff6b21b`.
Pass an output directory as the first argument to use another location;
rerunning reuses the verified archive and checks the existing dataset and
shards. No separate shard-preparation command is needed. With a custom output
directory, pass the corresponding paths to `train_student.sh` (see `--help`).

The extracted training inputs are:

```text
data/far-prism-data/
  clips.csv                                # Per-clip index
  data/train-student/data/motion_bank/      # Motion NPZs, commands and object map
  data/train-student/data/contact_sidecars/ # Matching contact intervals and points
  data/train-student/data/robot_assets/     # G1 URDF and depth-rendering meshes
  data/train-student/vis/                   # Reference replays
```

**Current release gap:** this dataset revision omits the object visual/collision
meshes referenced by its URDFs under `data/recon-hoi/`. The public package must
include those assets before it can support training. Until then, the script
exits with an error at the asset check and does not report successful preparation.

With the complete assets, the script generates `data/student_shards_ws8/`
covering all 129 published clips. After it succeeds, follow the
[student distillation instructions](training.md#student) to use them.
The dataset and generated shards remain local under Git-ignored `data/`.
