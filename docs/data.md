# Dataset preparation

Download [Amazon-FAR/far-prism-data](https://huggingface.co/datasets/Amazon-FAR/far-prism-data)
from Hugging Face. The current release contains **129 successful teacher-rollout
trajectories** (35 box, 32 bin, 34 barrel, 28 ball), precomputed policy commands,
matching contact sidecars, G1 assets and reference-replay videos. This is a
filtered subset of the original 137 trajectories; the raw-video, reconstruction
and teacher-training sections are still under review.

Run the download script from the repository root (requires `curl` and standard
Linux utilities). It downloads the single **317 MB** archive, checks its SHA256,
extracts it and verifies every file:

```bash
bash download_data.sh
```

The script pins dataset revision `2498b1dbfda63a8000d28aa1b5d504cf9ff6b21b`.
Pass an output directory as the first argument to use another location;
rerunning reuses the verified archive and checks the existing dataset without
overwriting it. The commands below use the default repository `data/` directory.

The extracted training inputs are:

```text
data/far-prism-data/
  clips.csv                                # Per-clip index
  data/train-student/data/motion_bank/      # Motion NPZs, commands and object map
  data/train-student/data/contact_sidecars/ # Matching contact intervals and points
  data/train-student/data/robot_assets/     # G1 URDF and depth-rendering meshes
  data/train-student/vis/                   # Reference replays
```

**Current asset dependency:** this dataset revision does not include the object
visual/collision meshes referenced by its URDFs under `data/recon-hoi/`.
Those exact assets must be supplied before preparing shards or starting
distillation. The preparation command below checks these dependencies and stops
if any are missing; downloading the archive alone is not yet sufficient to train.

Once those assets are present, prepare eight rank shards locally. Keep the
extracted directory layout intact so relative object paths resolve correctly:

```bash
STUDENT_DATA="$PWD/data/far-prism-data/data/train-student/data"
STUDENT_SHARDS="$PWD/data/student_shards_ws8"
python scripts/prepare_as_rank_shards.py \
  --motion-dir "$STUDENT_DATA/motion_bank" \
  --object-map "$STUDENT_DATA/motion_bank/_clip_object_urdf_map.json" \
  --world-size 8 --environments-per-rank 2048 \
  --output-root "$STUDENT_SHARDS"
```

The shards cover all 129 published clips. Follow the
[student distillation instructions](training.md#student) to use them.
The dataset and generated shards remain local under Git-ignored `data/`.
