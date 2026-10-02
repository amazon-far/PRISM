# Dataset preparation

Download [Amazon-FAR/far-prism-data](https://huggingface.co/datasets/Amazon-FAR/far-prism-data)
from Hugging Face. The current release contains **129 successful teacher-rollout
trajectories** (35 box, 32 bin, 34 barrel, 28 ball), precomputed policy commands,
matching contact sidecars, object visual/collision meshes, G1 assets and
reference-replay videos. This is a filtered subset of the original 137
trajectories. Raw videos and the teacher-training motion bank remain under review.

Run the data script from the repository root (requires Python 3.11, `curl` and
standard Linux utilities). It downloads the single **495 MB** archive, checks
its SHA256, extracts and verifies every file, then checks the object assets and
prepares eight training shards:

```bash
bash download_data.sh
```

The script pins dataset revision `4240492ccc6bde73c3c4e3009ab3de4859bdbdf7`.
Pass an output directory as the first argument to use another location;
rerunning reuses the verified archive and checks the existing dataset and
shards. No separate shard-preparation command is needed. With a custom output
directory, pass the corresponding paths to `train_student.sh` (see `--help`).
If an older dataset is already present, choose a new output directory; the
script preserves existing data and refuses to mix revisions.

The extracted training inputs are:

```text
data/far-prism-data/
  clips.csv                                # Per-clip index
  data/train-student/data/motion_bank/      # Motion NPZs, commands and object map
  data/train-student/data/contact_sidecars/ # Matching contact intervals and points
  data/train-student/data/robot_assets/     # G1 URDF and depth-rendering meshes
  data/train-student/vis/                   # Reference replays
  data/recon-hoi/                           # Object URDFs, meshes and retargeted motions
```

The package includes 129 visual meshes simplified to 50,000 faces each and
2,580 convex collision meshes. Collision geometry and training URDF physics
match the original assets. Visual geometry differs from the original training
assets; depth-observation equivalence has not been validated. Original
visual meshes are also available in the Hugging Face dataset.

The script generates `data/student_shards_ws8/`
covering all 129 published clips. After it succeeds, follow the
[student distillation instructions](training.md#student) to use them.
The dataset and generated shards remain local under Git-ignored `data/`.
