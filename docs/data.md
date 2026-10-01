# Dataset preparation

Download [Amazon-FAR/far-prism-data](https://huggingface.co/datasets/Amazon-FAR/far-prism-data)
from Hugging Face. The current release contains **129 successful teacher-rollout
trajectories** (35 box, 32 bin, 34 barrel, 28 ball), precomputed policy commands,
matching contact sidecars, G1 assets and reference-replay videos. This is a
filtered subset of the original 137 trajectories; the raw-video, reconstruction
and teacher-training sections are still under review.

After [installation](../README.md#installation), from the repository root, download the single **317 MB** archive to avoid
thousands of individual requests. These commands pin the dataset revision and
verify both the archive and its extracted files (requires `curl`):

```bash
mkdir -p data
curl -fL --retry 3 \
  https://huggingface.co/datasets/Amazon-FAR/far-prism-data/resolve/2498b1dbfda63a8000d28aa1b5d504cf9ff6b21b/far-prism-data.tar.gz \
  -o data/far-prism-data.tar.gz && \
  (cd data && \
   echo 'eb1fbcd2ef2714218292f9faf98e6cf57ac7f792da0becf89b3ca89d65850f5d  far-prism-data.tar.gz' | sha256sum --check && \
   tar -xzf far-prism-data.tar.gz && \
   cd far-prism-data && sha256sum --check SHA256SUMS --quiet)
```

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
