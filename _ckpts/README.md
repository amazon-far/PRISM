# Checkpoints

| File | Purpose |
|---|---|
| [`teacher_40000.pt`](teacher_40000.pt) | Privileged teacher · 40K |
| [`student_28000.pt`](student_28000.pt) | Distilled depth policy · 28K |
| [`box_23000.pt`](box_23000.pt) | Box policy warm-start · 23K |

Use these weights for inference or actor initialization. They include runtime
configuration but omit optimizer and training-session state, so they cannot be
used for an exact training resume. Checksums are in [`manifest.json`](manifest.json).

New training uses `peak_height` button labels; released policies retain their
trained command semantics.
