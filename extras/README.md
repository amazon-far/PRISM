# Optional tools and project material

The default deployment is `bash real_ffs.sh` from the repository root. It does not
depend on this directory. The files here are retained for development, comparison
and provenance; they have not been deleted.

| Directory | Contents |
| --- | --- |
| `analysis/` | Capture paired camera frames, compare native/FFS depth, and inspect deployment evidence. |
| `native_depth/` | Alternate native D435i depth launcher, compatibility policy entrypoint, and depth relay tools. |
| `tests/` | Offline release, deployment-audit, clock and latency regression tests. |
| `community/` | Citation, contributors, contribution guide and code of conduct. |
| `provenance/` | [Source manifest](provenance/source_manifest.json) with original-source and current-release checksums. Moved records retain their original `source_path`. |

Run optional tools from the repository root with the deployment environment active:

```bash
bash extras/native_depth/real_depth.sh --help
bash extras/native_depth/real_run.sh --help
python extras/analysis/capture_depth_pairs.py --help
python extras/analysis/compare_depth_sources.py --help
python extras/analysis/compare_real_depth_evidence.py --help
python extras/native_depth/depth_relay_pub.py --help
python extras/native_depth/depth_relay_sub.py --help
```

The alternate native-depth pipeline uses `bash install.sh --native-depth`.
The archived policy entrypoint still invokes the required root `real_drop.sh`;
both relocated shell scripts resolve the repository root from their own location.
Runtime evidence recording remains available through `HOLOSOMA_DEPLOYMENT_AUDIT=1`;
only the offline comparison tools and their tests moved here.

To run the regression suite after installing deployment dependencies and `pytest`:

```bash
PYTHONPATH=src/holosoma:src/holosoma_inference python -m pytest extras/tests
```

The FFS tests also need the pinned submodule (`git submodule update --init --recursive`).
Tests do not construct robot interfaces or start a real robot policy.
