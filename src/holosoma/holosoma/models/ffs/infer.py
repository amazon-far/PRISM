"""Fast-FoundationStereo depth predictor.

Drop-in alternative to `holosoma.models.gum.infer.GUM`: same `predict()` contract,
so `image_server.py` can produce the `depth_gum` channel from Fast-FoundationStereo
(NVlabs, CVPR 2026) instead of GUM without any change to the policy side.

Enable by setting `HOLOSOMA_DEPTH_PREDICTOR=ffs` alongside an image server config
that already has `enable_gum_depth_prediction=True` (e.g. `real_depth_gum_d435i`).

The pinned upstream code is installed by install.sh. Set HOLOSOMA_FFS_MODEL to
an upstream checkpoint with its sibling cfg.yaml; real-time inference uses CUDA.

"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class FFSConfig:
    """Configuration for the Fast-FoundationStereo depth predictor."""

    repo_dir: str = ""
    """Fast-FoundationStereo checkout. Defaults to $HOLOSOMA_FFS_REPO."""

    model_checkpoint: str = ""
    """Checkpoint path; its directory must also contain cfg.yaml. Defaults to $HOLOSOMA_FFS_MODEL."""

    device: str = "cuda"
    valid_iters: int = 8
    """Refinement iterations. Matches cfg.yaml shipped with the c-fast-foundationstereo
    checkpoint. At 512x288 on the RTX PRO 5000 laptop GPU: 8 iters ~27 ms p99,
    16 iters ~41 ms p99, so 16 blows the 33 ms image-server budget."""

    max_disp: int = 192
    hierarchical: bool = False
    """Use run_hierachical() (upstream spelling) for high-resolution inputs."""

    infer_height: int = 288
    infer_width: int = 512
    """Resolution the network runs at (per eye). The input pair is downscaled to
    this and the depth is resampled back to the input size, so callers get depth
    at their own resolution. On the RTX PRO 5000 laptop GPU, 512x288 holds
    ~27 ms p99 against the 33 ms image-server budget; native 848x480 does not.
    0 disables the rescale."""

    depth_min: float = 0.2
    depth_max: float = 10.0


class FastFoundationStereo:
    """Predict metric depth from a side-by-side stereo pair."""

    def __init__(self, cfg: FFSConfig | None = None, dtype=None):
        import torch

        self.cfg = cfg or FFSConfig()
        self.torch = torch

        repo_dir = self.cfg.repo_dir or os.environ.get("HOLOSOMA_FFS_REPO", "")
        if not repo_dir:
            repo_dir = str(Path(__file__).resolve().parents[5] / "third_party" / "Fast-FoundationStereo")
        model_path = self.cfg.model_checkpoint or os.environ.get("HOLOSOMA_FFS_MODEL", "")
        if not repo_dir:
            raise ValueError("Fast-FoundationStereo repo not set. Set HOLOSOMA_FFS_REPO or FFSConfig.repo_dir.")
        if not model_path:
            raise ValueError("Fast-FoundationStereo checkpoint not set. Set HOLOSOMA_FFS_MODEL.")

        repo_dir = str(Path(repo_dir).expanduser().resolve())
        if not (Path(repo_dir) / "core" / "foundation_stereo.py").is_file():
            raise FileNotFoundError("FFS submodule is missing. Run bash install.sh or set HOLOSOMA_FFS_REPO.")
        if repo_dir not in sys.path:
            # Upstream modules import each other as top-level packages (core.*, Utils).
            sys.path.insert(0, repo_dir)

        model_path = Path(model_path).expanduser().resolve()
        cfg_path = model_path.parent / "cfg.yaml"
        if not model_path.exists():
            raise FileNotFoundError(f"Fast-FoundationStereo checkpoint not found: {model_path}")
        if not cfg_path.exists():
            raise FileNotFoundError(f"Expected cfg.yaml next to the checkpoint: {cfg_path}")

        self.device = torch.device(self.cfg.device)
        self.dtype = dtype

        # Upstream ships the whole model pickled, not a state dict.
        self.model = torch.load(str(model_path), map_location="cpu", weights_only=False)
        self.model.args.max_disp = self.cfg.max_disp
        self.model.args.valid_iters = self.cfg.valid_iters

        # The commercially licensed c-fast-foundationstereo checkpoint was pickled
        # without `normalize`, which core/foundation_stereo.py reads unconditionally
        # when building the cost volume. Upstream's own run_demo.py crashes on it the
        # same way. Fill it with the default that build_gwc_volume_* declares.
        if "normalize" not in self.model.args:
            self.model.args.normalize = True
            print("[FFS] checkpoint args had no 'normalize'; defaulting to True (repo default)")

        self.model = self.model.to(self.device).eval()

        # Only importable once repo_dir is on sys.path.
        from core.utils.utils import InputPadder  # noqa: E402

        self._InputPadder = InputPadder

        print(f"[FFS] Initialized on {self.device} from {model_path}")
        print(f"[FFS] valid_iters={self.cfg.valid_iters} max_disp={self.cfg.max_disp}")
        print(f"[FFS] Depth range: [{self.cfg.depth_min}, {self.cfg.depth_max}] meters")

    @staticmethod
    def _baseline_from_extrinsics(camera_extrinsics: np.ndarray) -> float:
        """Stereo baseline in meters from the (2, 4, 4) extrinsics pair.

        Matches the convention in holosoma.sensors.zed, where the right eye's
        extrinsics carry the horizontal offset in the translation column.
        """
        extr = np.asarray(camera_extrinsics, dtype=np.float64)
        if extr.shape != (2, 4, 4):
            raise ValueError(f"camera_extrinsics must be (2, 4, 4), got {extr.shape}")
        baseline = abs(float(extr[1][0, 3]) - float(extr[0][0, 3]))
        if baseline <= 0.0:
            raise ValueError(f"Non-positive stereo baseline derived from extrinsics: {baseline}")
        return baseline

    def predict(
        self,
        side_by_side_image: np.ndarray,
        camera_intrinsics: np.ndarray,  # (2, 3, 3)
        camera_extrinsics: np.ndarray,  # (2, 4, 4)
    ) -> np.ndarray:
        """Return metric depth (H, W) in meters for one side-by-side stereo image."""
        torch = self.torch

        img = np.asarray(side_by_side_image)
        if img.ndim != 3 or img.shape[1] % 2 != 0:
            raise ValueError(f"expected a side-by-side HxWx3 image with even width, got {img.shape}")
        half = img.shape[1] // 2
        left, right = img[:, :half], img[:, half:]
        h0, w0 = left.shape[:2]

        intr = np.asarray(camera_intrinsics, dtype=np.float64)
        if intr.shape != (2, 3, 3):
            raise ValueError(f"camera_intrinsics must be (2, 3, 3), got {intr.shape}")
        fx = float(intr[0][0, 0])
        baseline = self._baseline_from_extrinsics(camera_extrinsics)

        # Run the network at a fixed, smaller resolution for speed and resample the
        # depth back afterwards. Disparity scales with image width, so fx is scaled
        # the same way to keep the depth metric.
        rescale = (
            self.cfg.infer_height > 0
            and self.cfg.infer_width > 0
            and (self.cfg.infer_height, self.cfg.infer_width) != (h0, w0)
        )
        if rescale:
            size = (self.cfg.infer_width, self.cfg.infer_height)
            left = cv2.resize(left, size, interpolation=cv2.INTER_AREA)
            right = cv2.resize(right, size, interpolation=cv2.INTER_AREA)
            fx *= self.cfg.infer_width / w0

        with torch.no_grad():
            # Upstream feeds raw 0-255 values, not normalized images.
            t0 = torch.as_tensor(np.ascontiguousarray(left)).to(self.device).float()[None].permute(0, 3, 1, 2)
            t1 = torch.as_tensor(np.ascontiguousarray(right)).to(self.device).float()[None].permute(0, 3, 1, 2)
            padder = self._InputPadder(t0.shape, divis_by=32, force_square=False)
            t0, t1 = padder.pad(t0, t1)

            # fp16 autocast matches Utils.AMP_DTYPE upstream; the model is trained
            # and profiled under it, so running without it changes the output.
            with torch.amp.autocast(self.device.type, enabled=True, dtype=torch.float16):
                if self.cfg.hierarchical:
                    disp = self.model.run_hierachical(
                        t0, t1, iters=self.cfg.valid_iters, test_mode=True, small_ratio=0.5
                    )
                else:
                    disp = self.model.forward(
                        t0, t1, iters=self.cfg.valid_iters, test_mode=True, optimize_build_volume="pytorch1"
                    )

            disp = padder.unpad(disp.float())

        h, w = left.shape[:2]
        disp = disp.detach().cpu().numpy().reshape(h, w).clip(0, None)

        # disparity -> metric depth; guard the division so zero/negative disparity
        # becomes far-plane rather than inf/NaN, which the policy cannot consume.
        with np.errstate(divide="ignore", invalid="ignore"):
            depth = (fx * baseline) / disp
        depth[~np.isfinite(depth)] = self.cfg.depth_max
        depth[disp <= 0] = self.cfg.depth_max
        depth = np.clip(depth, self.cfg.depth_min, self.cfg.depth_max).astype(np.float32)

        if rescale:
            # Back to the caller's resolution so downstream crop constants, which
            # are expressed in native pixels, keep their meaning.
            depth = cv2.resize(depth, (w0, h0), interpolation=cv2.INTER_LINEAR)
        return depth
