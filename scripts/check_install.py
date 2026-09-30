"""Verify deployment imports without opening a camera or robot interface."""
import argparse
import importlib
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--relay', action='store_true')
    mode.add_argument('--native-depth', action='store_true')
    args = parser.parse_args()
    modules = ['numpy', 'pyrealsense2', 'zmq']
    if not args.relay:
        modules += ['cv2', 'pinocchio', 'onnxruntime', 'unitree_interface',
                    'holosoma.sensors.image_server', 'holosoma_inference.run_policy']
    if not args.relay and not args.native_depth:
        repo = Path(__file__).resolve().parents[1] / 'third_party/Fast-FoundationStereo'
        if not (repo / 'core/foundation_stereo.py').is_file():
            parser.error('FFS submodule is missing; run git submodule update --init --recursive.')
        sys.path.insert(0, str(repo))
        modules += ['torch', 'torchvision', 'xformers', 'timm', 'einops', 'omegaconf',
                    'skimage', 'imageio', 'yaml', 'open3d', 'core.foundation_stereo',
                    'holosoma.models.ffs.infer']
    for module in modules:
        importlib.import_module(module)
    print(f'Installation verified: {len(modules)} imports; no camera or robot opened.')
    if not args.relay and not args.native_depth:
        import torch
        print(f'PyTorch {torch.__version__}; CUDA runtime {torch.version.cuda}; CUDA available: {torch.cuda.is_available()}')
        print('Set HOLOSOMA_FFS_MODEL to the model file with cfg.yaml in the same directory.')


if __name__ == '__main__':
    main()
