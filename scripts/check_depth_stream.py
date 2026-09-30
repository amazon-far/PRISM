"""Check the source FFS depth range and frame-change criteria before policy startup."""
import argparse
from multiprocessing import resource_tracker, shared_memory
import os
from pathlib import Path
import time
import numpy as np


def depth_server_running():
    # Match both the script and module launch forms, excluding our own process.
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            if int(path.parent.name) == os.getpid():
                continue
            args = path.read_bytes().split(b'\0')
            if b'remote_ffs_d435i' in args and any(
                arg == b'holosoma.sensors.image_server' or arg.endswith(b'/image_server.py') for arg in args
            ):
                return True
        except (OSError, ValueError):
            continue
    return False


def inspect_frames(frames):
    frame = frames[-1]
    if not all(np.isfinite(x).all() and x.min() >= -0.5001 and x.max() <= 0.5001 for x in frames):
        raise ValueError('Non-finite or out-of-range depth; expected [-0.5, 0.5].')
    distinct = len({x.tobytes() for x in frames})
    saturated = bool(np.all(np.isclose(frame, -0.5)) or np.all(np.isclose(frame, 0.5)))
    if distinct < 5 and not saturated:
        raise ValueError(f'Depth is not updating ({distinct} distinct frames).')
    # Keep the source exception for fully saturated scenes explicit.
    if saturated:
        print('Depth is fully saturated; frame-change liveness cannot be established.')
    return distinct, float(np.isclose(frame, 0.5).mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timeout', type=float, default=120)
    args = parser.parse_args()
    deadline = time.monotonic() + args.timeout
    last_error = 'No local FFS image server.'
    while time.monotonic() < deadline:
        if not depth_server_running():
            time.sleep(0.2)
            continue
        shm = None
        try:
            shm = shared_memory.SharedMemory(name='depth_img_shm')
            resource_tracker.unregister(shm._name, 'shared_memory')
            frame = np.ndarray((1, 1, 58, 87), dtype=np.float32, buffer=shm.buf)
            samples = []
            until = time.monotonic() + 1.5
            while time.monotonic() < until:
                samples.append(frame.copy())
                time.sleep(0.02)
            distinct, far = inspect_frames(samples)
            if not depth_server_running():
                raise ValueError('Depth server exited during the check.')
            print(f'Depth check passed: {distinct} distinct frames; far-plane fraction {far:.1%}.')
            return
        except (FileNotFoundError, ValueError) as exc:
            last_error = str(exc)
        finally:
            if shm is not None:
                shm.close()
        time.sleep(0.2)
    raise SystemExit(f'Depth check failed: {last_error} Start real_ffs_depth.sh first.')


if __name__ == '__main__':
    main()
