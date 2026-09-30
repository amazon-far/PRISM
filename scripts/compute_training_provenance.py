#!/usr/bin/env python3
"""Compatibility entry point for the installed HoloSoma provenance utility."""
import sys
from holosoma.utils import compute_training_provenance as implementation

if __name__ == "__main__":
    implementation.main()
else:
    sys.modules[__name__] = implementation
