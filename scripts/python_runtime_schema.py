"""Shared schema for the content-addressed AS Python runtime overlay.

The overlay deliberately contains only packages whose code is executed by the
scientific AS training path. DeFM weights are required to be local and
authenticated by HoloSoma, so the Hugging Face network client is not part of
this profile. Keeping the profile explicit prevents an unrelated optional
network stack from changing non-DeFM training identity.
"""

from __future__ import annotations


DISTRIBUTION_CONTRACT_NAME = ".holosoma-runtime-distributions.json"
DISTRIBUTION_CONTRACT_VERSION = 3
RUNTIME_PROFILE = "as-core-onnx-v2"
ROOT_DISTRIBUTIONS = (
    "attrs",
    "numpy",
    "omegaconf",
    "onnx",
    "onnxruntime",
)
