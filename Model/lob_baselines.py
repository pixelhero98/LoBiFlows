"""Flat-folder and package-compatible exports for the LoBiFlow bundle.

This file lets the rest of the project keep using:
    from lob_baselines import ...

It works in both cases:
1. the files live inside a real package directory, or
2. the files are copied into a plain folder and imported directly.
"""

from __future__ import annotations

import importlib
import os
import sys
import types


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SYNTHETIC_PACKAGE = "_lobiflow_local"


def _ensure_package() -> str:
    if _SYNTHETIC_PACKAGE not in sys.modules:
        pkg = types.ModuleType(_SYNTHETIC_PACKAGE)
        pkg.__path__ = [_THIS_DIR]
        sys.modules[_SYNTHETIC_PACKAGE] = pkg
    return _SYNTHETIC_PACKAGE


_PKG = _ensure_package()
_config = importlib.import_module(f"{_PKG}.config")
_modules = importlib.import_module(f"{_PKG}.modules")
_conditioning = importlib.import_module(f"{_PKG}.conditioning")
_baselines = importlib.import_module(f"{_PKG}.baselines")

LOBDataConfig = _config.LOBDataConfig
SharedModelConfig = _config.SharedModelConfig
FMConfig = _config.FMConfig
NFConfig = _config.NFConfig
TrainConfig = _config.TrainConfig
SampleConfig = _config.SampleConfig
LOBConfig = _config.LOBConfig

MLP = _modules.MLP
ResBlock = _modules.ResBlock
ResMLP = _modules.ResMLP
build_mlp = _modules.build_mlp
AdaLN = _modules.AdaLN
TransformerFUBlock = _modules.TransformerFUBlock
TransformerFUNet = _modules.TransformerFUNet
FiLMModulation = _modules.FiLMModulation
EMAModel = _modules.EMAModel

CondEmbedder = _conditioning.CondEmbedder
build_context_encoder = _conditioning.build_context_encoder
CrossAttentionConditioner = _conditioning.CrossAttentionConditioner
SharedConditioningBackbone = _conditioning.SharedConditioningBackbone

RectifiedFlowLOB = _baselines.RectifiedFlowLOB
BiFlowLOB = _baselines.BiFlowLOB
BiFlowNFBaseline = _baselines.BiFlowNFBaseline
BiFlowNFLOB = _baselines.BiFlowNFLOB

__all__ = [
    "LOBDataConfig",
    "SharedModelConfig",
    "FMConfig",
    "NFConfig",
    "TrainConfig",
    "SampleConfig",
    "LOBConfig",
    "MLP",
    "ResBlock",
    "ResMLP",
    "build_mlp",
    "AdaLN",
    "TransformerFUBlock",
    "TransformerFUNet",
    "FiLMModulation",
    "EMAModel",
    "CondEmbedder",
    "build_context_encoder",
    "CrossAttentionConditioner",
    "SharedConditioningBackbone",
    "RectifiedFlowLOB",
    "BiFlowLOB",
    "BiFlowNFBaseline",
    "BiFlowNFLOB",
]