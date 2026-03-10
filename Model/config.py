from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch


@dataclass
class LOBDataConfig:
    levels: int = 10
    history_len: int = 100
    standardize: bool = True
    use_cond_features: bool = True
    cond_depths: Tuple[int, ...] = (1, 3, 5, 10)
    cond_vol_window: int = 50
    cond_standardize: bool = True

    @property
    def state_dim(self) -> int:
        return 4 * int(self.levels)


@dataclass
class SharedModelConfig:
    hidden_dim: int = 128
    dropout: float = 0.1
    cond_dim: int = 0
    cfg_dropout: float = 0.1
    ctx_encoder: str = "multiscale"
    ctx_heads: int = 4
    ctx_layers: int = 2
    use_res_mlp: bool = True
    film_conditioning: bool = True
    fu_net_type: str = "transformer"
    fu_net_layers: int = 3
    fu_net_heads: int = 4
    lobiflow_profile: str = "recommended"
    use_multiscale_context: bool = False


@dataclass
class FMConfig:
    lambda_mean: float = 1.0
    lambda_pair: float = 0.25
    lambda_prior: float = 1e-3
    lambda_consistency: float = 0.0
    lambda_imbalance: float = 0.05
    pair_align_weight: float = 0.25
    use_minibatch_ot: bool = True
    use_paired_encoder: bool = True
    use_pair_regularizer: bool = True
    use_consistency_training: bool = False
    consistency_steps: int = 32
    consistency_step_choices: Tuple[int, ...] = ()


@dataclass
class NFConfig:
    flow_layers: int = 6
    flow_scale_clip: float = 2.0
    share_coupling_backbone: bool = True


@dataclass
class TrainConfig:
    batch_size: int = 64
    steps: int = 20_000
    lr: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    lr_warmup_steps: int = 500
    lr_schedule: str = "cosine"
    use_swa: bool = True
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eps: float = 1e-8


@dataclass
class SampleConfig:
    steps: int = 32
    cfg_scale: float = 1.0


@dataclass(init=False)
class LOBConfig:
    data: LOBDataConfig = field(default_factory=LOBDataConfig)
    model: SharedModelConfig = field(default_factory=SharedModelConfig)
    fm: FMConfig = field(default_factory=FMConfig)
    nf: NFConfig = field(default_factory=NFConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    sample: SampleConfig = field(default_factory=SampleConfig)

    def __init__(
        self,
        data: Optional[LOBDataConfig] = None,
        model: Optional[SharedModelConfig] = None,
        fm: Optional[FMConfig] = None,
        nf: Optional[NFConfig] = None,
        train: Optional[TrainConfig] = None,
        sample: Optional[SampleConfig] = None,
        **flat_overrides: Any,
    ):
        object.__setattr__(self, "data", data if data is not None else LOBDataConfig())
        object.__setattr__(self, "model", model if model is not None else SharedModelConfig())
        object.__setattr__(self, "fm", fm if fm is not None else FMConfig())
        object.__setattr__(self, "nf", nf if nf is not None else NFConfig())
        object.__setattr__(self, "train", train if train is not None else TrainConfig())
        object.__setattr__(self, "sample", sample if sample is not None else SampleConfig())
        if flat_overrides:
            self.apply_overrides(**flat_overrides)

    def __getattr__(self, name: str) -> Any:
        for section_name in ("data", "model", "fm", "nf", "train", "sample"):
            section = object.__getattribute__(self, section_name)
            if hasattr(section, name):
                return getattr(section, name)
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    def apply_overrides(self, **flat_overrides: Any) -> "LOBConfig":
        for key, value in flat_overrides.items():
            matched = False
            for section_name in ("data", "model", "fm", "nf", "train", "sample"):
                section = getattr(self, section_name)
                if hasattr(section, key):
                    setattr(section, key, value)
                    matched = True
                    break
            if not matched:
                raise TypeError(f"Unknown config field: {key}")
        return self

    @property
    def state_dim(self) -> int:
        return self.data.state_dim

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        return {
            "data": asdict(self.data),
            "model": asdict(self.model),
            "fm": asdict(self.fm),
            "nf": asdict(self.nf),
            "train": asdict(self.train),
            "sample": asdict(self.sample),
        }