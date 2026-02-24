from dataclasses import dataclass

from trl import BCOConfig, CPOConfig, DPOConfig, GKDConfig, GRPOConfig, KTOConfig, NashMDConfig, OnlineDPOConfig
from trl import ORPOConfig, PPOConfig, RLOOConfig, SFTConfig
from trl import GRPOTrainer, DPOTrainer, PPOTrainer, ORPOTrainer, KTOTrainer
from trl import RLOOTrainer, SFTTrainer, BCOTrainer, CPOTrainer, GKDTrainer, NashMDTrainer, OnlineDPOTrainer

from .BaseTrainerGradregDeepSpeed import BaseTrainerGradregDeepSpeed
from .gradreg_config import GradRegConfigMixin, TrainArgsGradReg


GRPOGradRegConfig = dataclass(type("GRPOGradRegConfig", (GradRegConfigMixin, GRPOConfig), {}))
DPOGradRegConfig = dataclass(type("DPOGradRegConfig", (GradRegConfigMixin, DPOConfig), {}))
PPOGradRegConfig = dataclass(type("PPOGradRegConfig", (GradRegConfigMixin, PPOConfig), {}))
ORPOGradRegConfig = dataclass(type("ORPOGradRegConfig", (GradRegConfigMixin, ORPOConfig), {}))
KTOGradRegConfig = dataclass(type("KTOGradRegConfig", (GradRegConfigMixin, KTOConfig), {}))
RLOOGradRegConfig = dataclass(type("RLOOGradRegConfig", (GradRegConfigMixin, RLOOConfig), {}))
SFTGradRegConfig = dataclass(type("SFTGradRegConfig", (GradRegConfigMixin, SFTConfig), {}))
BCOGradRegConfig = dataclass(type("BCOGradRegConfig", (GradRegConfigMixin, BCOConfig), {}))
CPOGradRegConfig = dataclass(type("CPOGradRegConfig", (GradRegConfigMixin, CPOConfig), {}))
GKDGradRegConfig = dataclass(type("GKDGradRegConfig", (GradRegConfigMixin, GKDConfig), {}))
NashMDGradRegConfig = dataclass(type("NashMDGradRegConfig", (GradRegConfigMixin, NashMDConfig), {}))
OnlineDPOGradRegConfig = dataclass(type("OnlineDPOGradRegConfig", (GradRegConfigMixin, OnlineDPOConfig), {}))


GRPOTrainerGradreg = type("GRPOTrainerGradreg", (BaseTrainerGradregDeepSpeed, GRPOTrainer), {})
DPOTrainerGradreg = type("DPOTrainerGradreg", (BaseTrainerGradregDeepSpeed, DPOTrainer), {})
PPOTrainerGradreg = type("PPOTrainerGradreg", (BaseTrainerGradregDeepSpeed, PPOTrainer), {})
ORPOTrainerGradreg = type("ORPOTrainerGradreg", (BaseTrainerGradregDeepSpeed, ORPOTrainer), {})
KTOTrainerGradreg = type("KTOTrainerGradreg", (BaseTrainerGradregDeepSpeed, KTOTrainer), {})
RLOOTrainerGradreg = type("RLOOTrainerGradreg", (BaseTrainerGradregDeepSpeed, RLOOTrainer), {})
SFTTrainerGradreg = type("SFTTrainerGradreg", (BaseTrainerGradregDeepSpeed, SFTTrainer), {})
BCOTrainerGradreg = type("BCOTrainerGradreg", (BaseTrainerGradregDeepSpeed, BCOTrainer), {})
CPOTrainerGradreg = type("CPOTrainerGradreg", (BaseTrainerGradregDeepSpeed, CPOTrainer), {})
GKDTrainerGradreg = type("GKDTrainerGradreg", (BaseTrainerGradregDeepSpeed, GKDTrainer), {})
NashMDTrainerGradreg = type("NashMDTrainerGradreg", (BaseTrainerGradregDeepSpeed, NashMDTrainer), {})
OnlineDPOTrainerGradreg = type("OnlineDPOTrainerGradreg", (BaseTrainerGradregDeepSpeed, OnlineDPOTrainer), {})


__all__ = [
    "BaseTrainerGradregDeepSpeed",
    "GradRegConfigMixin",
    "TrainArgsGradReg",
    "GRPOTrainerGradreg",
    "DPOTrainerGradreg",
    "PPOTrainerGradreg",
    "ORPOTrainerGradreg",
    "KTOTrainerGradreg",
    "RLOOTrainerGradreg",
    "SFTTrainerGradreg",
    "BCOTrainerGradreg",
    "CPOTrainerGradreg",
    "GKDTrainerGradreg",
    "NashMDTrainerGradreg",
    "OnlineDPOTrainerGradreg",
    "GRPOGradRegConfig",
    "DPOGradRegConfig",
    "PPOGradRegConfig",
    "ORPOGradRegConfig",
    "KTOGradRegConfig",
    "RLOOGradRegConfig",
    "SFTGradRegConfig",
    "BCOGradRegConfig",
    "CPOGradRegConfig",
    "GKDGradRegConfig",
    "NashMDGradRegConfig",
    "OnlineDPOGradRegConfig",
]
