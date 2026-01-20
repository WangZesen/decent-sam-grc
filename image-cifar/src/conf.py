from pydantic import BaseModel, ConfigDict, Field
from typing import Literal, Union, TypeAlias, List
from src.data import Dataset
from src.model import Model
import tomli_w
import tomli
from loguru import logger
from enum import Enum


class _BaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SGDConfig(_BaseModel):
    name: Literal["sgd"] = "sgd"
    momentum: float = 0.9
    weight_decay: float = 5e-4


class CosineLRSchedulerConfig(_BaseModel):
    name: Literal["cosine"] = "cosine"
    warmup_epochs: int = 10
    warmup_decay: float = 0.1
    eta_min: float = 0.0


class NormalMixConfig(_BaseModel):
    name: Literal["normal"] = "normal"


class AdaptiveMixConfig(_BaseModel):
    name: Literal["adaptive"] = "adaptive"
    p: float = 3.0
    min_gamma: float = 0.0
    start_epoch: int = 10


ALL_MIX_CONFIGS: TypeAlias = Union[NormalMixConfig, AdaptiveMixConfig]


class SyncTrainConfig(_BaseModel):
    name: Literal["sync"] = "sync"


class Topology(str, Enum):
    RING = "ring"
    EXP = "exp"
    COMPLETE = "complete"


class DecentTrainConfig(_BaseModel):
    name: Literal["decent"] = "decent"
    topology: Topology = Topology.COMPLETE
    mix: ALL_MIX_CONFIGS = Field(default_factory=NormalMixConfig, discriminator="name")


ALL_TRAINER_CONFIGS: TypeAlias = Union[SyncTrainConfig, DecentTrainConfig]


class LogConfig(_BaseModel):
    eval_interval: int = 1
    project: str = "decent-sam-image"


class Config(_BaseModel):
    dataset: Dataset = Dataset.CIFAR10
    model: Model = Model.WRN_28_10
    epochs: int = 200
    base_lr: float = 0.1
    scheduler: CosineLRSchedulerConfig = Field(default_factory=CosineLRSchedulerConfig)
    optimizer: SGDConfig = Field(default_factory=SGDConfig)
    batch_size: int = 128
    seed: int = 42
    log: LogConfig = Field(default_factory=LogConfig)
    trainer: ALL_TRAINER_CONFIGS = Field(default_factory=DecentTrainConfig)
    amp: bool = False


class Env(_BaseModel):
    world_size: int
    gpu: str
    node_list: str


def merge_dicts_recursive(d1, d2):
    """
    Recursively merges two dictionaries.
    Values from d2 take precedence when keys overlap.

    Args:
        d1 (dict): The first dictionary.
        d2 (dict): The second dictionary (overrides d1 on conflicts).

    Returns:
        dict: A new merged dictionary.
    """
    merged = dict(d1)  # Make a shallow copy to avoid mutating the original

    for key, value in d2.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            merged[key] = merge_dicts_recursive(merged[key], value)
        else:
            # Override or add the value from d2
            if (key in merged) and (merged[key] != value):
                logger.warning(
                    f"Overriding key '{key}' from value '{merged[key]}' to '{value}'"
                )
            merged[key] = value

    return merged


def load_all_configs(config_files: List[str]) -> Config:
    """Load and merge multiple configuration files, with later files overriding earlier ones."""
    merged_data = {}

    for config_file in config_files:
        with open(config_file, "rb") as f:
            config_data = tomli.load(f)
        # Merge the current file's data into the merged_data
        # Later files override earlier ones
        merged_data = merge_dicts_recursive(merged_data, config_data)

    # Validate the merged data against the Config model
    return Config.model_validate(merged_data)


def dump_config_to_file(config: Config, file_dir: str):
    with open(file_dir, "wb") as f:
        tomli_w.dump(config.model_dump(), f)
