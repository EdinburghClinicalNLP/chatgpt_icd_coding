from enum import Enum
from typing import List, Optional

from pydantic import BaseModel


class TrainingConfigs(BaseModel):
    dataset_path: str
    random_seed: int = 1234
    num_process: int = 8
    outputs_dir: str = "outputs"


class ModelConfigs(BaseModel):
    temperature: float
    top_p: float
    max_tokens: int
    frequency_penalty: float = 0
    presence_penalty: float = 0
    stop: Optional[str] = None
    engine: str = "gpt-3.5-turbo-0613"


class PromptConfigs(BaseModel):
    system_prompt: str
    user_prompt: str


class APIConfigs(BaseModel):
    api_version: str


class Configs(BaseModel):
    training_configs: TrainingConfigs
    model_configs: ModelConfigs
    prompt_configs: PromptConfigs
    api_configs: APIConfigs
