from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import yaml
from pathlib import Path

class ParameterConfig(BaseModel):
    type: str
    default: Any
    optional: bool
    values: Optional[List[Any]] = None

class ModelConfig(BaseModel):
    type: str
    name: str
    key: str
    filename: str
    parameters: Dict[str, ParameterConfig]

class Config(BaseModel):
    models: Dict[str, ModelConfig]

    @classmethod
    def get(cls) -> "Config":
        config_path = Path("data/v2/model/config.yaml")
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        return cls(
            models={
                model['key']: model
                for model in config_data['models']
            }
        )