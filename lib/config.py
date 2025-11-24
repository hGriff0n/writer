from enum import Enum
import pathlib
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from pydantic_yaml import parse_yaml_file_as
import json
import os

from .util import load_markdown


class ApiProvider(str, Enum):
    google = 'google_genai'
    openai = 'openai'
    anthropic = 'anthropic'
    moonshot = 'moonshot'  # Kimi doesn't seem to have a provider? it's a community plugin
    local = 'local'

class ApiCosts(BaseModel):
    input: float
    output: float

class ModelConfig(BaseModel):
    temperature: float

class RateLimits(BaseModel):
    rpm: int

class ApiIntegration(BaseModel):
    name: str
    api_key: str = Field(alias='api-key')
    provider: ApiProvider
    costs: dict[str, ApiCosts]
    default_config: Optional[ModelConfig] = Field(alias='default-config', default=None)
    local_model: bool = Field(default=False)
    rate_limits: Optional[RateLimits] = Field(alias='rate-limits', default=None)

class Constants(BaseModel):
    supported_ais: List[ApiIntegration] = Field(alias='supported-ais')
    api_keys: dict[str, str] = Field(alias='api-keys')

    @staticmethod
    def load(path: pathlib.Path) -> Constants:
        return parse_yaml_file_as(Constants, path)
    
    def get_config_for_model(self, model_name: str) -> ApiIntegration | None:
        return next((
            a for a in self.supported_ais if a.name == model_name
        ), None)


class ApiProfile(BaseModel):
    model_name: str
    provider: Optional[ApiProvider] = Field(default=None)
    url: Optional[str] = Field(default=None)

class DirectoryMap(BaseModel):
    prompts: str
    output: str
    story: str

class Profiles(BaseModel):
    ai_profiles: dict[str, ApiProfile] = Field(alias='ai-profiles')
    directories: DirectoryMap

    @staticmethod
    def load(path: pathlib.Path) -> Profiles:
        return parse_yaml_file_as(Profiles, path)
    

class StoryDef(BaseModel):
    principles: str
    story_arch: Optional[str] = Field(default=None)
    fullspec: str
    writer: str
    first_turn: str
    path: Optional[pathlib.Path] = Field(default=None)

    @staticmethod
    def load(path: pathlib.Path, story: str) -> StoryDef:
        s = parse_yaml_file_as(StoryDef, path / story / '_story.yaml')
        s.path = path
        return s
    
    def load_file(self, file: str):
        return load_markdown(self.path / f'{file}.md')

# TODO: me - Handle file exceptions
class Config:
    _defs = Constants.load(pathlib.Path('data/constants.yaml'))
    _prof = Profiles.load(pathlib.Path('data/config.yaml'))

    @property
    def supported_models(self) -> List[str]:
        return list(self._prof.ai_profiles.keys())

    @property
    def prompt_dir(self) -> pathlib.Path:
        return pathlib.Path('.', self._prof.directories.prompts)
    
    @property
    def story_dir(self) -> pathlib.Path:
        return pathlib.Path('.', self._prof.directories.story)
    
    @property
    def output_dir(self) -> pathlib.Path:
        return pathlib.Path('.', self._prof.directories.output)
    
    def load_prompt(self, prompt: str) -> str:
        return load_markdown(self.prompt_dir / f'{prompt}.md')
    
    def load_story(self, story: str) -> StoryDef:
        return StoryDef.load(self.story_dir, story)

    def load_schema(self, schema: str) -> Dict:
        with open(self.prompt_dir / f'{schema}.json', 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_config_for_model(self, model: str) -> ApiIntegration:
        profile = self._prof.ai_profiles[model]
        config = self._defs.get_config_for_model(profile.model_name)
        if self._defs.api_keys[config.api_key]:
            os.environ[config.api_key] = self._defs.api_keys[config.api_key]
        return config
    
    def get_config_for_profile(self, profile: str) -> ApiIntegration:
        return self.get_config_for_model(profile)
