
from dataclasses import dataclass
from typing import Dict, Optional
import yaml


# TODO: me - not sure how I would handle provider in this situation
# typing.Annotated ???
@dataclass
class LlmConfig:
    name: str
    api_key: Optional[Dict[str, str]]
    provider: Optional[str]
    url: Optional[str]
    model_config: Dict[str, any]


# TODO: me - this would be a dataclass if I didn't use '-' in yaml
class Config:
    CONFIG_FILE_LOCATION = './data/config.yaml'

    def __init__(self):
        with open(self.CONFIG_FILE_LOCATION, 'r') as f:
            self.__data = yaml.safe_load(f)

    @property
    def ai_providers(self) -> Dict[str, LlmConfig]:
        return self.__data['ai-providers']
    
    @property
    def prompt_dir(self) -> str:
        return self.__data['prompt-dir']
    
    @property
    def output_dir(self) -> str:
        return self.__data['output-dir']
    
    @property
    def story_dir(self) -> str:
        return self.__data['story-dir']
    
    # Helpers for loading data from prompt and story files
    # TODO: me - Not sure if this is the best approach for
    # development, just cause I won't be iterating there
    def load_prompt_file(self, prompt: str) -> str:
        with open(f'./{self.prompt_dir}/{prompt}.md', 'r') as f:
            return f.read()

    def load_story_file(self, story: str, file: str) -> str:
        with open(f'./{self.story_dir}/{story}/{file}.md', 'r') as f:
            return f.read()


def load_config() -> Config:
    return Config()