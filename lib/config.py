
from dataclasses import dataclass
from typing import Dict, List, Optional
import yaml

# TODO: me - these would be a dataclass if I didn't use '-' in yaml
# TODO: me - Rewrite with dataclasses and one of these libraries
# https://catt.rs/en/stable/
# https://github.com/Fatal1ty/mashumaro?tab=readme-ov-file#usage-example

# For system/program constants that are not mutable
# Theoretically, this could be eventually used to support multiple different
# llms with specific configurations, while allowing for profiles to reduce
# the cost of remembering the options
class DataConstants:
    FILE_LOCATION = './data/constants.yaml'

    def __init__(self):
        with open(self.FILE_LOCATION, 'r') as f:
            self._data = yaml.safe_load(f)
        self._supported_names = [m['name']
                                 for m in self._data['supported-ais']]

    @property
    def supported_ais(self) -> List[str]:
        return self._supported_names

    def get_config_for_model(self, model: str) -> Optional[Dict[str, any]]:
        return next((
            a for a in self._data['supported-ais'] if a['name'] == model
        ), None)


@dataclass
class Directories:
    prompts: str
    output: str
    story: str


# For user-specific configurations (also ai profiles)
class Config:
    CONFIG_FILE_LOCATION = './data/config.yaml'

    def __init__(self, constants: DataConstants):
        self._defs = constants
        with open(self.CONFIG_FILE_LOCATION, 'r') as f:
            self._data = yaml.safe_load(f)
        self._dirs = Directories(**self._data['directories'])

    @property
    def constants(self) -> DataConstants:
        return self._defs

    @property
    def ai_profiles(self) -> Dict[str, any]:
        return self._data['ai-profiles']

    @property
    def ai_providers(self) -> Dict[str, any]:
        return self._data['ai-providers']

    @property
    def directories(self) -> Directories:
        return self._dirs

    @property
    def output_dir(self) -> str:
        return self._dirs.output
    
    def _strip_comments(self, data: str) -> str:
        idx = data.find('[[comments]]')
        return data[:idx].strip()

    # Helpers for loading data from prompt and story files
    # TODO: me - Not sure if this is the best approach for
    # development, just cause I won't be iterating there
    def load_prompt_file(self, prompt: str) -> str:
        with open(f'./{self.directories.prompts}/{prompt}.md', 'r') as f:
            return self._strip_comments(f.read())

    def load_story_file(self, story: str, file: str) -> str:
        with open(f'./{self.directories.story}/{story}/{file}.md', 'r') as f:
            return self._strip_comments(f.read())


def load_config(defaults: DataConstants) -> Config:
    return Config(defaults)
