
from dataclasses import dataclass
from typing import Dict, List, Optional
import yaml

# TODO: me - these would be a dataclass if I didn't use '-' in yaml
# TODO: me - Rewrite with dataclasses and one of these libraries
# https://catt.rs/en/stable/
# https://github.com/Fatal1ty/mashumaro?tab=readme-ov-file#usage-example

def _strip_comments(data: str) -> str:
    idx = data.find('[[comments]]')
    return data[:idx].strip()

def _load_yaml(file):
    with open(file, 'r') as f:
        return yaml.safe_load(f)
    
def _load_markdown(file):
    with open(file, 'r') as f:
        return _strip_comments(f.read())

# For system/program constants that are not mutable
# Theoretically, this could be eventually used to support multiple different
# llms with specific configurations, while allowing for profiles to reduce
# the cost of remembering the options
class DataConstants:
    FILE_LOCATION = './data/constants.yaml'

    def __init__(self):
        self._data = _load_yaml(self.FILE_LOCATION)
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


# Extremely basic story file
# TODO: Adjust when I change the aspects to be separated
class StoryFile:

    def __init__(self, story, path, yaml):
        self._story = story
        self._path = path
        self._yaml = yaml
        self._schemas = None
        self._rules = None
        self._engines = None
        self._concepts = None
        self._generation = None
        self._intent = None
        self._start = None
        self._writer = None

    def _load(self, file: str):
        return _load_markdown(f'{self._path}/{file}.md')

    def _load_all(self, files: List[str]):
        return [self._load(f) for f in files]
    
    @property
    def title(self) -> str:
        return self._story

    @property
    def writer(self) -> str:
        if not self._writer:
            self._writer = self._load(self._yaml['writer'])
        return self._writer

    @property
    def first_turn(self) -> str:
        if not self._start:
            self._start = self._load(self._yaml['first_turn'])
        return self._start

    @property
    def narrative_intent(self) -> str:
        if not self._intent:
            self._intent = self._load(self._yaml['first_turn'])
        return f'<narrative_intent>{self._intent}</narrative_intent>'
    
    @property
    def beat_assembly(self) -> str:
        if not self._generation:
            self._generation = self._load(self._yaml['generation'])
        return f'<beat_assembly_rules>{self._generation}</beat_assembly_rules>'
    
    @property
    def core_concepts(self) -> str:
        if not self._concepts:
            self._concepts = self._load_all(self._yaml['core_concepts'])
        return f'<core_concepts>{'\n\n'.join(self._concepts)}</core_concepts>'
    
    @property
    def engines(self) -> str:
        if not self._engines:
            self._engines = self._load_all(self._yaml['engines'])
        return f'<engines>{'\n\n'.join(self._engines)}</engines>'
    
    @property
    def rules(self) -> str:
        if not self._rules:
            self._rules = self._load_all(self._yaml['rules'])
        return f'<data_and_rules>{'\n\n'.join(self._rules)}</data_and_rules>'
    
    @property
    def schemas(self) -> str:
        if not self._schemas:
            self._schemas = self._load_all(self._yaml['schemas'])
        return '\n\n'.join(self._schemas)


# For user-specific configurations (also ai profiles)
class Config:
    CONFIG_FILE_LOCATION = './data/config.yaml'

    def __init__(self, constants: DataConstants):
        self._defs = constants
        self._data = _load_yaml(self.CONFIG_FILE_LOCATION)
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

    # Helpers for loading data from prompt and story files
    # TODO: me - Not sure if this is the best approach for
    # development, just cause I won't be iterating there
    def load_prompt_file(self, prompt: str) -> str:
        with open(f'./{self.directories.prompts}/{prompt}.md', 'r') as f:
            return _strip_comments(f.read())

    def load_story(self, story: str) -> str:
        base_file = f'./{self.directories.story}/{story}/principles'
        with open(f'{base_file}/story.yaml', 'r') as f:
            return StoryFile(story, base_file, yaml.safe_load(f))

    def load_story_file(self, story: str, file: str) -> str:
        with open(f'./{self.directories.story}/{story}/{file}.md', 'r') as f:
            return _strip_comments(f.read())


def load_config(defaults: DataConstants) -> Config:
    return Config(defaults)
