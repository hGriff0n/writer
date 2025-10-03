
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Dict, List, Union

from .config import Config

from langchain.chat_models import base, init_chat_model
from langchain_core.messages import AnyMessage, BaseMessage, HumanMessage


LlmModel = Union[base.BaseChatModel, base._ConfigurableModel]

# TODO: migrate this to separate app?
UNIT = 1_000_000
__API_COSTS = {
    'gemini': {
        'surge': 200_000,
        'input': 1.25 / UNIT,
        'in_surge': 2.5 / UNIT,
        'output': 10 / UNIT,
        'out_surge': 15 / UNIT
    }
}


def _cost_of_request(model_name: str, usage_metadata):
    api = __API_COSTS[model_name]
    input = usage_metadata['input_tokens']
    output = usage_metadata['output_tokens']
    icost = input > api['surge'] and api['in_surge'] or api['input']
    ocost = output > api['surge'] and api['out_surge'] or api['output']
    return input * icost + output * ocost


"""
GEMINI (2.5pro)
unit = 1_000_000
surge = 200_000

input: 1.25
input over surge: 2.5
output: 10
output over surge: 15
"""

"""
OpenAI (gpt5)
unit = 1_000_000

input: 1.25
input cached: 0.125
output: 10
"""

"""
Anthropic (opus 4.1)
unit = 1_000_000
prompt caching?

input: 15
output: 75
"""

"""
Mistral (magistral M)
unit = 1_000_000

input: 2
output: 5
"""

# TODO: me - How to provide defaults


@dataclass
class ChatLog:
    template: str
    conversation: List[Dict[str, str]]

    def _find_file_name(self, outdir: str, date: str) -> Path:
        p = Path(outdir, f'{date}.json')
        attempt = 0
        while p.exists():
            attempt += 1
            p = Path(outdir, f'{date}{attempt}.json')
        return p

    def save(self, outdir: str):
        date = datetime.now().strftime('%Y%m%d')
        file = self._find_file_name(outdir, date)
        file.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=4))

    def having_role(self, role: str) -> List[str]:
        return map(lambda m: m['msg'],
                   filter(lambda m: m['role'] == role, self.conversation))


def load_chat_log(dir: str, filename: str) -> ChatLog:
    p = Path(dir, f'{filename}.json')
    if not p.exists():
        raise FileNotFoundError(f'File `{p}` does not exist')
    return ChatLog(**json.loads(p.read_text()))


def init_model(config: Config, model: str) -> LlmModel:
    llm_config = config.ai_providers[model]
    if (llm_config.get('api-key')):
        os.environ[llm_config['api-key']['name']
                   ] = llm_config['api-key']['value']
    return init_chat_model(
        llm_config['name'], model_provider=llm_config.get('provider'))


# Class to manage llm communications with some automatic features
class LlmEngine:

    # TODO: me - This actually might be better to load from config
    @staticmethod
    def supported_models():
        return ['gemini', 'openai']
    
    DEFAULT_MODEL = 'gemini'

    def __init__(self, config: Config, model: str, prompt_file: str):
        self._model_name = model
        self._llm = init_model(config, model)
        self._prompt = config.load_prompt_file(prompt_file)
        self._log = ChatLog(template=self._prompt, conversation=[])
        self._usage = []

    def _record_chat(self, message: str, response: str):
        self._log.conversation.extend([
            {'role': 'ME', 'msg': message},
            {'role': 'AI', 'msg': response}
        ])

    @property
    def chat_log(self) -> ChatLog:
        return self._log

    @property
    def prompt(self) -> str:
        return self._prompt

    @property
    def llm(self) -> LlmModel:
        return self._llm

    @property
    def usage_stats(self):
        return self._usage

    def invoke(self, message: str, context: List[AnyMessage], *args, **kwargs) -> BaseMessage:
        context.append(HumanMessage(content=message))
        response = self._llm.invoke(input=context, *args, **kwargs)
        context.append(response)
        self._usage.append(response.usage_metadata)
        self._record_chat(message, response.content)
        return response.content

    def est_cost(self):
        return sum(map(lambda m: _cost_of_request(self._model_name, m), self._usage))
