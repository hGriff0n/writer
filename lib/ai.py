
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

from .config import Config, DataConstants

from langchain.chat_models import base, init_chat_model
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.messages.ai import UsageMetadata


LlmModel = Union[base.BaseChatModel, base._ConfigurableModel]

# TODO: migrate this to separate app?
UNIT = 1_000_000

@dataclass
class CallCost:
    input: float
    output: float
    threshold: Optional[int] = None
    applies_when: Optional[str] = None

    def normalize(self):
        self.input = self.input / UNIT
        self.output = self.output / UNIT

    def _get_check(self):
        if self.applies_when == "over":
            return lambda x: x > self.threshold
        if self.applies_when == "under":
            return lambda x: x < self.threshold
        return lambda _: True

    def cost_of(self, call: UsageMetadata) -> float:
        cost = 0
        check = self._get_check()
        if check(call['input_tokens']):
            cost += call['input_tokens'] * self.input
        if check(call['output_tokens']):
            cost += call['output_tokens'] * self.output
        return cost


@dataclass
class ApiCost:
    standard: CallCost
    surge: Optional[CallCost]
    batch: Optional[bool]

    def __init__(self, standard: Dict[str, any], surge: Optional[Dict[str, any]] = None, batch: bool = None):
        self.standard = CallCost(**standard)
        self.surge = surge and CallCost(**surge) or None
        batch = batch

    def normalize(self):
        self.standard.normalize()
        if self.surge:
            self.surge.normalize()

    def cost_of(self, call: UsageMetadata) -> float:
        total_cost = 0
        if self.surge:
            total_cost += self.surge.cost_of(call)
        return total_cost + self.standard.cost_of(call)


@dataclass
class UsageTracker:
    usage: List[UsageMetadata]
    costs: ApiCost

    def __init__(self, model: str, defs: DataConstants):
        vals = defs.get_config_for_model(model)
        if not vals:
            raise Exception(f'Attempt to load unsupported mode {model}')
        self.usage = []
        self.costs = ApiCost(**vals['costs'])
        self.costs.normalize()

    def compute_cost(self):
        return sum(self.costs.cost_of(call) for call in self.usage)

    def append(self, usage: UsageMetadata):
        self.usage.append(usage)


@dataclass
class ChatLog:
    template: str
    conversation: List[Dict[str, str]]

    def _find_file_name(self, outdir: str, date: str) -> Path:
        p = Path(outdir, f'{date}.json')
        attempt = 0
        while p.exists():
            attempt += 1
            p = Path(outdir, f'{date}_{attempt}.json')
        return p

    def save(self, outdir: str) -> str:
        date = datetime.now().strftime('%Y%m%d')
        file = self._find_file_name(outdir, date)
        file.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=4))
        return str(file.absolute())

    def having_role(self, role: str) -> List[str]:
        return map(lambda m: m['msg'],
                   filter(lambda m: m['role'] == role, self.conversation))


def load_chat_log(dir: str, filename: str) -> ChatLog:
    p = Path(dir, f'{filename}.json')
    if not p.exists():
        raise FileNotFoundError(f'File `{p}` does not exist')
    return ChatLog(**json.loads(p.read_text()))


def init_model(llm_config: Dict[str, any], *args, **kwargs) -> LlmModel:
    if (llm_config.get('api-key')):
        os.environ[llm_config['api-key']['name']
                   ] = llm_config['api-key']['value']
    return init_chat_model(
        llm_config['name'], model_provider=llm_config.get('provider'), *args, **kwargs)


# Class to manage llm communications with some automatic features
class LlmEngine:

    # TODO: me - This actually might be better to load from config
    # Profiles actually
    @staticmethod
    def supported_models():
        return ['gemini', 'openai', 'gemini-flash']

    DEFAULT_MODEL = 'gemini'

    def __init__(self, config: Config, model: str, prompt_file: str = None, prompt: str = None, flash: bool = False, *args, **kwargs):
        # Resolve the selected model, allowing for model to actually indicate
        # a profile which auto-includes specific model settings
        self._model_name = model + (flash and '-flash' or '')
        profile = config.ai_profiles.get(self._model_name)
        if profile:
            self._model_name = profile['model_name']
        self._model_config = config.constants.get_config_for_model(
            self._model_name)
        if not self._model_config:
            raise Exception(
                f'Attempt to load unsupported mode: {self._model_name}')

        print(f'Creating model `{self._model_name}`: {self._model_config}')
        # Setup the rest of the engine.
        self._llm = init_model(self._model_config)
        self._prompt = prompt if prompt else config.load_prompt_file(prompt_file)
        self._log = ChatLog(template=self._prompt, conversation=[])
        self._usage = UsageTracker(self._model_name, config.constants)

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
    
    @prompt.setter
    def prompt(self, msg: str):
        self._prompt = msg

    @property
    def llm(self) -> LlmModel:
        return self._llm

    @property
    def usage_stats(self) -> UsageTracker:
        return self._usage

    def invoke(self, message: str, context: List[AnyMessage], *args, **kwargs) -> str:
        context.append(HumanMessage(content=message))
        response = self._llm.invoke(input=context, *args, **kwargs)
        context.append(response)
        self._usage.append(response.usage_metadata)
        self._record_chat(message, response.content)
        return response.content

    def est_cost(self):
        return self._usage.compute_cost()

    def ask_for_help(self, help_prompt: str, context: List[AnyMessage]) -> str:
        if not context:
            raise Exception("Asking for help requires a context to ask in")
        response = self._llm.invoke(input=[
            SystemMessage(content=help_prompt),
            HumanMessage(content=context[-1].content)
        ])
        self._usage.append(response.usage_metadata)
        return response.content
