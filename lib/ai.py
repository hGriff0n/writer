
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

from .config2 import Config, ApiIntegration, ApiCosts

from langchain.chat_models import base, init_chat_model
from langchain_core.messages.base import BaseMessage
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.callbacks import UsageMetadataCallbackHandler


LlmModel = Union[base.BaseChatModel, base._ConfigurableModel]

# GOOGLE: https://docs.langchain.com/oss/python/integrations/providers/google
# OPENAI: https://docs.langchain.com/oss/python/integrations/chat/openai
# KIMI: https://docs.langchain.com/oss/python/integrations/chat/moonshot
# CLAUDE: https://docs.langchain.com/oss/python/integrations/chat/anthropic


# TODO: migrate this to separate app?
UNIT = 1_000_000

@dataclass
class CostCalculator:
    costs: dict[str, ApiCosts]

    def normalize(self, val: float) -> float:
        return val / UNIT

    def cost_of(self, call: UsageMetadata) -> float:
        cost = self.costs['standard']
        return call.get('input_tokens', 0) * cost.input + call.get('output_tokens', 0) * cost.output


@dataclass
class UsageTracker:
    usage: List[UsageMetadata]
    costs: CostCalculator

    def __init__(self, model: ApiIntegration):
        self.usage = []
        self.costs = CostCalculator(model.costs)

    def compute_cost(self):
        return sum(self.costs.cost_of(call) for call in self.usage)

    def append(self, usage: UsageMetadata):
        self.usage.append(usage)


@dataclass
class ChatLog:
    template: str
    conversation: List[Dict[str, str]]

    def _find_file_name(self, outdir: Path, date: str) -> Path:
        p = outdir / f'{date}.json'
        attempt = 0
        while p.exists():
            attempt += 1
            p = outdir / f'{date}_{attempt}.json'
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
    
    @staticmethod
    def load(dir: str, file: str) -> ChatLog:
        p = Path(dir, f'{file}.json')
        if not p.exists():
            raise FileNotFoundError(f'File `{p}` does not exist')
        return ChatLog(**json.loads(p.read_text()))


# Class to manage llm communications with some automatic features
class LlmEngine:

    DEFAULT_MODEL = 'gemini'

    def __init__(
        self,
        conf: Config,
        profile: str,
        prompt: str,
        schema: dict = None
    ):
        self._model = conf.get_config_for_model(profile)
        self._prompt = prompt
        if not self._model:
            raise Exception(f'Attempt to load unsupported profile: {profile}')
        
        self._log = ChatLog(template=self._prompt, conversation=[])
        self._usage = UsageTracker(self._model)
        self._llm = init_chat_model(self._model.name, model_provider=self._model.provider)

        self._structured = schema is not None
        if self._structured:
            self._llm = self._llm.with_structured_output(schema, method='json_schema', include_raw=True)
        self._callback = UsageMetadataCallbackHandler()

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

    def invoke(self, message: str | Dict, context: List[AnyMessage], *args, **kwargs) -> Tuple[str | Dict, Dict]:
        if isinstance(message, dict):
            context.append(HumanMessage([message]))
        else:
            context.append(HumanMessage(content=message))
        response = self._llm.invoke(input=context, *args, **kwargs)

        if not self._structured:
            context.append(response)
            usage = response.usage_metadata
            response = response.content
        elif 'raw' in response:
            context.append(response['raw'])
            usage = context[-1].usage_metadata
            response = response['parsed']
        else:
            usage = self._callback.usage_metadata
            context.append(SystemMessage([response]))
        
        self._usage.append(usage)
        self._record_chat(message, response)
        return response, usage

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
