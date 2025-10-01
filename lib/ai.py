
from dataclasses import asdict, dataclass
import json
import os
from typing import Dict, List, Union

from .config import Config

from langchain.chat_models import base, init_chat_model
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage


LlmModel = Union[base.BaseChatModel, base._ConfigurableModel]

@dataclass
class ChatLog:
    template: str
    conversation: List[Dict[str, str]]

    def save(self, outdir: str):
        with open(f'./{outdir}/data.json', 'a', encoding='utf-8') as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=4)

    def having_role(self, role: str) -> List[str]:
        return map(lambda m: m['msg'],
                   filter(lambda m: m['role'] == role, self.conversation))


def load_chat_log(dir: str) -> ChatLog:
    with open(f'./{dir}/data.json', 'r', encoding='utf-8') as f:
        return ChatLog(**json.load(f))



def init_model(config: Config, model: str) -> LlmModel:
    llm_config = config.ai_providers[model]
    if (llm_config.get('api-key')):
        os.environ[llm_config['api-key']['name']] = llm_config['api-key']['value']
    return init_chat_model(
        llm_config['name'], model_provider=llm_config.get('provider'))

