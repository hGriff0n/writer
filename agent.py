import random
from typing import Dict, Tuple

from langchain_core.tools import tool

class PlayerObj:
    skills: Dict[str, int] = {}
    attr: Dict[str, int] = {}

SKILL_ATTR_MAP: Dict[str, str] = {}

def get_character(id: str) -> PlayerObj:
    return PlayerObj()

@tool
def skill_check(skill: str, character: str, req: int) -> Tuple[bool, int]:
    roll = random.randint(1, 20)

    player = get_character(character)
    attr = SKILL_ATTR_MAP(skill)
    player_attr = player.attr[attr]
    player_lvl = player.skills[skill]
    # TODO: add equip modifiers/etc.
    total = roll + player_attr + player_lvl

    return total >= req, total

# FORCE TOOL: https://python.langchain.com/docs/how_to/tool_choice/
