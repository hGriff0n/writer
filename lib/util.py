import pathlib
import re

# Helper to extract xml encoded text sections
# This is useful because llms prefer xml for referential data for some reason
def extract_between_tags(tag, text):
    m = re.match(f"<{tag}>((?:.|[\r\n])*)</{tag}>", text)
    return m and m.group(1) or ""

# Helper to load a specific file and strip off comments
def _strip_comments(data: str) -> str:
    idx = data.find('[[comments]]')
    return (data[:idx].strip() if idx != -1 else data)

def load_markdown(file: str | pathlib.Path):
    try:
        with open(file, 'r', encoding='utf-8') as f:
            return _strip_comments(f.read())
    except Exception as e:
        print(f'[ERROR]: {e}')
        return ''
