
import re

# Helper to extract xml encoded text sections
# This is useful because llms prefer xml for referential data for some reason
def extract_between_tags(tag, text):
    m = re.match(f"<{tag}>((?:.|[\r\n])*)</{tag}>", text)
    return m and m.group(1) or ""
