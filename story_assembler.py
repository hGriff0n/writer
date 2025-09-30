import json
import os
import re
import yaml

# 
# Load config file
#
with open('./data/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Simple script to take the last conversation and merge it into a single narrative. Appends to a list of stories in stories.json and then overwrites the book.txt file.
def extract_between_tags(tag, text):
    m = re.match(f"<{tag}>((?:.|[\r\n])*)</{tag}>", text)
    return m and m.group(1) or ""

# Load the story in from the output chat log
# This is basically the opposite of 'prompt_plus', dropping all human msgs
word_count = 0
with open(f'./{config['output-dir']}/data.json', 'r', encoding='utf-8') as f:
    story = []
    for response in filter(lambda m: m['role'] == 'AI', json.load(f)['conversation']):
        story.append(extract_between_tags('prose', response['msg']))
        word_count += len(story[-1].split())
print(f'The latest story contains {word_count} words')

# Create the 'stories.json' file if it doesn't exist already
# TODO: me - Not sure if i want to keep this stoarage approach
STORY_BOOK_PATH=f'./{config['output-dir']}/stories.json'
if not os.path.exists(STORY_BOOK_PATH):
    with open(STORY_BOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump({}, f, ensure_ascii=False, indent=4)

# Load the existing database of stories so I can add a new one and then save
with open(STORY_BOOK_PATH, 'r', encoding='utf-8') as f:
    stories = json.load(f)
stories[len(stories) + 1] = story
with open(STORY_BOOK_PATH, 'w', encoding='utf-8') as f:    
    json.dump(stories, f, ensure_ascii=False, indent=4)

# Because the saved json file isn't pretty, this also outputs just
# the text to a text file. This is overwritten on every run.
TMP_BOOK_PATH=f'./{config['output-dir']}/book.txt'
with open(TMP_BOOK_PATH, 'w', encoding='utf-8') as f:
    f.write('\n---\n'.join(story))
