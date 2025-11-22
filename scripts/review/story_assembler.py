
import sys  # Allow this file to import like it was in the "main" folder
sys.path.append(r'C:\Users\ghoop\Desktop\writer')

import json
import os

from lib.ai import ChatLog
from lib.config2 import Config
from lib.util import extract_between_tags


config = Config()

# Simple script to take the last conversation and merge it into a single narrative. Appends to a list of stories in stories.json and then overwrites the book.txt file.

# Load the story in from the output chat log
# This is basically the opposite of 'prompt_plus', dropping all human msgs
story = []
word_count = 0
chat_log = ChatLog.load(config.output_dir, 'data')
for response in chat_log.having_role('AI'):
    story.append(extract_between_tags('prose', response['msg']))
    word_count += len(story[-1].split())
print(f'The latest story contains {word_count} words')

# Create the 'stories.json' file if it doesn't exist already
# TODO: me - Not sure if i want to keep this stoarage approach
STORY_BOOK_PATH=f'./{config.output_dir}/stories.json'
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
TMP_BOOK_PATH=f'./{config.output_dir}/book.txt'
with open(TMP_BOOK_PATH, 'w', encoding='utf-8') as f:
    f.write('\n---\n'.join(story))
