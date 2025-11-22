
from typing import Dict, Set, List
import regex as re
from io import StringIO

def _make_bookmark_regex(bookmarks):
    options = "|".join(bookmarks)
    return (
        re.compile(f"""<!-- START_({options}) -->""", re.DOTALL),
        re.compile(f"""<!-- END_({options}) -->""", re.DOTALL)
    )

# TODO: me - this needs to handle more bookmark types
STRT, END = _make_bookmark_regex(['RULE_INVESTIGATION', 'LENS_MAPPING'])
EXTRACT_TAG = re.compile(r"<(?P<tag>\w+)>(.*?)</(?P=tag)>", re.DOTALL)


# TODO: me - This should work on the langchain message array
# TODO: me - This technically doesn't handle architect_sidebar
class ContextManager(object):
    """
    """
    _bookmark_register: Dict[re.Pattern, int] = {}
    _context: List[str] = []
    _living_doc: Dict[str, Set[str]] = {}

    def __init__(self, msgs: List[str] = None, has_action_tags: bool = False):
        self._context = msgs or []
        self._action_tags = has_action_tags

    def _revert_to_investigation_start(self, bookmark: re.Pattern):
        idx = self._bookmark_register[bookmark]
        del self._bookmark_register[bookmark]
        while len(self._context) > idx:
            self._context.pop()

    def _update_internal(self, msg: str):
        for tag, update in EXTRACT_TAG.findall(msg):
            if tag not in self._living_doc:
                self._living_doc[tag] = set()
            if self._action_tags:
                for action, content in EXTRACT_TAG.findall(update.strip()):
                    match action:
                        case 'add': self._living_doc[tag].add(content)
                        case 'remove': self._living_doc[tag].remove(content)
            else:
                self._living_doc[tag].add(update.strip())
        self._context.append(msg)

    def _parse_bookmarks(self, msg: str, pattern: re.Pattern) -> List[str]:
        dtct = pattern.split(msg)
        if len(dtct) != 3:
            return [None, None, msg]
        return dtct
    
    def clear(self):
        self._context.clear()
        self._bookmark_register.clear()

    def add_to_context(self, msg: str):
        _, ebook, middle = self._parse_bookmarks(msg, END)
        middle, sbook, postfix = self._parse_bookmarks(middle, STRT)

        # If the end bookmark is set and we have a known start idx for that
        # bookmark type, revert the context to the state before that start idx.
        # Then push the middle part of the prompt on the stack iff set
        if ebook in self._bookmark_register:
            self._revert_to_investigation_start(ebook)
            if middle is not None:
                self._update_internal(middle.strip())

        # If we are not currently investigating this bookmark, record
        # the investigation start point so we can revert here later
        if sbook and sbook not in self._bookmark_register:
            self._bookmark_register[sbook] = len(self._context)
        
        self._update_internal(postfix.strip())
    
    @property
    def context(self) -> List[str]:
        return self._context

    @property
    def living_doc(self) -> Dict[str, Set[str]]:
        return self._living_doc
    
    def assemble_snapshot(self) -> str:
        doc = StringIO()
        for key, aspects in self._living_doc.items():
            doc.write(f'### {' '.join(k.capitalize() for k in key.split('_'))}')
            for component in aspects:
                doc.write('\n\n')
                doc.write(component)
            doc.write('\n\n')
        return doc.getvalue()