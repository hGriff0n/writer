
from dataclasses import dataclass

from lib.config import Config
from lib.ai import LlmEngine

from rich.console import RenderableType
from rich.markdown import Markdown
from textual import events, on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Footer, Input, OptionList, Placeholder, TextArea


# Initialize chat app
config = Config()
model = LlmEngine(config, 'gemini')


# Chat interface
# TODO: me - split this into a more organized structure
# https://rich.readthedocs.io/en/latest/
# https://github.com/qnixsynapse/rich-chat/blob/main/source/rich-chat.py
# https://github.com/textualize/rich-cli

# https://github.com/darrenburns/elia/blob/main/elia_chat/screens/home_screen.py
# https://github.com/darrenburns/elia/blob/main/elia_chat/widgets/chat.py

class Header(Placeholder):
    DEFAULT_CSS = """
    Header {
        height: 3;
        dock: top;
    }
    """


# TODO: me - Add options
# TODO: me - want to take up full width
class Navigation(OptionList):
    DEFAULT_CSS = """
    Navigation {
        width: 100%;
        margin: 0;
        padding: 0;
        height: 100%;
    }
    """
    def __init__(self, id):
        super().__init__("option1", "option2", id=id)


# https://github.com/darrenburns/elia/blob/main/elia_chat/widgets/chat_list.py#L59
class Menu(Vertical):
    DEFAULT_CSS = """
    Menu {
        width: 16;
        height: 100%;
        dock: left;
    }
    """
    def compose(self):
        yield Header(id="title")
        yield Navigation(id="options")


# Display a single message
# Has different formatting for assistant and human
# Copied from elia Chatbox
class Chatbox(Widget, can_focus=True):
    DEFAULT_CSS = """
    Chatbox {
        height: auto;
        width: auto;
        min-width: 12;
        max-width: 100%;
        margin: 0 1;
        padding: 0 2;

        &.assistant-message.response-in-progress {
            background: red 3%;
            min-width: 30%;
        }

        &.assistant-message {
            width: 1fr;
            margin-left: 10;
            border: round red 60%;
        }

        &.human-message {
            border: round green;
            margin-right: 10;
        }
    }
    """
    BINDINGS = []

    def __init__(self, message, role, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.message = message
        self.role = role

    ROLE_HUMAN = "human"
    def Human(message, *args, **kwargs):
        return Chatbox(message, role=Chatbox.ROLE_HUMAN, *args, **kwargs)
    
    def AiResponse(message, *args, **kwargs):
        return Chatbox(message, role="ai", *args, **kwargs)

    def on_mount(self) -> None:
        if self.role != Chatbox.ROLE_HUMAN:
            self.add_class("assistant-message")
            self.border_title = "Agent"
        else:
            self.add_class("human-message")
            self.border_title = "You"

    # Deleted `action_up`
    # Deleted `action_down`
    # Deleted `action_select`
    # Deleted `copy_to_clipboard`
    # Deleted `watch_selection_mode`
    # Deleted `leave_selection_mode`
    # Deleted `watch_has_focus`
    # Deleted `handle_visual_select`

    @property
    def markdown(self) -> Markdown:
        """Return the content as a Rich Markdown object."""
        content = self.message
        if not isinstance(content, str):
            content = ""

        return Markdown(content)

    def render(self) -> RenderableType:
        return self.markdown

    def append_chunk(self, chunk: str) -> None:
        """Append a chunk of text to the end of the message."""
        content = self.message
        if isinstance(content, str):
            content += chunk
            self.message = content
            self.refresh(layout=True)


# Handles getting input from keyboard and initiating llm calls
# TODO: me - arrow key movements not correct
# TODO: me - could use some touchups
class ChatInput(TextArea):
    BINDINGS = [
        Binding("ctrl+j", "submit_prompt", "Send message", key_display="^j")
    ]

    DEFAULT_CSS = """
    ChatInput {
        dock: bottom;
        height: auto;
        max-height: 50%;
        width: 100%;
        
        &.-submit-blocked {
            border: round $error 50%;
        }
    }
    """
    submit_ready = reactive(True)

    @dataclass
    class PromptSubmitted(Message):
        text: str
        prompt_input: "ChatInput"

    # TODO: me - Handling this message is required for
    # undoing focus
    @dataclass
    class CursorEscapingTop(Message):
        pass

    @dataclass
    class CursorEscapingBottom(Message):
        pass

    def __init__(self, *args, **kwargs):
        super().__init__(language="markdown", placeholder=">", *args, **kwargs)

    def on_key(self, event: events.Key) -> None:
        if self.cursor_location == (0, 0) and event.key == "up":
            event.prevent_default()
            # self.post_message(self.CursorEscapingTop())
            event.stop()
        elif self.cursor_at_end_of_text and event.key == "down":
            event.prevent_default()
            # self.post_message(self.CursorEscapingBottom())
            event.stop()

    def watch_submit_ready(self, submit_ready: bool) -> None:
        self.set_class(not submit_ready, "-submit-blocked")

    def on_mount(self):
        self.border_title = "Enter your [u]m[/]essage..."

    @on(TextArea.Changed)
    async def prompt_changed(self, event: TextArea.Changed) -> None:
        text_area = event.text_area
        text_area.set_class(text_area.wrapped_document.height > 1, "multiline")

        # TODO - when the height of the textarea changes
        #  things don't appear to refresh correctly.
        #  I think this may be a Textual bug.
        #  The refresh below should not be required.
        self.parent.refresh()

    def action_submit_prompt(self) -> None:
        if self.text.strip() == "":
            self.notify("Cannot send empty message!")
            return

        if self.submit_ready:
            message = self.PromptSubmitted(self.text, prompt_input=self)
            self.clear()
            self.post_message(message)
        else:
            self.app.bell()
            self.notify("Please wait for response to complete.")


# Analogous to Elia chat
class Chat(Vertical):
    BINDINGS = []
    DEFAULT_CSS = """
    """

    allow_input_submit = reactive(True)

    @property
    def chat_container(self) -> VerticalScroll:
        return self.query_one("#chat-container", VerticalScroll)

    def compose(self):
        yield Header(id="chat-title")
        with VerticalScroll(id="chat-container") as container:
            container.can_focus = False
            yield Chatbox.Human("utf8")
            yield Chatbox.AiResponse("ascii")
        yield ChatInput(id="chat-textbox")

    @property
    def prompt(self) -> ChatInput:
        return self.query_one(ChatInput)

    # Adds the user message from the prompt to the chat
    @on(ChatInput.PromptSubmitted)
    async def user_chat_message_submitted(self, event: ChatInput.PromptSubmitted) -> None:
        if not self.allow_input_submit:
            return
        
        new_message = Chatbox.Human(event.text)

        container = self.chat_container
        await container.mount(new_message)
        # scroll_to_latest_message
        container.refresh()
        container.scroll_end(animate=False, force=True)
        # post event
        # add to history
        self.prompt.submit_ready = False
        self.send_to_llm(event.text)

    # Send message to llm and wait for response
    # TODO: me - This'll need to be more involved given the complex context operations I'm planning
    @work(thread=True, group="agent_response")
    async def send_to_llm(self, message: str) -> None:
        response = Chatbox.AiResponse("")

        container = self.chat_container
        self.app.call_from_thread(container.mount, response)
        # Using `response.loading = True` doesn't work when moving back to the text state
        response.border_title = "Agent responding..."

        ai_resp = model.invoke(message)

        # Because the response has arrived, we restore the border state to indicate that
        response.border_title = "Agent"
        response.append_chunk(ai_resp.content)
        container.scroll_end(animate=False, force=True)
        self.prompt.submit_ready = True

    # Handle model response, not sure how text box appears
    # async def stream_agent_response(self) -> None:
    # @on(AgentResponseStarted + AgentResponseFailed)
    # @on(AgentResponseComplete)
    #Restore state on agent failure
    # def restore_state_on_agent_failure(self, evdnt) -> None:

    # TODO: me - Loading history and naming chats is the key remaining work item
    # Method to load a specific chat context from the sidebar
    # ie. load the chat history (for now) and prompts
    # Elia loads from an sql database
    def load_chat(self) -> None:
        pass


# TODO: me - Loading+saving chats and switching between them via OptionList
class ChatApp(App):
    CSS = """
    #chat-title { height: 2; }
    """
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=False, priority=True)
    ]

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Menu(id="Menu")
            yield Chat(id="Box")
        yield Footer(id="Footer")


# https://chaoticengineer.hashnode.dev/textual-and-chatgpt
app = ChatApp()
app.run()