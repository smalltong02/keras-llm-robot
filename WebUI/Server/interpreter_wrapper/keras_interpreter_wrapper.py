import os
import time
import json
import traceback
import datetime
from enum import Enum
from WebUI.Server.interpreter_wrapper.default_system_message import (default_local_system_message, default_docker_system_message, force_task_completion_message)
from WebUI.Server.utils import fschat_openai_api_address, GetKerasInterpreterConfig
from WebUI.Server.interpreter_wrapper.utils import extract_markdown_code_blocks, split_with_code_blocks
from WebUI.Server.interpreter_wrapper.computer.computer import Computer
from WebUI.Server.interpreter_wrapper.terminal.terminal import Terminal
from WebUI.Server.interpreter_wrapper.local_llm.localllm import LocalLLM

KERAS_CONVERSATION_HISTORY_PATH = "./WebUI/knowledge_base"

class SafeModeType(Enum):
    OffMode = 0
    AskMode = 1
    AutoMode = 2

class BaseInterpreter:
    def __init__(self):
        pass

    def chat(self, query):
        pass

class KerasInterpreter(BaseInterpreter):
    def __init__(self,
        model_name="",
        offline=False,
        auto_run=False,
        verbose=False,
        safe_mode=SafeModeType.OffMode,
        docker_mode=False,
        conversation_history=True,
        llm_base=fschat_openai_api_address(),
        system_message=default_local_system_message,
        custom_instructions="",
        ):
        super().__init__()
        # State
        self.messages = []
        self.responding = False
        self.last_messages_count = 0

        # Settings
        self.offline = offline
        self.auto_run = auto_run
        self.verbose = verbose
        self.ws_client = None
        self.safe_mode = safe_mode
        self.docker_mode = docker_mode
        self.max_output=2800

        # Conversation history
        self.conversation_history = conversation_history
        self.conversation_filename = None
        self.conversation_history_path = KERAS_CONVERSATION_HISTORY_PATH

        # LLM
        self.model_name = model_name
        self.model = LocalLLM(
            model_name=model_name,
            api_base=llm_base,
            streaming=True,
            temperature=0.2,
            max_tokens=1000,
            )

        config = GetKerasInterpreterConfig()
        if config:
            self.interpreter_host = config["host"]
            self.interpreter_port = config["port"]
            config_custom_instructions = config.get("custom_instructions", "")
            if config_custom_instructions:
                custom_instructions = config_custom_instructions
            config_system_message = config.get("system_message", "")
            if config_system_message:
                system_message = config_system_message

        # These are LLM related
        self.system_message = system_message
        self.custom_instructions = custom_instructions
        self.computer = None

        if not self.docker_mode:
            self.computer = Computer()
            self.sync_computer = (
                True  # Sync the interpreter's computer with the user's interpreter.computer
            )
        else:
           if self.system_message == default_local_system_message:
               self.system_message = default_docker_system_message
        self.terminal = self.start_terminal()

    def start_terminal(self):
        return Terminal(host=self.interpreter_host, port=self.interpreter_port, docker_mode=self.docker_mode)

    def chat(self, message=None, stream=False):
        try:
            self.responding = True
            if stream:
                return self._streaming_chat(message=message)

            for _ in self._streaming_chat(message=message):
                pass

            self.responding = False
            return self.messages[self.last_messages_count :]

        except Exception:
            self.responding = False
            raise

    def truncate_output(self, data, max_output_chars=2000):
        needs_truncation = False

        message = f"Output truncated. Showing the last {max_output_chars} characters.\n\n"

        # Remove previous truncation message if it exists
        if data.startswith(message):
            data = data[len(message) :]
            needs_truncation = True

        # If data exceeds max length, truncate it and add message
        if len(data) > max_output_chars or needs_truncation:
            data = message + data[-max_output_chars:]

        return data
    
    def respond(self):
        last_unsupported_code = ""
        insert_force_task_completion_message = False

        while True:
            ## RENDER SYSTEM MESSAGE ##

            system_message = self.system_message

            # Add language-specific system messages
            # for language in self.terminal_config.get("languages", {}):
            #     if hasattr(language, "system_message"):
            #         system_message += "\n\n" + language.system_message

            # Add custom instructions
            if self.custom_instructions:
                system_message += "\n\n" + self.custom_instructions

            # Storing the messages so they're accessible in the interpreter's computer
            #output = self.terminal.run("python", f"messages={self.messages}")

            ## Rendering ↓
            #rendered_system_message = render_message(self, system_message)
            ## Rendering ↑

            rendered_system_message = {
                "role": "system",
                "type": "message",
                "content": system_message,
            }

            # Create the version of messages that we'll send to the LLM
            history = self.messages.copy()
            query = {}
            if history:
                query = history.pop()
            history = [rendered_system_message] + history

            # if insert_force_task_completion_message:
            #     messages_for_llm.append(
            #         {
            #             "role": "user",
            #             "type": "message",
            #             "content": force_task_completion_message,
            #         }
            #     )
            #     # Yield two newlines to seperate the LLMs reply from previous messages.
            #     yield {"role": "assistant", "type": "message", "content": "\n\n"}

            ### RUN THE LLM ###
            model_response = ""
            try:
                
                for chunk in self.model.run(history, query):
                    model_response += chunk
                    # yield {"role": "assistant", "type": "message", "content": chunk + '\n'}
            # Provide extra information on how to change API keys, if we encounter that error
            # (Many people writing GitHub issues were struggling with this)
            except Exception as e:
                print(traceback.format_exc())
                raise Exception(
                    "Error occurred. "
                    + str(e))

            code_blocks = extract_markdown_code_blocks(model_response, self.terminal.get_languages())
            model_response= split_with_code_blocks(model_response, code_blocks)
            ### RUN CODE (if it's there) ###

            index = 0
            if model_response:
                for response in model_response:
                    if response.get("type") == "message":
                        content = response.get("content")
                        yield {
                            "role": "assistant",
                            "type": "message",
                            "format": "output",
                            "content": content,
                        }
                    elif response.get("type") == "code":
                        code_block = code_blocks[index]
                        for trunk in self.terminal.run(code_block[0], code_block[1]):
                            content += trunk
                        yield {
                            "role": "terminal",
                            "type": "code",
                            "format": "output",
                            "content": content,
                        }
                        index += 1
            # if self.messages[-1]["type"] == "code":
            #     if self.verbose:
            #         print("Running code:", self.messages[-1])

            #     try:
            #         # What language/code do you want to run?
            #         language = self.messages[-1]["format"].lower().strip()
            #         code = self.messages[-1]["content"]

            #         if language == "text":
            #             # It does this sometimes just to take notes. Let it, it's useful.
            #             # In the future we should probably not detect this behavior as code at all.
            #             continue

            #         # Is this language enabled/supported?
            #         if language not in self.terminal.get_languages():
            #             output = f"`{language}` disabled or not supported."

            #             yield {
            #                 "role": "computer",
            #                 "type": "console",
            #                 "format": "output",
            #                 "content": output,
            #             }

            #             # Let the response continue so it can deal with the unsupported code in another way. Also prevent looping on the same piece of code.
            #             if code != last_unsupported_code:
            #                 last_unsupported_code = code
            #                 continue
            #             else:
            #                 break

            #         # Yield a message, such that the user can stop code execution if they want to
            #         try:
            #             yield {
            #                 "role": "computer",
            #                 "type": "confirmation",
            #                 "format": "execution",
            #                 "content": {
            #                     "type": "code",
            #                     "format": language,
            #                     "content": code,
            #                 },
            #             }
            #         except GeneratorExit:
            #             # The user might exit here.
            #             # We need to tell python what we (the generator) should do if they exit
            #             break

            #         ## ↓ CODE IS RUN HERE

            #         for line in self.terminal.run(language, code, stream=True):
            #             yield {"role": "computer", **line}

            #         ## ↑ CODE IS RUN HERE

            #         # yield final "active_line" message, as if to say, no more code is running. unlightlight active lines
            #         # (is this a good idea? is this our responsibility? i think so — we're saying what line of code is running! ...?)
            #         yield {
            #             "role": "computer",
            #             "type": "console",
            #             "format": "active_line",
            #             "content": None,
            #         }

            #     except Exception:
            #         yield {
            #             "role": "computer",
            #             "type": "console",
            #             "format": "output",
            #             "content": traceback.format_exc(),
            #         }

            # else:
            #     ## FORCE TASK COMLETION
            #     # This makes it utter specific phrases if it doesn't want to be told to "Proceed."
            #     #if interpreter.os:
            #     #    force_task_completion_message.replace(
            #     #        "If the entire task I asked for is done,",
            #     #        "If the entire task I asked for is done, take a screenshot to verify it's complete, or if you've already taken a screenshot and verified it's complete,",
            #     #    )
            #     force_task_completion_responses = [
            #         "the task is done.",
            #         "the task is impossible.",
            #         "let me know what you'd like to do next.",
            #     ]

            #     if (
            #         self.messages
            #         and not any(
            #             task_status in self.messages[-1].get("content", "").lower()
            #             for task_status in force_task_completion_responses
            #         )
            #     ):
            #         # Remove past force_task_completion messages
            #         self.messages = [
            #             message
            #             for message in self.messages
            #             if message.get("content", "") != force_task_completion_message
            #         ]
            #         # Combine adjacent assistant messages, so hopefully it learns to just keep going!
            #         combined_messages = []
            #         for message in self.messages:
            #             if (
            #                 combined_messages
            #                 and message["role"] == "assistant"
            #                 and combined_messages[-1]["role"] == "assistant"
            #                 and message["type"] == "message"
            #                 and combined_messages[-1]["type"] == "message"
            #             ):
            #                 combined_messages[-1]["content"] += "\n" + message["content"]
            #             else:
            #                 combined_messages.append(message)
            #         self.messages = combined_messages

            #         # Send model the force_task_completion_message:
            #         insert_force_task_completion_message = True

            #         continue

            #     # Doesn't want to run code. We're done!
            #     break

        return

    def _streaming_chat(self, message=None):
        if message is not None:
            if message == "":
                message = "No entry from user - please suggest something to enter."

            if isinstance(message, dict):
                if "role" not in message:
                    message["role"] = "user"
                self.messages.append(message)
            elif isinstance(message, str):
                self.messages.append(
                    {"role": "user", "type": "message", "content": message}
                )
            elif isinstance(message, list):
                self.messages = message

            self.last_messages_count = len(self.messages)

            yield from self._respond_and_store()

            # Save conversation if we've turned conversation_history on
            if self.conversation_history:
                # If it's the first message, set the conversation name
                if not self.conversation_filename:
                    first_few_words = "_".join(
                        self.messages[0]["content"][:25].split(" ")[:-1]
                    )
                    for char in '<>:"/\\|?*!':  # Invalid characters for filenames
                        first_few_words = first_few_words.replace(char, "")

                    date = datetime.now().strftime("%B_%d_%Y_%H-%M-%S")
                    self.conversation_filename = (
                        "__".join([first_few_words, date]) + ".json"
                    )

                # Check if the directory exists, if not, create it
                if not os.path.exists(self.conversation_history_path):
                    os.makedirs(self.conversation_history_path)
                # Write or overwrite the file
                with open(
                    os.path.join(
                        self.conversation_history_path, self.conversation_filename
                    ),
                    "w",
                ) as f:
                    json.dump(self.messages, f)
            return

        raise Exception(
            "`interpreter.chat()` requires a display. Set `display=True` or pass a message into `interpreter.chat(message)`."
        )

    def _respond_and_store(self):
        def is_active_line_chunk(chunk):
            return "format" in chunk and chunk["format"] == "active_line"

        last_flag_base = None

        for chunk in self.respond():
            if chunk["content"] == "":
                continue

            # Handle the special "confirmation" chunk, which neither triggers a flag or creates a message
            if chunk["type"] == "confirmation":
                # Emit a end flag for the last message type, and reset last_flag_base
                if last_flag_base:
                    yield {**last_flag_base, "end": True}
                    last_flag_base = None
                yield chunk
                # We want to append this now, so even if content is never filled, we know that the execution didn't produce output.
                # ... rethink this though.
                self.messages.append(
                    {
                        "role": "computer",
                        "type": "console",
                        "format": "output",
                        "content": "",
                    }
                )
                continue

            # Check if the chunk's role, type, and format (if present) match the last_flag_base
            if (
                last_flag_base
                and "role" in chunk
                and "type" in chunk
                and last_flag_base["role"] == chunk["role"]
                and last_flag_base["type"] == chunk["type"]
                and (
                    "format" not in last_flag_base
                    or (
                        "format" in chunk
                        and chunk["format"] == last_flag_base["format"]
                    )
                )
            ):
                # If they match, append the chunk's content to the current message's content
                # (Except active_line, which shouldn't be stored)
                if not is_active_line_chunk(chunk):
                    self.messages[-1]["content"] += chunk["content"]
            else:
                # If they don't match, yield a end message for the last message type and a start message for the new one
                if last_flag_base:
                    yield {**last_flag_base, "end": True}

                last_flag_base = {"role": chunk["role"], "type": chunk["type"]}

                # Don't add format to type: "console" flags, to accomodate active_line AND output formats
                if "format" in chunk and chunk["type"] != "console":
                    last_flag_base["format"] = chunk["format"]

                yield {**last_flag_base, "start": True}

                # Add the chunk as a new message
                if not is_active_line_chunk(chunk):
                    self.messages.append(chunk)

            # Yield the chunk itself
            yield chunk

            # Truncate output if it's console output
            if chunk["type"] == "console" and chunk["format"] == "output":
                self.messages[-1]["content"] = self.truncate_output(
                    self.messages[-1]["content"], self.max_output
                )

        # Yield a final end flag
        if last_flag_base:
            yield {**last_flag_base, "end": True}

    def wait(self):
        while self.responding:
            time.sleep(0.2)
        return self.messages[self.last_messages_count :]
    
    def reset(self):
        if self.computer is not None:
            self.computer.terminate()  # Terminate computer
        self.computer = None
        self.__init__()