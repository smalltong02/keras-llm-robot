import os
import time
import json
import uuid
import base64
import traceback
from datetime import datetime
from enum import Enum
from pathlib import Path
from WebUI.configs.basicconfig import TMP_DIR
from WebUI.Server.interpreter_wrapper.default_system_message import (default_local_system_message, default_docker_system_message, force_task_completion_message)
from WebUI.Server.utils import fschat_openai_api_address, GetKerasInterpreterConfig
from WebUI.Server.interpreter_wrapper.utils import (TaskResult, extract_markdown_code_blocks, split_with_code_blocks, is_task_completion)
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
        conversation_history=False,
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
            temperature=0.6,
            max_tokens=5000,
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
        system_message = self.system_message
        if self.custom_instructions:
            system_message += "\n\n" + self.custom_instructions

        rendered_system_message = {
                "role": "system",
                "type": "message",
                "content": system_message,
            }
        history = self.messages.copy()
        history = [rendered_system_message] + history

        if history[-1]["role"] != "user":
            yield {
                "role": "assistant",
                "type": "end",
                "format": "output",
                "content": "\n\nThe user has not input any queries.",
            }
            return

        continuous_no_code = 0
        continuous_task_cycles = 0
        query = history.pop()
        while True:
            if not query:
                query = {
                    "role": "user",
                    "type": "message",
                    "content": force_task_completion_message,
                }

            ### RUN THE LLM ###
            model_response = ""
            try:
                for chunk in self.model.run(history, query):
                    model_response += chunk
            except Exception as e:
                print(traceback.format_exc())
                raise Exception(
                    "Error occurred. "
                    + str(e))
            
            history = history + [
                    query,
                    {
                        "role": "assistant",
                        "type": "message",
                        "content": model_response,
                    },
                ]

            code_blocks = extract_markdown_code_blocks(model_response, self.terminal.get_languages())
            model_response_org = model_response
            model_response= split_with_code_blocks(model_response, code_blocks)
            ### RUN CODE (if it's there) ###
            query = {}
            index = 0
            continuous_no_code += 1
            if model_response:
                for response in model_response:
                    if response.get("type") == "message":
                        message_answer = response.get("content")
                        yield {
                            "role": "assistant",
                            "type": "message",
                            "format": "output",
                            "content": "\n" + message_answer,
                        }
                    elif response.get("type") == "code":
                        continuous_no_code = 0
                        code_answer = ""
                        response["content"] = "\n" + response["content"]
                        yield response
                        code_block = code_blocks[index]
                        yield {
                                    "role": "terminal",
                                    "type": "code",
                                    "format": "output",
                                    "content": "\n\nExecution completed, the result is: ",
                                }
                        for trunk in self.terminal.run(code_block[0], code_block[1]):
                            if trunk.startswith("image-data:"):
                                imgpath = str(TMP_DIR / Path(str(uuid.uuid4()) + ".jpg"))
                                decoded_data = base64.b64decode(trunk[len("image-data:"):])
                                with open(imgpath, 'wb') as f:
                                    f.write(decoded_data)
                                yield {
                                    "role": "terminal",
                                    "type": "code",
                                    "format": "output",
                                    "content": f'image-file:{imgpath}',
                                }
                                code_answer += "\nSuccessfully drew a picture for the user.\n"
                            else:
                                code_answer += trunk
                                yield {
                                    "role": "terminal",
                                    "type": "code",
                                    "format": "output",
                                    "content": f'{trunk}',
                                }
                        query = {
                            "role": "user",
                            "type": "message",
                            "content": f'Execution completed, the result is: "{code_answer}", If the entire task I asked for is done, Please repeat the final result again and say exactly **All tasks done!** If it is impossible, say **The task is impossible.**',
                        }
                        index += 1

            task_result = is_task_completion(model_response_org)
            if task_result == TaskResult.task_success:
                yield {
                    "role": "assistant",
                    "type": "end",
                    "format": "output",
                    "content": "\n\nAll tasks done!",
                }
                break
            elif task_result == TaskResult.task_impossible:
                yield {
                    "role": "assistant",
                    "type": "end",
                    "format": "output",
                    "content": "\n\nTask is impossible!",
                }
                break
            continuous_task_cycles += 1

            if continuous_no_code >= 5 or continuous_task_cycles >= 20:
                yield {
                    "role": "assistant",
                    "type": "end",
                    "format": "output",
                    "content": "\n\n" + "Task is impossible!",
                }
                break
            yield {
                "role": "assistant",
                "type": "message",
                "format": "output",
                "content": "\n\n",
            }
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
        for chunk in self.respond():
            yield chunk

    def wait(self):
        while self.responding:
            time.sleep(0.2)
        return self.messages[self.last_messages_count :]
    
    def reset(self):
        if self.computer is not None:
            self.computer.terminate()  # Terminate computer
        self.computer = None
        self.__init__()