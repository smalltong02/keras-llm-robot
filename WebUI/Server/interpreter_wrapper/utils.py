import re
from enum import Enum

class TaskResult(Enum):
    task_processing = 0
    task_success = 1
    task_impossible = 2

def render_message(interpreter, message):
    """
    Renders a dynamic message into a string.
    """
    # Split the message into parts by {{ and }}, including multi-line strings
    parts = re.split(r"({{.*?}})", message, flags=re.DOTALL)

    for i in range(len(parts)):
        part = parts[i]
        # If the part is enclosed in {{ and }}
        if part.startswith("{{") and part.endswith("}}"):
            # Run the code inside the brackets
            output = interpreter.terminal.run(
                "python", part[2:-2].strip()
            )

            # Turn it into just a simple string
            outputs = []
            for line in output:
                if line.get("format") == "output":
                    outputs.append(line["content"])
            output = "\n".join(outputs)

            # Replace the part with the output
            parts[i] = output

    # Join the parts back into the message
    rendered_message = "".join(parts)

    return rendered_message

def extract_markdown_code_blocks(text, languages=None):
    if languages is None:
        pattern = r'```([a-zA-Z]+)\n([\s\S]*?)\n```'
    else:
        pattern = r'```(' + '|'.join(languages) + r')\n([\s\S]*?)\n```'
    code_blocks = re.findall(pattern, text, re.IGNORECASE)
    #code_blocks = [(item[0].strip(), item[1].strip()) for item in code_blocks]
    return code_blocks

def split_with_code_blocks(initial_string, delimiters):
    result_list = []
    start = 0
    index = 0
    if not delimiters:
        result_list.append({"role": "assistant", "type": "message", "content": initial_string[start:]})
        return result_list
    item_lists = ["```" + item[0] + "\n" + item[1] + "\n```" for item in delimiters]
    for item in item_lists:
        index = initial_string.find(item, start)
        if index != -1:
            result_list.append({"role": "assistant", "type": "message", "content": initial_string[start:index]})
            result_list.append({"role": "assistant", "type": "code", "content": item})
            start = index + len(item)
        else:
            result_list.append({"role": "assistant", "type": "message", "content": initial_string[start:]})
            break

    if index != -1:
        result_list.append({"role": "assistant", "type": "message", "content": initial_string[start:]})
    return result_list

def is_task_completion(message):
    if re.search("all tasks done", message, re.IGNORECASE):
        return TaskResult.task_success
    if re.search("task is impossible", message, re.IGNORECASE):
        return TaskResult.task_impossible
    return TaskResult.task_processing