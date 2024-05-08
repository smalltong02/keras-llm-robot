import json
import datetime
import geocoder
from langchain_core.tools import tool
from langchain.tools.render import render_text_description
from WebUI.configs.basicconfig import ExtractJsonStrings

@tool
def get_current_location():
    """Get current location information."""
    location = geocoder.ip('me')

    if location:
        return location.address
    else:
        return "Location information not available."

@tool
def get_current_time():
    """Get current time."""
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")   
    return formatted_time

funcall_tools = [get_current_location, get_current_time]
tool_names = {
    "get_current_location": get_current_location,
    "get_current_time": get_current_time,
}

def GetFuncallList() ->list:
    funcall_list = []
    for call_tool in funcall_tools:
        funcall_list.append(call_tool.name)
    return funcall_list

def GetFuncallDescription(func_name: str = "") ->str:
    description = ""
    for call_tool in funcall_tools:
        if func_name == call_tool.name:
            description = call_tool.description
    return description

def GetToolsSystemPrompt() ->str:
    rendered_tools = render_text_description(funcall_tools)
    tools_system_prompt = f"""You can access to the following set of tools. Here are the names and descriptions for each tool:
    {rendered_tools}
    Given the user input, you need to use your own judgment whether to use tools. If not needed, please answer the questions to the best of your ability.
    If tools are needed, return the name and input of the tool to use. Return your response as a JSON blob with 'name' and 'arguments' keys."""
    return tools_system_prompt

def RunFunctionCalling(json_data: str) ->str:
    try:
        func = json.loads(json_data)
        print("func: ", func)
        func_name = func.get("name", "")
        func_arg = func.get("arguments", [])
        if func_name in tool_names:
            result = tool_names[func_name].run(func_arg)
            return result
        else:
            return json_data
    except json.JSONDecodeError:
        return json_data

def split_with_calling_blocks(orgin_string: str, json_lists: list[str]):
    result_list = []
    start = 0
    index = 0
    if not json_lists:
        result_list.append(orgin_string[start:])
        return result_list
    for item in json_lists:
        index = orgin_string.find(item, start)
        if index != -1:
            result_list.append(orgin_string[start:index])
            result = RunFunctionCalling(item)
            result_list.append(result)
            start = index + len(item)
        else:
            result_list.append(orgin_string[start:])
            break

    if index != -1:
        result_list.append(orgin_string[start:])
    return result_list

def use_function_calling(text: str) ->str:
    print("calling_text: ", text)
    json_lists = ExtractJsonStrings(text)
    result_list = split_with_calling_blocks(text, json_lists)
    result_text = ""
    for result in result_list:
        result_text += result
    return result_text
