import datetime
import geocoder
from langchain_core.tools import tool

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