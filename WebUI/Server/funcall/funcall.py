import json
import datetime
import geocoder
from langchain_core.tools import tool
#from WebUI.configs.basicconfig import ExtractJsonStrings

@tool
def get_current_location():
    """Get current location information.
        Here is an example of calling the function 'get_current_location':
            User: Where am I?
            Bot: Okay, I will call the function 'get_current_location' to help you get current location.
            {
                "name": get_current_location,
                "arguments": {}
            }
            API Output: "123 Main St, City, State, ZIP"
            Bot: You are currently located at 123 Main St, City, State, ZIP.
    """
    location = geocoder.ip('me')

    if location:
        return location.address
    else:
        return "Location information not available."

@tool
def get_current_time():
    """Get current time.
    Here is an example of calling the function 'get_current_time':
        User: Do you know the current time?
        Bot: Okay, I will call the function 'get_current_time' to help you get current time.
        {
            "name": get_current_time,
            "arguments": {}
        }
        API Output: 2024-01-01 12:30:05
        Bot: The current time is 2024-01-01 12:30:05
    """
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")   
    return formatted_time

@tool
def submit_warranty_claim(caption: str, description: str):
    """Submit warranty claim.
    Here is an example of calling the function 'submit_warranty_claim':
        User: My network adapter is broken. Please help me submit a repair request.
        Bot: Sure, please provide the following information:
            1. Your full name.
            2. Email address.
            3. The product serial number.
            4. Date of purchase.
            5. A brief description of the issue.
        User: My name is John Doe. My email address is johndoe@example.com. The product serial number is XYZ123456789. The date of purchase was on January 1, 2022. The issue is that the network adapter is not working properly.
        Bot: Thank you for providing the necessary information. Please wait while we submit your repair request.
            {
                "name": submit_warranty_claim,
                "arguments": {
                    "caption": "My network adapter is broken.",
                    "description": "Customer: John Doe\nEmail: johndoe@example.com\nSerial number: XYZ123456789\nPurchase date: January 1, 2022\nIssue: The network adapter is not working properly.\nAdditional information: None\nWarranty expiration date: None\n"
                }
            }
        API Output: Submitted warranty success. The repair number is: 84905783
        Bot: Hi John Doe, your repair request has been submitted successfully. Your repair number is 84905783. We will get back to you soon.
    """
    warranty_context = f"""
     Title: {caption}

     Description: {description}

     Sent email to support@sony.com
"""
    import random
    repair_number = str(random.randint(100000, 999999))
    return f"""Submitted warranty success. The repair number is: {repair_number}\n
    The content of the email -
    {warranty_context}
    """

funcall_tools = [get_current_location, get_current_time, submit_warranty_claim]
tool_names = {
    "get_current_location": get_current_location,
    "get_current_time": get_current_time,
    "submit_warranty_claim": submit_warranty_claim,
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

def GetFuncallName(json_data: str) ->str:
    try:
        func = json.loads(json_data)
        func_name = func.get("name", "")
        for call_tool in funcall_tools:
            if func_name == call_tool.name:
                return func_name
    except json.JSONDecodeError:
        return ""
    
def RunNormalFunctionCalling(json_data: str) ->str:
    try:
        func = json.loads(json_data)
        print("func: ", func)
        func_name = func.get("name", "")
        func_arg = func.get("arguments", {})
        if func_name in tool_names:
            result = tool_names[func_name].run(func_arg)
            return func_name, result
        else:
            return "", ""
    except json.JSONDecodeError:
        return "", ""
