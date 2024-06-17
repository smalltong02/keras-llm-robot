import json
import os.path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from WebUI.configs import (GOOGLE_OAUTH_PORT)
from WebUI.Server.funcall.google_toolboxes.calendar_funcall import CALENDAR_FULL_SCOPES, is_calendar_enable, calendar_toolboxes, calendar_tool_names
from WebUI.Server.funcall.google_toolboxes.gmail_funcall import GMAIL_FULL_SCOPES, is_email_enable, email_toolboxes, email_tool_names
from WebUI.Server.funcall.google_toolboxes.gmap_funcall import is_map_enable, map_toolboxes, map_tool_names
from WebUI.Server.funcall.google_toolboxes.gcloud_funcall import DRIVE_FULL_SCOPES, is_cloud_storage_enable, drive_toolboxes, drive_tool_names
from WebUI.Server.funcall.google_toolboxes.youtube_funcall import YOUTUBE_FULL_SCOPES, is_youtube_enable, youtube_toolboxes, youtube_tool_names

GOOGLE_TOKEN_FILE = "google_token.json"

glob_credentials = None

def init_credential() -> bool:
    from WebUI.configs.basicconfig import GetCredentialsPath
    creds = None
    global glob_credentials
    if glob_credentials:
        return True
    credentials_path = GetCredentialsPath()
    if not credentials_path:
        return False
    scopes = []
    scopes.append(CALENDAR_FULL_SCOPES[0])
    scopes.append(DRIVE_FULL_SCOPES[0])
    scopes.append(GMAIL_FULL_SCOPES[0])
    scopes.append(YOUTUBE_FULL_SCOPES[0])
    if os.path.exists(GOOGLE_TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(GOOGLE_TOKEN_FILE, scopes)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                credentials_path, scopes
            )
            creds = flow.run_local_server(port=GOOGLE_OAUTH_PORT)
        if creds:
            glob_credentials = creds
            return True
        return False
    glob_credentials = creds
    return True

def GetFuncallInToolBoxesList() ->list:
    funcall_list = []
    
    if is_calendar_enable():
        for call_tool in calendar_toolboxes:
            funcall_list.append(call_tool.name)
    if is_email_enable():
        for call_tool in email_toolboxes:
            funcall_list.append(call_tool.name)
    if is_map_enable():
        for call_tool in map_toolboxes:
            funcall_list.append(call_tool.name)
    if is_cloud_storage_enable():
        for call_tool in drive_toolboxes:
            funcall_list.append(call_tool.name)
    if is_youtube_enable():
        for call_tool in youtube_toolboxes:
            funcall_list.append(call_tool.name)
    return funcall_list

def GetFuncallInToolBoxesDescription(func_name: str = "") ->str:
    if is_calendar_enable():
        for call_tool in calendar_toolboxes:
            if func_name == call_tool.name:
                return call_tool.description
    if is_email_enable():
        for call_tool in email_toolboxes:
            if func_name == call_tool.name:
                return call_tool.description
    if is_map_enable():
        for call_tool in map_toolboxes:
            if func_name == call_tool.name:
                return call_tool.description
    if is_cloud_storage_enable():
        for call_tool in drive_toolboxes:
            if func_name == call_tool.name:
                return call_tool.description
    if is_youtube_enable():
        for call_tool in youtube_toolboxes:
            if func_name == call_tool.name:
                return call_tool.description
    return ""

def GetFuncallInToolBoxesName(json_data: str) ->str:
    try:
        func = json.loads(json_data)
        func_name = func.get("name", "")
        if is_calendar_enable():
            for call_tool in calendar_toolboxes:
                if func_name == call_tool.name:
                    return func_name
        if is_email_enable():
            for call_tool in email_toolboxes:
                if func_name == call_tool.name:
                    return func_name
        if is_map_enable():
            for call_tool in map_toolboxes:
                if func_name == call_tool.name:
                    return func_name
        if is_cloud_storage_enable():
            for call_tool in drive_toolboxes:
                if func_name == call_tool.name:
                    return func_name    
        if is_youtube_enable():
            for call_tool in youtube_toolboxes:
                if func_name == call_tool.name:
                    return func_name
    except json.JSONDecodeError:
        return ""
    
def RunFunctionCallingInToolBoxes(json_data: str):
    try:
        func = json.loads(json_data)
        func_name = func.get("name", "")
        func_arg = func.get("arguments", [])

        if func_name in calendar_tool_names:
            result = calendar_tool_names[func_name].run(func_arg)
            return func_name, result, {}
        if func_name in email_tool_names:
            result = email_tool_names[func_name].run(func_arg)
            return func_name, result, {}
        if func_name in map_tool_names:
            result, map_dict = map_tool_names[func_name].run(func_arg)
            return func_name, result, map_dict
        if func_name in drive_tool_names:
            result = drive_tool_names[func_name].run(func_arg)
            return func_name, result, {}
        if func_name in youtube_tool_names:
            result = youtube_tool_names[func_name].run(func_arg)
            return func_name, result, {}
        return "", "", {}
    except json.JSONDecodeError:
        return "", "", {}