from datetime import datetime
from dateutil.tz import gettz
from langchain_core.tools import tool
from googleapiclient.discovery import build
import google.generativeai as genai

DEFAULT_MAX_EVENTS = 10

CALENDAR_READONLY_SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]
CALENDAR_FULL_SCOPES = ["https://www.googleapis.com/auth/calendar"]


def convert_time_to_rfc3339_time(time_str:str) ->str:
    if not time_str:
        return ""
    import re
    pattern_time = re.compile(r'^\d{2}:\d{2}:\d{2}$')
    pattern_datetime = re.compile(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$')
    pattern_datetime_tz = re.compile(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}-\d{2}:\d{2}$')

    local_timezone = gettz()
    if pattern_datetime_tz.match(time_str):
        return pattern_datetime_tz
    if pattern_datetime.match(time_str):
        time_format = "%Y-%m-%dT%H:%M:%S"
        datetime_obj = datetime.strptime(time_str, time_format)
        combined_datetime = datetime_obj
    elif pattern_time.match(time_str):
        time_format = "%H:%M:%S"
        time_obj = datetime.strptime(time_str, time_format).time()
        today_date = datetime.today().date()
        combined_datetime = datetime.combine(today_date, time_obj)

    time_with_timezone = combined_datetime.replace(tzinfo=local_timezone)
    return time_with_timezone.isoformat()

def get_event_from_gcalendar(start_time:str, end_time:str) ->str:
    """get event from google calendar."""

    from WebUI.Server.funcall.google_toolboxes.credential import glob_credentials
    if not glob_credentials:
        from WebUI.Server.funcall.google_toolboxes.credential import init_credential
        init_credential()
        from WebUI.Server.funcall.google_toolboxes.credential import glob_credentials
        if not glob_credentials:
            return "Credentials not found. This error is unrecoverable."

    calendar_message = ""
    try:
        service = build("calendar", "v3", credentials=glob_credentials)
        if not service:
            return "Calendar object create failed. This error is unrecoverable."
        event_counts = 1
        rfc3339_start = convert_time_to_rfc3339_time(start_time)
        rfc3339_end = convert_time_to_rfc3339_time(end_time)
        if not rfc3339_start or not rfc3339_end:
            return "Invalid time format. Please check the time format."

        # Call the Calendar API
        events_result = (
            service.events()
            .list(
                calendarId="primary",
                timeMin=rfc3339_start,
                timeMax=rfc3339_end,
                maxResults=DEFAULT_MAX_EVENTS,
                singleEvents=True,
                orderBy="startTime",
            )
            .execute()
        )
        events = events_result.get("items", [])

        if not events:
            calendar_message = "No events found from calendar."
            return calendar_message

        # Prints the start and name of the next 10 events
        for event in events:
            start = event["start"].get("dateTime", event["start"].get("date"))
            end = event["end"].get("dateTime", event["end"].get("date"))
            htmlLink = event.get("htmlLink")
            summary = event["summary"]

            prefix = f"This is event #{event_counts}: \n"
            start = f"Start Time: {start} \n"
            end = f"End Time: {end} \n"
            summary = f"Summary: {summary} \n"
            htmlLink = f"htmlLink: {htmlLink} \n"
            if calendar_message:
                calendar_message += "\n\n"
            calendar_message += f"{prefix}{start}{end}{summary}{htmlLink}"
            event_counts += 1
    except Exception as e:
        calendar_message = f"get event from calendar failed, error: {e}"
        print(f"get_event_from_gcalendar Error: {e}")
    return calendar_message

def create_event_to_gcalendar(summary: str, description: str, start_time: str, end_time: str) ->str:
    """create event to google calendar."""
    
    from WebUI.Server.funcall.google_toolboxes.credential import glob_credentials
    if not glob_credentials:
        from WebUI.Server.funcall.google_toolboxes.credential import init_credential
        init_credential()
        from WebUI.Server.funcall.google_toolboxes.credential import glob_credentials
        if not glob_credentials:
            return "Credentials not found. This error is unrecoverable."
    
    calendar_message = ""
    try:
        service = build("calendar", "v3", credentials=glob_credentials)
        if not service:
            return "Calendar object create failed. This error is unrecoverable."
        new_event = {
            "summary": summary,
            "description": description,
            "start": {
                "dateTime": convert_time_to_rfc3339_time(start_time),
            },
            "end": {
                "dateTime": convert_time_to_rfc3339_time(end_time),
            }
        }
        event = service.events().insert(calendarId='primary', body=new_event).execute()
        calendar_message = f"Event created success, Please refer to the following link: \n\n {event.get('htmlLink')}"
    except Exception as e:
        calendar_message = f"create event to calendar failed, error: {e}"
        print(f"create_event_to_calendar Error: {e}")
        
    return calendar_message

@tool
def get_event_from_calendar(start_time:str, end_time:str) ->str:
    """get event from calendar.
        Here is an example of calling the function 'get_event_from_calendar':
            User: Please check today's schedule and report my availability for today.
            Bot: Okay, I will call the function 'get_event_from_calendar' to get your schedule today.
            {
                "name": "get_event_from_calendar",
                "arguments": {
                    "start_time": "08:00:00",
                    "end_time": "18:00:00",
                }
            }
            API Output: "
            This is event #1: 
                Start Time: 2024-05-25T08:00:00-07:00
                End Time: 2024-05-25T11:00:00.000000-07:00
                Summary: Meeting
                htmlLink: https://calendar.google.com/calendar/event?eid=cGV0ZXJ2aW5nMjAyMzAwMjlAZ21haWwuY29t

            This is event #2:
                Start Time: 2024-05-25T15:00:00-07:00
                End Time: 2024-05-25T17:00:00-07:00
                Summary: Interview
                htmlLink: https://calendar.google.com/calendar/event?eid=cGV0ZXJ2aW5nMjAyMzAwMjlAZ21haWwuY29t
            "
            Bot: According to today's schedule, your free time is from 2024-05-25T11:00:00.000000-07:00 to 2024-05-25T15:00:00.000000-07:00, for 4 hours; and from 2024-05-25T17:00:00.000000-07:00 to 2024-05-25T18:00:00.000000-07:00, for 1 hour.
    """
    return get_event_from_gcalendar(start_time, end_time)

@tool
def create_event_to_calendar(summary: str, description: str, start_time: str, end_time: str) ->str:
    """create event to calendar.
        Here is an example of calling the function 'create_event_to_calendar':
            User: Please add an appointment reminder in Calendar. There is an optometry appointment on June 1st at 11:00 AM, I will have a dental cleaning and an examination.
            Bot: Okay, I will call the function 'get_event_from_calendar' to get your schedule on June 1st.
            {
                "name": "get_event_from_calendar",
                "arguments": {
                    "start_time": "2024-06-01T08:00:00",
                    "end_time": "2024-06-01T18:00:00",
                }
            }
            API Output: "
            This is event #1: 
                Start Time: 2024-06-01T13:00:00.000000-07:00
                End Time: 2024-06-01T15:00:00.000000-07:00
                Summary: Attend a gathering
                htmlLink: https://calendar.google.com/calendar/event?eid=cGV0ZXJ5aW8nMjAyMzAwMjlAZ16haWwuY34t
            Bot: According to schedule, You have 2 hours window of free time after 11:00, so you can schedule something during that time. I will call the function 'create_event_to_calendar' to create this event to your calendar.
            {
                "name": "create_event_to_calendar",
                "arguments": {
                    "summary": "optometry appointment",
                    "description": "Dental cleaning and examination",
                    "start_time": "2024-06-01T11:00:00",
                    "end_time": "2024-06-01T13:00:00"
                }
            }
            API Output: "Event created success, Please refer to the following link: https://calendar.google.com/calendar/event?eid=cGV0ZXJ5aW8nMjAyMzAwMjlAZ16haWwuY34t"
            Bot: New event created successfully. You can check it through this link: https://calendar.google.com/calendar/event?eid=cGV0ZXJ5aW8nMjAyMzAwMjlAZ16haWwuY34t
    """
    return create_event_to_gcalendar(summary, description, start_time, end_time)

calendar_toolboxes = [get_event_from_calendar, create_event_to_calendar]
calendar_tool_names = {
    "get_event_from_calendar": get_event_from_calendar,
    "create_event_to_calendar": create_event_to_calendar,
}

# for google gemini
get_event_from_calendar_func = genai.protos.Tool(
    function_declarations=[
      genai.protos.FunctionDeclaration(
        name='get_event_from_calendar',
        description="get event from calendar.",
        parameters=genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                'start_time':genai.protos.Schema(type=genai.protos.Type.STRING, description="This is start time. If the user provides a date and time, then the format “2024-06-01T08:00:00” can be used. If only the time is provided, but the date is today, then the format “08:00:00” can be used. If the date or time are not available, you need to ask the user to provide them."),
                'end_time':genai.protos.Schema(type=genai.protos.Type.STRING, description="This is end time. If the user provides a date and time, then the format “2024-06-01T08:00:00” can be used. If only the time is provided, but the date is today, then the format “08:00:00” can be used. If the date or time are not available, you need to ask the user to provide them."),
            },
            required=['start_time', 'end_time']
        )
      )
    ])

create_event_to_calendar_func = genai.protos.Tool(
    function_declarations=[
      genai.protos.FunctionDeclaration(
        name='create_event_to_calendar',
        description="create a new event reminder to calendar.",
        parameters=genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                'summary':genai.protos.Schema(type=genai.protos.Type.STRING, description="Title of the event reminder."),
                'description':genai.protos.Schema(type=genai.protos.Type.STRING, description="Brief description of the event reminder."),
                'start_time':genai.protos.Schema(type=genai.protos.Type.STRING, description="This is start time. If the user provides a date and time, then the format “2024-06-01T08:00:00” can be used. If only the time is provided, but the date is today, then the format “08:00:00” can be used. If the date or time are not available, you need to ask the user to provide them."),
                'end_time':genai.protos.Schema(type=genai.protos.Type.STRING, description="This is end time. If the user provides a date and time, then the format “2024-06-01T08:00:00” can be used. If only the time is provided, but the date is today, then the format “08:00:00” can be used. If the date or time are not available, you need to ask the user to provide them."),
            },
            required=['summary', 'description', 'start_time', 'end_time']
        )
      )
    ])

google_calendar_tools = [
        get_event_from_calendar_func,
        create_event_to_calendar_func,
    ]

def GetCalendarFuncallList() ->list:
    funcall_list = []
    for call_tool in calendar_toolboxes:
        funcall_list.append(call_tool.name)
    return funcall_list

def GetCalendarFuncallDescription(func_name: str = "") ->str:
    description = ""
    for call_tool in calendar_toolboxes:
        if func_name == call_tool.name:
            description = call_tool.description
    return description

def is_calendar_enable() ->bool:
    from WebUI.configs.webuiconfig import InnerJsonConfigWebUIParse
    configinst = InnerJsonConfigWebUIParse()
    tool_boxes = configinst.get("ToolBoxes")
    if not tool_boxes:
        return False
    google_toolboxes = tool_boxes.get("Google ToolBoxes")
    if not google_toolboxes:
        return False
    calendar = google_toolboxes.get("Tools").get("Google Calendar")
    enable = calendar.get("enable", False)
    return enable