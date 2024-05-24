from langchain_core.tools import tool
from googleapiclient.discovery import build

DEFAULT_MAX_EVENTS = 10

CALENDAR_READONLY_SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]
CALENDAR_FULL_SCOPES = ["https://www.googleapis.com/auth/calendar"]

def get_event_from_gcalendar(start_time:str, end_time:str) ->str:
    """get event from google calendar."""

    from WebUI.Server.funcall.google_toolboxes.credential import glob_credentials
    if not glob_credentials:
        return "Credentials not found. This error is unrecoverable."

    calendar_message = ""
    try:
        service = build("calendar", "v3", credentials=glob_credentials)
        if not service:
            return "Calendar object create failed. This error is unrecoverable."
        event_counts = 1

        # Call the Calendar API
        events_result = (
            service.events()
            .list(
                calendarId="primary",
                timeMin=start_time,
                timeMax=end_time,
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

def create_event_to_gcalendar(new_event: str) ->str:
    """create event to google calendar."""
    
    import json
    from WebUI.Server.funcall.google_toolboxes.credential import glob_credentials
    if not glob_credentials:
        return "Credentials not found. This error is unrecoverable."
    
    calendar_message = ""
    try:
        event_json = json.loads(new_event)
        if "summary" not in event_json:
            return "error: Event summary not found."
        if "start" not in event_json:
            return "error: Event start time not found."
        if "end" not in event_json:
            return "error: Event end time not found."
        
        service = build("calendar", "v3", credentials=glob_credentials)
        if not service:
            return "Calendar object create failed. This error is unrecoverable."
        
        event = service.events().insert(calendarId='primary', body=event_json).execute()
        calendar_message = f"Event created success, Please refer to the following link: \n\n {event.get('htmlLink')}"
    except Exception as e:
        calendar_message = f"create event to calendar failed, error: {e}"
        print(f"create_event_to_calendar Error: {e}")
        
    return calendar_message

@tool
def get_event_from_calendar(start_time:str, end_time:str) ->str:
    """get event from calendar."""
    return get_event_from_gcalendar(start_time, end_time)

@tool
def create_event_to_calendar(new_event: str) ->str:
    """create event to calendar."""
    return create_event_to_gcalendar(new_event)

# calendar_toolboxes = [get_event_from_calendar, create_event_to_calendar]
# calendar_tool_names = {
#     "get_event_from_calendar": get_event_from_calendar,
#     "create_event_to_calendar": create_event_to_calendar,
# }