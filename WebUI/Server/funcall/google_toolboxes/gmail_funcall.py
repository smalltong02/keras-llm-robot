import base64
from langchain_core.tools import tool
from googleapiclient.discovery import build

def search_in_gmails(search_criteria: str) ->str:
    """search mail in google email. The parameter 'search_criteria' conforms to Gmail's advanced search syntax format."""
    from WebUI.Server.funcall.google_toolboxes.credential import glob_credentials
    if not glob_credentials:
        return ""
    
    email_messages = ""
    try:
        service = build("gmail", "v1", credentials=glob_credentials)
        if not service:
            return ""
        email_counts = 1

        message_ids = []
        messages = service.users().messages().list(userId="me", q=search_criteria).execute()
        if 'messages' in messages:
            message_ids.extend(messages['messages'])

        if not message_ids:
            return ""

        for message_id in message_ids:
            msg = service.users().messages().get(userId="me", id=message_id['id']).execute()
            payload = msg.get('payload', {})
            headers = payload.get('headers', [])
            
            subject = ""
            for hdr in headers:
                if hdr.get('name', "").lower() == "subject":
                    subject = hdr.get('value', "")
            
            parts = payload.get('parts')
            if not parts:
                continue

            body = ""
            for part in parts:
                data = part.get('body',{}).get('data', "")
                if not data:
                    continue
                if body:
                    body += "\n\n"
                body += base64.urlsafe_b64decode(data).decode('utf-8')

            if body:
                prefix = f"This is email #{email_counts}: \n\n"
                title = f"Subject: {subject} \n\n"
                if email_messages:
                    email_messages += "\n\n"
                email_messages += f"{prefix}{title}{body}"
                email_counts += 1
    except Exception as e:
        email_messages = ""
        print(f"search_gmails Error: {e}")
    return email_messages

# google_funcall_toolboxes = [search_in_gmails]
# tool_names = {
#     "search_in_gmails": search_in_gmails,
# }