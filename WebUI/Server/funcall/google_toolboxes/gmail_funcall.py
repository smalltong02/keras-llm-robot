import os
import base64
import mimetypes
from langchain_core.tools import tool
from googleapiclient.discovery import build
import google.generativeai as genai

GMAIL_READONLY_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
GMAIL_MODIFY_SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]
GMAIL_FULL_SCOPES = ["https://mail.google.com/"]

def search_in_gmails(search_criteria: str) ->str:
    """search mail in gmail. The parameter 'search_criteria' conforms to Gmail's advanced search syntax format."""
    from WebUI.Server.funcall.google_toolboxes.credential import glob_credentials
    if not glob_credentials:
        from WebUI.Server.funcall.google_toolboxes.credential import init_credential
        init_credential()
        from WebUI.Server.funcall.google_toolboxes.credential import glob_credentials
        if not glob_credentials:
            return "Credentials not found. This error is unrecoverable."
    
    email_messages = ""
    try:
        service = build("gmail", "v1", credentials=glob_credentials)
        if not service:
            return "email object create failed. This error is unrecoverable."
        email_counts = 1

        message_ids = []
        messages = service.users().messages().list(userId="me", q=search_criteria).execute()
        if 'messages' in messages:
            message_ids.extend(messages['messages'])

        if not message_ids:
            return "Search email successful, but no emails matching the criteria were found."

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
                prefix = f"This is email #{email_counts}: \n"
                title = f"Subject: {subject} \n"
                if email_messages:
                    email_messages += "\n\n"
                email_messages += f"{prefix}{title}{body}"
                email_counts += 1
    except Exception as e:
        email_messages = f"Search email failed, error: {e}"
        print(f"search_in_gmails Error: {e}")
    return email_messages

def build_file_part(file):
    """Creates a MIME part for a file.

    Args:
        file: The path to the file to be attached.

    Returns:
        A MIME part that can be attached to a message.
    """
    from email.mime.audio import MIMEAudio
    from email.mime.base import MIMEBase
    from email.mime.image import MIMEImage
    from email.mime.text import MIMEText

    content_type, encoding = mimetypes.guess_type(file)

    if content_type is None or encoding is not None:
        content_type = "application/octet-stream"
    main_type, sub_type = content_type.split("/", 1)
    if main_type == "text":
        with open(file, "rb"):
            msg = MIMEText("r", _subtype=sub_type)
    elif main_type == "image":
        with open(file, "rb"):
            msg = MIMEImage("r", _subtype=sub_type)
    elif main_type == "audio":
        with open(file, "rb"):
            msg = MIMEAudio("r", _subtype=sub_type)
    else:
        with open(file, "rb") as fp:
            msg = MIMEBase(main_type, sub_type)
            msg.set_payload(fp.read())
    filename = os.path.basename(file)
    msg.add_header("Content-Disposition", "attachment", filename=filename)
    return msg

def create_draft_in_gmails(subject: str, body: str, to_address: str, from_address: str, attachment_file: str) ->str:
    """create draft in gmail."""
    from email.message import EmailMessage
    from WebUI.Server.funcall.google_toolboxes.credential import glob_credentials
    if not glob_credentials:
        from WebUI.Server.funcall.google_toolboxes.credential import init_credential
        init_credential()
        from WebUI.Server.funcall.google_toolboxes.credential import glob_credentials
        if not glob_credentials:
            return "Credentials not found. This error is unrecoverable."
    
    if not to_address or not from_address:
        return "Call failed, This function call must include the 'to' and 'from' parameters."

    email_messages = ""
    try:
        service = build("gmail", "v1", credentials=glob_credentials)

        message = EmailMessage()

        message.set_content(body)

        message["To"] = to_address
        message["From"] = from_address
        message["Subject"] = subject

        # attachment
        attachment_part = build_file_part(attachment_file)
        message.add_attachment(
            attachment_part.get_payload(decode=True),
            maintype=attachment_part.get_content_maintype(),
            subtype=attachment_part.get_content_subtype(),
            filename=attachment_part.get_filename()
        )

        # encoded message
        encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()

        create_message = {"message": {"raw": encoded_message}}
        # pylint: disable=E1101
        draft = (
            service.users()
            .drafts()
            .create(userId="me", body=create_message)
            .execute()
        )
        email_messages = f"Draft create successful, Draft id: `{draft['id']}`\nDraft message: `{draft['message']}`"
    except Exception as e:
        email_messages = f"Draft create failed, error: {e}"
        print(f"create_draft_in_gmails Error: {e}")
    return email_messages

def gmail_send_mail(subject: str, body: str, to_address: str, from_address: str, attachment_file: str)->bool:
    """send email in gmail."""
    from email.message import EmailMessage
    from WebUI.Server.funcall.google_toolboxes.credential import glob_credentials
    if not glob_credentials:
        from WebUI.Server.funcall.google_toolboxes.credential import init_credential
        init_credential()
        from WebUI.Server.funcall.google_toolboxes.credential import glob_credentials
        if not glob_credentials:
            return "Credentials not found. This error is unrecoverable."
    
    if not to_address or not from_address:
        return "Call failed, This function call must include the 'to' and 'from' parameters."

    email_messages = ""
    try:
        service = build("gmail", "v1", credentials=glob_credentials)

        message = EmailMessage()
        message.set_content(body)

        message["To"] = to_address
        message["From"] = from_address
        message["Subject"] = subject

        # attachment
        if os.path.exists(attachment_file):
            attachment_part = build_file_part(attachment_file)
            message.add_attachment(
                attachment_part.get_payload(decode=True),
                maintype=attachment_part.get_content_maintype(),
                subtype=attachment_part.get_content_subtype(),
                filename=attachment_part.get_filename()
            )

        # encoded message
        encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()

        create_message = {"raw": encoded_message}
        # pylint: disable=E1101
        send_message = (
            service.users()
            .messages()
            .send(userId="me", body=create_message)
            .execute()
        )
        id = send_message.get("id")
        email_messages = f"Email create successful, id: `{id}`"
    except Exception as e:
        email_messages = f"Email create failed, error: {e}"
        print(f"create_draft_in_gmails Error: {e}")

    return email_messages

@tool
def search_in_emails(search_criteria: str) ->str:
    """search mail in email. The parameter 'search_criteria' conforms to email's advanced search syntax format.
        Here is an example of calling the function 'search_in_emails', Please use JSON blob:
            User: Please check my email for messages with the subject containing "HipsHook Project" sent on May 30th, 2024.
            Bot: Okay, I will call the function 'search_in_emails' to help you find the email.
            {
                "name": "search_in_emails",
                "arguments": {
                    "search_criteria": "subject: 'HipsHook Project' AND after:2024/05/30"
                }
            }
            API Output: "This is email #1:\nSubject: HipsHook Project Introduction \nSummary: This is a brief introduction to the HipsHook Project."
            Bot: An email has been found: 
                Subject: HipsHook Project Introduction
                Summary: This is a brief introduction to the HipsHook Project.
    """
    return search_in_gmails(search_criteria)

@tool
def create_draft_in_emails(subject: str, body: str, to_address: str, from_address: str, attachment_file: str) ->bool:
    """create draft mail in email.
        Here is an example of calling the function 'create_draft_in_emails', Please use JSON blob:
            User: Please help me create an email draft with the following details:
                Subject: Happy Birthday
                Body: Hi Lyn, happy birthday to you!
                To: dave@gmail.com
                From: lyn@gmail.com
                attachment_file: "./gift_card.png"
            Bot: Okay, I will call the function 'create_draft_in_emails' to help you create an email draft.
            {
                "name": "create_draft_in_emails",
                "arguments": {
                    "subject": "Happy Birthday",
                    "body": "Hi Lyn, happy birthday to you!",
                    "to_address": "dave@gmail.com",
                    "from_address": "lyn@gmail.com",
                    "attachment_file": "./gift_card.png"
                }
            }
            API Output: "Draft create successful, Draft id: 18748946\nDraft message: "
            Bot: Draft create successful, Draft id is '18748946'. You can login to email to check it.
    """
    return create_draft_in_gmails(subject, body, to_address, from_address, attachment_file)

@tool
def send_mail_in_emails(subject: str, body: str, to_address: str, from_address: str, attachment_file: str) ->bool:
    """send mail in email.
        Here is an example of calling the function 'send_mail_in_emails', Please use JSON blob:
            User: Please help me send an email to dave:
                Subject: The issue regarding property fees.
                Body: Hi Dave, Congratulations! I have helped you apply for a reduction in property fees, and the property management has approved it. Your property fees will be reduced by 10% each month!
                To: dave@gmail.com
                From: me
            Bot: Okay, I will call the function 'send_mail_in_emails' to help you send this email.
            {
                "name": "send_mail_in_emails",
                "arguments": {
                    "subject": "The issue regarding property fees.",
                    "body": "Hi Dave, Congratulations! I have helped you apply for a reduction in property fees, and the property management has approved it. Your property fees will be reduced by 10% each month!",
                    "to_address": "dave@gmail.com",
                    "from_address": "me",
                    "attachment_file": ""
                }
            }
            API Output: "Email send successful, id: 47587987\message: [message]"
            Bot: Email send successful, Email id is '47587987'. You can login to email to check it.
    """
    return gmail_send_mail(subject, body, to_address, from_address, attachment_file)

email_toolboxes = [search_in_emails, create_draft_in_emails, send_mail_in_emails]
email_tool_names = {
    "search_in_emails": search_in_emails,
    "create_draft_in_emails": create_draft_in_emails,
    "send_mail_in_emails": send_mail_in_emails
}

# for google gemini
search_in_emails_gemini = genai.protos.Tool(
    function_declarations=[
      genai.protos.FunctionDeclaration(
        name='search_in_emails',
        description="Search in email.",
        parameters=genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                'search_criteria':genai.protos.Schema(type=genai.protos.Type.STRING, description="conforms to email's advanced search syntax format."),
            },
            required=['search_criteria']
        )
      )
    ])

create_draft_in_emails_gemini = genai.protos.Tool(
    function_declarations=[
      genai.protos.FunctionDeclaration(
        name='create_draft_in_emails',
        description="Create draft in email.",
        parameters=genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                'subject':genai.protos.Schema(type=genai.protos.Type.STRING, description="Title of the email."),
                'body':genai.protos.Schema(type=genai.protos.Type.STRING, description="Body of the email."),
                'to_address':genai.protos.Schema(type=genai.protos.Type.STRING, description="The address to receive the email, if not provided, needs to be confirmed by the user."),
                'from_address':genai.protos.Schema(type=genai.protos.Type.STRING, description='The sending address of the email, if using the default address, can be replaced with "me".'),
                'attachment_file':genai.protos.Schema(type=genai.protos.Type.STRING, description='The attachment parameter can be a file path. If there are no attachments, please pass "".'),
            },
            required=['subject', 'body', 'to_address', 'from_address', 'attachment_file']
        )
      )
    ])

send_mail_in_emails_gemini = genai.protos.Tool(
    function_declarations=[
      genai.protos.FunctionDeclaration(
        name='send_mail_in_emails',
        description="send mail in email.",
        parameters=genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                'subject':genai.protos.Schema(type=genai.protos.Type.STRING, description="Title of the email."),
                'body':genai.protos.Schema(type=genai.protos.Type.STRING, description="Body of the email."),
                'to_address':genai.protos.Schema(type=genai.protos.Type.STRING, description="The address to receive the email, if not provided, needs to be confirmed by the user."),
                'from_address':genai.protos.Schema(type=genai.protos.Type.STRING, description='The sending address of the email, if using the default address, can be replaced with "me".'),
                'attachment_file':genai.protos.Schema(type=genai.protos.Type.STRING, description='The attachment parameter can be a file path. If there are no attachments, please pass "".'),
            },
            required=['subject', 'body', 'to_address', 'from_address', 'attachment_file']
        )
      )
    ])

google_email_tools = [
    search_in_emails_gemini,
    create_draft_in_emails_gemini,
    send_mail_in_emails_gemini,
]

# for openai
search_in_emails_openai = {
    "type": "function",
    "function": {
        "name": "search_in_emails",
        "description": "Search in email.",
        "parameters": {
            "type": "object",
            "properties": {
                "search_criteria": {"type": "string", "description": "conforms to email's advanced search syntax format."},
            },
            "required": ["search_criteria"],
        },
    }
}

create_draft_in_emails_openai = {
    "type": "function",
    "function": {
        "name": "create_draft_in_emails",
        "description": "Create draft in email.",
        "parameters": {
            "type": "object",
            "properties": {
                "subject": {"type": "string", "description": "Title of the email."},
                "body": {"type": "string", "description": "Body of the email."},
                "to_address": {"type": "string", "description": "The address to receive the email, if not provided, needs to be confirmed by the user."},
                "from_address": {"type": "string", "description": 'The sending address of the email, if using the default address, can be replaced with "me".'},
                "attachment_file": {"type": "string", "description": 'The attachment parameter can be a file path. If there are no attachments, please pass "".'},
            },
            "required": ["subject", "body", "to_address", "from_address", "attachment_file"],
        },
    }
}

send_mail_in_emails_openai = {
    "type": "function",
    "function": {
        "name": "send_mail_in_emails",
        "description": "send mail in email.",
        "parameters": {
            "type": "object",
            "properties": {
                "subject": {"type": "string", "description": "Title of the email."},
                "body": {"type": "string", "description": "Body of the email."},
                "to_address": {"type": "string", "description": "The address to receive the email, if not provided, needs to be confirmed by the user."},
                "from_address": {"type": "string", "description": 'The sending address of the email, if using the default address, can be replaced with "me".'},
                "attachment_file": {"type": "string", "description": 'The attachment parameter can be a file path. If there are no attachments, please pass "".'},
            },
            "required": ["subject", "body", "to_address", "from_address", "attachment_file"],
        },
    }
}

openai_email_tools = [
    search_in_emails_openai,
    create_draft_in_emails_openai,
    send_mail_in_emails_openai,
]

def GetMailFuncallList() ->list:
    funcall_list = []
    for call_tool in email_toolboxes:
        funcall_list.append(call_tool.name)
    return funcall_list

def GetMailFuncallDescription(func_name: str = "") ->str:
    description = ""
    for call_tool in email_toolboxes:
        if func_name == call_tool.name:
            description = call_tool.description
    return description

def is_email_enable() ->bool:
    from WebUI.configs.basicconfig import GetCurrentRunningCfg
    config = GetCurrentRunningCfg()
    if not config:
        return None
    tool_boxes = config.get("ToolBoxes")
    if not tool_boxes:
        return False
    google_toolboxes = tool_boxes.get("Google ToolBoxes")
    if not google_toolboxes:
        return False
    email = google_toolboxes.get("Tools").get("Google Mail")
    enable = email.get("enable", False)
    return enable