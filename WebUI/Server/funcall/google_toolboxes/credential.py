import os.path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from WebUI.configs import (GOOGLE_OAUTH_PORT)

GOOGLE_TOKEN_FILE = "google_token.json"

READONLY_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
MODIFY_SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]
FULL_SCOPES = ["https://mail.google.com/"]

glob_credentials = None

def init_credential(credentials_path: str, scopes: list[str]) -> bool:
    creds = None
    global glob_credentials
    if glob_credentials:
        return True
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
            # Save the credentials for the next run
        with open(GOOGLE_TOKEN_FILE, "w") as token:
            token.write(creds.to_json())
            glob_credentials = creds
            return True
    else:
        glob_credentials = creds
        return True
    return False