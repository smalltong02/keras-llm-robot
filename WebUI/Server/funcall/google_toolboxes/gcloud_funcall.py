import os
from langchain_core.tools import tool
from googleapiclient.discovery import build
from typing import Any

DRIVE_READONLY_SCOPES = ["https://www.googleapis.com/auth/drive.metadata.readonly"]
DRIVE_FULL_SCOPES = ["https://www.googleapis.com/auth/drive"]

GOOGLE_DOCS_TYPES = [
    {"application/vnd.google-apps.audio": "audio/wav"},
    {"application/vnd.google-apps.document": "application/pdf"},
    {"application/vnd.google-apps.drive-sdk": ""},
    {"application/vnd.google-apps.drawing": "image/bmp"},
    {"application/vnd.google-apps.file": "application/octet-stream"},
    {"application/vnd.google-apps.folder": ""},
    {"application/vnd.google-apps.form": "application/vnd.ms-excel"},
    {"application/vnd.google-apps.fusiontable": ""},
    {"application/vnd.google-apps.jam": ""},
    {"application/vnd.google-apps.mail-layout": ""},
    {"application/vnd.google-apps.map": ""},
    {"application/vnd.google-apps.photo": "image/jpeg"},
    {"application/vnd.google-apps.presentation": "application/pdf"},
    {"application/vnd.google-apps.script": ""},
    {"application/vnd.google-apps.shortcut": ""},
    {"application/vnd.google-apps.site": ""},
    {"application/vnd.google-apps.spreadsheet": ""},
    {"application/vnd.google-apps.unknown": ""},
    {"application/vnd.google-apps.video": "video/mp4"},
]

def get_gcloud_path(service, parent_ids) ->str:
    parent_names = []
    for parent_id in parent_ids:
        parent = service.files().get(fileId=parent_id).execute()
        parent_name = parent.get("name")
        parent_names.append(parent_name)
    if not parent_names:
        return ""
    full_path = "/".join(parent_names)
    return full_path

def search_cloud(service, search_criteria: str) ->Any:
    if not service:
        return None
    
    results = (
        service.files()
        .list(
            q=search_criteria,
            pageSize=10, 
            fields="nextPageToken, files(id, name, size, createdTime, modifiedTime, mimeType, parents)")
        .execute()
    )
    items = results.get("files", [])

    if not items:
        return None
    return items

def export_media(service, file_id, mime_type, filename) ->bool:
    if not service or not mime_type or not filename:
        return False
    filename = filename + '.' + mime_type.split("/")[-1]
    request = service.files().export_media(fileId=file_id, mimeType=mime_type)
    response = request.execute()
    with open(filename, 'wb') as fp:
        fp.write(response)
    return True

def get_media(service, file_id, filename) ->bool:
    if not service or not filename:
        return False
    request = service.files().get_media(fileId=file_id)
    response = request.execute()
    with open(filename, 'wb') as fp:
        fp.write(response)
    return True

def upload_to_cloud(service, file_path):
    from googleapiclient.http import MediaFileUpload
    file_name = os.path.basename(file_path)
    media = MediaFileUpload(file_path, resumable=True)
    body = {
        "name": file_name,
    }
    file = service.files().create(body=body, media_body=media, fields="id").execute()
    return file

def search_in_gdrive(search_criteria: str) ->str:
    """search file in google drive. The parameter 'search_criteria' conforms to google drive's advanced search syntax format."""
    from WebUI.Server.funcall.google_toolboxes.credential import glob_credentials
    if not glob_credentials:
        return "Credentials not found. This error is unrecoverable."
    
    gdrive_messages = ""
    service = build("drive", "v3", credentials=glob_credentials)
    if not service:
        return "google drive object create failed. This error is unrecoverable."
    try:
        file_items = search_cloud(service=service, search_criteria=search_criteria)
        if not file_items:
            return "Search files successful, but no files matching the criteria were found."
        
        file_counts = 1
        for item in file_items:
            #full_path = get_gcloud_path(service, item.get("parents", ""))
            prefix = f"This is file #{file_counts}: \n"
            file_name = f"File Name: {item.get('name', "")} \n"
            file_id = f"File Id: {item.get('id', "")} \n"
            file_size = f"File Size: {item.get('size', "")} \n"
            file_created_time = f"Created Time: {item.get('createdTime', "")} \n"
            file_modified_time = f"Modified Time: {item.get('modifiedTime', "")} \n"
            file_mime_type = f"Mime Type: {item.get('mimeType', "")} \n"
            if gdrive_messages:
                gdrive_messages += "\n\n"
            gdrive_messages += f"{prefix}{file_name}{file_id}{file_size}{file_created_time}{file_modified_time}{file_mime_type}"
            file_counts += 1
    except Exception as e:
        gdrive_messages = f"Search file failed, error: {e}"
        print(f"search_in_gdrive Error: {e}")
    return gdrive_messages

def download_from_gdrive(search_criteria: str, download_path: str) ->str:
    """download file from google drive."""
    from WebUI.Server.funcall.google_toolboxes.credential import glob_credentials
    if not glob_credentials:
        return "Credentials not found. This error is unrecoverable."
    
    gdrive_messages = ""
    service = build("drive", "v3", credentials=glob_credentials)
    if not service:
        return "google drive object create failed. This error is unrecoverable."
    items = search_cloud(service=service, search_criteria=search_criteria)
    if not items:
        return "Search files successful, but no files matching the criteria were found."
    try:
        for item in items:
            file_path = os.path.join(download_path, item['name'])
            matched_items = [types for types in GOOGLE_DOCS_TYPES if item["mimeType"] in types]
            if matched_items:
                matched_item = matched_items[0]
                export_media(service=service, file_id=item['id'], mime_type=matched_item[item["mimeType"]], filename=file_path)
            else:
                get_media(service=service, file_id=item['id'], filename=file_path)
            if gdrive_messages:
                gdrive_messages += "\n\n"
            gdrive_messages += f"file '{file_path}' downloaded successfully.\n"
    except Exception as e:
        gdrive_messages = f"Download failed, error: {e}"
        print(f"download_from_gdrive Error: {e}")
    return gdrive_messages

def upload_to_gdrive(upload_file: str) ->str:
    """upload file to google drive."""
    from WebUI.Server.funcall.google_toolboxes.credential import glob_credentials
    if not glob_credentials:
        return "Credentials not found. This error is unrecoverable."
    
    gdrive_messages = ""
    service = build("drive", "v3", credentials=glob_credentials)
    if service:
        return "google drive object create failed. This error is unrecoverable."
    try:
        if os.path.isfile(upload_file):
            file = upload_to_cloud(service, upload_file)
            gdrive_messages = f"file '{upload_file}' upload successfully.\n"
        elif os.path.isdir(upload_file):
            for root, _, files in os.walk(upload_file):
                for file in files:
                    file_path = os.path.join(root, file)
                    file = upload_to_cloud(service, file_path)
                    if gdrive_messages:
                        gdrive_messages += "\n\n"
                    gdrive_messages += f"file '{file_path}' upload successfully.\n"
    except Exception as e:
        gdrive_messages = f"Upload failed, error: {e}"
        print(f"upload_to_gdrive Error: {e}")
    return gdrive_messages

@tool
def search_in_cloud_storage(search_criteria: str) ->str:
    """search in cloud storage."""
    return search_in_gdrive(search_criteria)

@tool
def download_from_cloud_storage(search_criteria: str, download_path: str) ->str:
    """download from cloud storage."""
    return download_from_gdrive(search_criteria, download_path)

@tool
def upload_to_cloud_storage(upload_file: str) ->str:
    """upload file to cloud storage."""
    return upload_to_gdrive(upload_file)

# drive_toolboxes = [search_in_cloud_storage, download_from_cloud_storage, upload_to_cloud_storage]
# drive_tool_names = {
#     "search_in_cloud_storage": search_in_cloud_storage,
#     "download_from_cloud_storage": download_from_cloud_storage,
#     "upload_to_cloud_storage": upload_to_cloud_storage,
# }