import os
from io import BytesIO
from PIL import Image
from googleapiclient.discovery import build
from langchain_core.tools import tool
import google.generativeai as genai
from google.auth.transport.requests import AuthorizedSession

DEFAULT_MAX_ITEMS = 10

PHOTO_READONLY_SCOPES = ["https://www.googleapis.com/auth/photoslibrary.readonly"]
PHOTO_FULL_SCOPES = ["https://www.googleapis.com/auth/photoslibrary"]

def search_gphotos(albums_name: str, max_items: int = DEFAULT_MAX_ITEMS)->list:
    """ Search for photos on Google Photos Album. """
    from WebUI.Server.funcall.google_toolboxes.credential import glob_credentials
    if not glob_credentials:
        from WebUI.Server.funcall.google_toolboxes.credential import init_credential
        init_credential()
        from WebUI.Server.funcall.google_toolboxes.credential import glob_credentials
        if not glob_credentials:
            return []
    
    photos_list = []
    service = build('photoslibrary', 'v1', credentials=glob_credentials, static_discovery=False)
    if not service:
        return photos_list
    try:
        album_id = ""
        results = service.albums().list(
            pageSize=50, fields="nextPageToken,albums(id,title)").execute()
        items = results.get('albums', [])
        if not items:
            return photos_list
        else:
            for item in items:
                title = str(item.get('title', ""))
                if albums_name in title:
                    album_id = item['id']
                    break
        if album_id:
            results = service.mediaItems().search(
                body={
                    "albumId": album_id,
                    "pageSize": 50,
                }
            ).execute()
            items = results.get('mediaItems', [])
            if not items:
                return photos_list
            else:
                print('Media Items:')
                item_count = min(max_items, len(items))
                photos_list = items[:item_count]
    except Exception as e:
        print(e)
    return photos_list

@tool
def search_photos(albums_name: str, max_items: int = DEFAULT_MAX_ITEMS):
    """
    Search and download for photos on Photos Album.
    Args:
        albums_name (str): The name of the album will be used to search for photos within the album.
        max_items (int): The maximum number of results to return photos.

    Here is an example of calling the function 'search_photos':
        User: Please help me find the first photo in the album "funny".
        Bot: Okay, I will call the function 'search_photos' to find first photo in the album "funny".
        {
            "name": "search_photos",
            "arguments": {
                "albums_name": "funny",
                "max_items": 1
            }
        }
        API Output: got the photos in album 'funny':
                    photo #1:
                        filename: funny_photo_1.jpg
                        description: a funny photo
                        creationTime: 2020-09-05T21:39:04.235Z
    """
    from pathlib import Path
    from WebUI.configs.basicconfig import TMP_DIR
    photo_dict = {
        "name": "search_photos",
    }
    photo_list = []
    photo_prompt = ""

    photos = search_gphotos(albums_name, max_items)
    if not photos:
        return "No photos found in album '{}'".format(albums_name), {}
    else:
        photo_index = 1
        for photo in photos:
            filename = photo.get('filename', "")
            description = photo.get('description', "")
            creation_time = photo.get('mediaMetadata', {}).get('creationTime', "")
            baseUrl = photo.get('baseUrl', "")
            mimeType = photo.get('mimeType', "")

            imgpath = ""
            if mimeType.startswith('image/'):
                from WebUI.Server.funcall.google_toolboxes.credential import glob_credentials
                authed_session = AuthorizedSession(glob_credentials)
                image_data_response = authed_session.get(baseUrl)
                if image_data_response and image_data_response.reason == "OK":
                    imgpath = str(TMP_DIR / Path(filename))
                    image = Image.open(BytesIO(image_data_response.content))
                    image.save(imgpath)
                photo_list.append({
                    "filename": filename,
                    "description": description,
                    "creationTime": creation_time,
                    "imgpath": imgpath
                })
                photo_prompt += f"""photo #{photo_index}:
                                    filename: {filename}
                                    description: {description}
                                    creationTime: {creation_time}
                                    imgpath: {imgpath}
                """ + "\n\n"
                photo_index += 1
    photo_dict["photos"] = photo_list
    return photo_prompt, photo_dict

@tool
def upload_photos(albums_name: str, upload_path: str):
    """
    Upload photos to Photos Album.
    Args:
        albums_name (str): The name of the album will be used to search for photos within the album.
        upload_path (str): The path of the photo will be used to upload to the album.
        
    Here is an example of calling the function 'upload_photos':
        
    """
    pass

photo_toolboxes = [search_photos]
photo_tool_names = {
    "search_photos": search_photos,
}

# for gemini
search_photos_gemini = genai.protos.Tool(
    function_declarations=[
      genai.protos.FunctionDeclaration(
        name='search_photos',
        description="Search and download for photos on Photos Album.",
        parameters=genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                'albums_name':genai.protos.Schema(type=genai.protos.Type.STRING, description="The name of the album will be used to search for photos within the album."),
                'max_items':genai.protos.Schema(type=genai.protos.Type.INTEGER, description="The maximum number of results to return photos."),
            },
            required=['albums_name']
        )
      )
    ])

google_photo_tools = [
    search_photos_gemini,
]

# for openai
search_photos_openai = {
    "type": "function",
    "function": {
        "name": "search_photos",
        "description": "Search and download for photos on Photos Album.",
        "parameters": {
            "type": "object",
            "properties": {
                "albums_name": {"type": "string", "description": "The name of the album will be used to search for photos within the album."},
                "max_items": {"type": "integer", "description": "The maximum number of results to return photos."},
            },
            "required": ["albums_name"],
        },
    }
}

openai_photo_tools = [
    search_photos_openai,
]

def GetPhotoFuncallList() ->list:
    funcall_list = []
    for call_tool in photo_toolboxes:
        funcall_list.append(call_tool.name)
    return funcall_list

def GetPhotoFuncallDescription(func_name: str = "") ->str:
    description = ""
    for call_tool in photo_tool_names:
        if func_name == call_tool.name:
            description = call_tool.description
    return description

def is_photo_enable() ->bool:
    from WebUI.configs.basicconfig import GetCurrentRunningCfg
    config = GetCurrentRunningCfg(True)
    if not config:
        return None
    tool_boxes = config.get("ToolBoxes")
    if not tool_boxes:
        return False
    google_toolboxes = tool_boxes.get("Google ToolBoxes")
    if not google_toolboxes:
        return False
    maps = google_toolboxes.get("Tools").get("Google Photos")
    enable = maps.get("enable", False)
    return enable