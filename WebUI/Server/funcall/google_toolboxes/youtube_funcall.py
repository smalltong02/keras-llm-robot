from googleapiclient.discovery import build
from langchain_core.tools import tool
from bs4 import BeautifulSoup
import google.generativeai as genai

YOUTUBE_READONLY_SCOPES = ["https://www.googleapis.com/auth/youtube.readonly"]
YOUTUBE_FULL_SCOPES = ["https://www.googleapis.com/auth/youtube"]

def get_video_url(service, video_id):
    request = service.videos().list(
        part='player',
        id=video_id
    ).execute()
    embed_html = request['items'][0]['player']['embedHtml']
    soup = BeautifulSoup(embed_html, 'html.parser')
    video_url = soup.iframe['src']
    if video_url:
        video_url = "https:" + video_url
    return video_url

def get_youtube_video_url_i(title: str) ->str:
    """ Get URL about Youtube video from your own Youtube channel. """

    from WebUI.Server.funcall.google_toolboxes.credential import glob_credentials
    if not glob_credentials:
        from WebUI.Server.funcall.google_toolboxes.credential import init_credential
        init_credential()
        from WebUI.Server.funcall.google_toolboxes.credential import glob_credentials
        if not glob_credentials:
            return "Credentials not found. This error is unrecoverable."

    youtube_message = ""
    service = build('youtube', 'v3', credentials=glob_credentials)
    if not service:
        return "youtube object create failed. This error is unrecoverable."
    try:
        channels_response = service.channels().list(
            part='contentDetails',
            mine=True
        ).execute()

        for channel in channels_response.get("items", {}):
            playlist_id = channel.get("contentDetails", {}).get("relatedPlaylists", {}).get("uploads")
            playlistItems_list = service.playlistItems().list(
                part="snippet",
                playlistId=playlist_id,
                maxResults=50
            )
            while playlistItems_list:
                playlist_items = playlistItems_list.execute()

                for item in playlist_items.get("items", {}):
                    video_title = item.get("snippet", {}).get("title")
                    if title.lower() in video_title.lower():
                        video_description = item.get("snippet", {}).get("description")
                        video_id = item.get("snippet", {}).get("resourceId", {}).get("videoId")
                        video_url = get_video_url(service, video_id)
                        youtube_message = f" The video found:\n  video title: '{video_title}'\n video description: '{video_description}'\n  video url: '{video_url}'"
                        break
                if youtube_message:
                    break    
                playlistItems_list = service.playlistItems().list_next(
                        playlistItems_list, playlist_items
                    )
    except Exception as e:
        youtube_message = f"get video url failed, error: {e}"
        print(f"get_youtube_video_url Error: {e}")
    if not youtube_message:
        youtube_message = "The video not found."
    return youtube_message


@tool
def get_youtube_video_url(title: str) ->str:
    """ Get URL about Youtube video from your own Youtube channel. 
        Here is an example of calling the function 'get_youtube_video_url':
        User: Please retrieve the shared URL of the video 'Language Translation' from my YouTube channel.
        Bot: Okay, I will call the function 'get_youtube_video_url' to get this video URL.
        {
            "name": "get_youtube_video_url",
            "arguments": {
                "title": "Language Translation"
            }
        }
        API Output: "The video found:
            video title: 'Real-time Language Translation Agent'
            video url: 'https://youtu.be/H78ABFocRrQ'
        Bot: The video has been successfully found, and its shared URL is 'https://youtu.be/H78ABFocRrQ'
    """
    return get_youtube_video_url_i(title)

youtube_toolboxes = [get_youtube_video_url]
youtube_tool_names = {
    "get_youtube_video_url": get_youtube_video_url,
}

# for google gemini
get_youtube_video_url_func = genai.protos.Tool(
    function_declarations=[
      genai.protos.FunctionDeclaration(
        name='get_youtube_video_url',
        description="Get URL about Youtube video from your own Youtube channel.",
        parameters=genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                'title':genai.protos.Schema(type=genai.protos.Type.STRING, description="Use the advanced search syntax like the Youtube API, Here's an example: 'Language Translation'"),
            },
            required=['title']
        )
      )
    ])

google_youtube_tools = [
    get_youtube_video_url_func,
]

get_youtube_video_url_openai = {
    "type": "function",
    "function": {
        "name": "get_youtube_video_url",
        "description": "Get URL about Youtube video from your own Youtube channel.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Use the advanced search syntax like the Youtube API, Here's an example: 'Language Translation'"},
            },
            "required": ["title"],
        },
    }
}

openai_youtube_tools = [
    get_youtube_video_url_openai,
]

def GetYoutubeFuncallList() ->list:
    funcall_list = []
    for call_tool in youtube_toolboxes:
        funcall_list.append(call_tool.name)
    return funcall_list

def GetYoutubeFuncallDescription(func_name: str = "") ->str:
    description = ""
    for call_tool in youtube_toolboxes:
        if func_name == call_tool.name:
            description = call_tool.description
    return description

def is_youtube_enable() ->bool:
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
    youtube = google_toolboxes.get("Tools").get("Google Youtube")
    enable = youtube.get("enable", False)
    return enable