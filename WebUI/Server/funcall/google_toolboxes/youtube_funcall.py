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

def get_watch_url(video_id):
    return f'https://www.youtube.com/watch?v={video_id}'

def search_from_mine_channel(service, title, max_results=4)->list:
    video_results = []
    if not service or not title:
        return video_results

    channels_response = service.channels().list(
            part='contentDetails',
            mine=True
        ).execute()

    for channel in channels_response.get("items", []):
        playlist_id = channel.get("contentDetails", {}).get("relatedPlaylists", {}).get("uploads")
        playlistItems_list = service.playlistItems().list(
            part="snippet",
            playlistId=playlist_id,
            maxResults=max_results
        )
        while playlistItems_list:
            playlist_items = playlistItems_list.execute()

            for item in playlist_items.get("items", {}):
                if max_results <= 0:
                    break
                video_title = item.get("snippet", {}).get("title")
                if title.lower() in video_title.lower():
                    video_description = item.get("snippet", {}).get("description")
                    video_id = item.get("snippet", {}).get("resourceId", {}).get("videoId")
                    #video_url = get_video_url(service, video_id)
                    video_url = get_watch_url(video_id)
                    result = {
                        "video_id": video_id,
                        "video_title": video_title,
                        "video_description": video_description,
                        "video_url": video_url
                    }
                    video_results.append(result)
                    max_results -= 1
            if max_results <= 0:
                break
            playlistItems_list = service.playlistItems().list_next(
                playlistItems_list, playlist_items
            )
    return video_results

def search_from_all_channel(service, title, max_results=4)->list:
    video_results = []
    if not service or not title:
        return video_results

    request = service.search().list(
    part="id,snippet",
        type='video',
        q=title,
        videoDefinition='high',
        maxResults=max_results
    )
    search_response = request.execute()
    for search in search_response.get("items", []):
        if max_results <= 0:
            break
        video_id = search.get("id", {}).get("videoId", "")
        video_url = get_watch_url(video_id)
        video_title = search.get("snippet", {}).get("title", "")
        video_description = search.get("snippet", {}).get("description", "")
        result = {
            "video_id": video_id,
            "video_title": video_title,
            "video_description": video_description,
            "video_url": video_url
        }
        video_results.append(result)
        max_results -= 1
    return video_results

def get_youtube_video_url_i(title: str, mine: bool = False, max_results=4):
    """ Get URL about Youtube video from your own Youtube channel. """

    youtube_results = {}
    from WebUI.Server.funcall.google_toolboxes.credential import glob_credentials
    if not glob_credentials:
        from WebUI.Server.funcall.google_toolboxes.credential import init_credential
        init_credential()
        from WebUI.Server.funcall.google_toolboxes.credential import glob_credentials
        if not glob_credentials:
            return "Credentials not found. This error is unrecoverable.", youtube_results

    
    service = build('youtube', 'v3', credentials=glob_credentials)
    if not service:
        return "youtube object create failed. This error is unrecoverable.", youtube_results
    try:
        if mine:
            youtube_results = search_from_mine_channel(service, title, max_results)
        else:
            youtube_results = search_from_all_channel(service, title, max_results)

        if not youtube_results:
            return "The video not found.", youtube_results
        youtube_dict = {
            "name": "youtube_video_url",
        }
        count = 1
        video_list = []
        youtube_message = ""
        for result in youtube_results:
            video_id = result['video_id']
            video_title = result['video_title']
            video_description = result['video_description']
            video_url = result['video_url']
            video_list.append({"video_id": video_id, "video_title": video_title, "video_description": video_description, "video_url": video_url})
            youtube_message += f" The video #{count}:\n  video title: '{video_title}'\n video description: '{video_description}'\n  video url: '{video_url}'\n\n"
            count += 1
        youtube_dict["videos"] = video_list
    except Exception as e:
        youtube_message = f"get video url failed, error: {e}"
        print(f"get_youtube_video_url Error: {e}")
    if not youtube_message:
        youtube_message = "The video not found."
    return youtube_message, youtube_dict


@tool
def get_youtube_video_url(title: str, mine: bool = False):
    """ Get URL about Youtube video from Youtube. 
        Args:
            title (str): Search for video titles.
            mine (boolean): Search only within my channel.

        Here is an example of calling the function 'get_youtube_video_url':
        User: Please retrieve the shared URL of the video 'Language Translation' from my YouTube channel.
        Bot: Okay, I will call the function 'get_youtube_video_url' to get this video URL.
        {
            "name": "get_youtube_video_url",
            "arguments": {
                "title": "Language Translation",
                "mine": true
            }
        }
        API Output: "The video found:
            video title: 'Real-time Language Translation Agent'
            video url: 'https://youtu.be/H78ABFocRrQ'
        Bot: The video has been successfully found, and its shared URL is 'https://youtu.be/H78ABFocRrQ'
    """
    return get_youtube_video_url_i(title, mine)

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
                'mine':genai.protos.Schema(type=genai.protos.Type.BOOLEAN, description="If true, it will Search only within my channel. If false, it will search throughout YouTube."),
            },
            required=['title', 'mine']
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
                "mine": {"type": "boolean", "description": "If true, it will Search only within my channel. If false, it will search throughout YouTube."},
            },
            "required": ["title", "mine"],
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