from googleapiclient.discovery import build
from langchain_core.tools import tool
from bs4 import BeautifulSoup

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
                    video_title_low = [title.lower() for title in video_title]
                    if title.lower() in video_title_low:
                        video_id = item.get("snippet", {}).get("resourceId", {}).get("videoId")
                        video_url = get_video_url(service, video_id)
                        youtube_message = f" The video found:\n  video title: '{video_title}'\n  video url: '{video_url}'"
                        break
                if youtube_message:
                    break    
                playlistItems_list = service.playlistItems().list_next(
                        playlistItems_list, playlist_items
                    )
    except Exception as e:
        youtube_message = f"get video url failed, error: {e}"
        print(f"get_youtube_video_url Error: {e}")
    return youtube_message


@tool
def get_youtube_video_url(title: str) ->str:
    """ Get URL about Youtube video from your own Youtube channel. """
    return get_youtube_video_url_i(title)

# youtube_toolboxes = [get_youtube_video_url]
# youtube_tool_names = {
#     "get_youtube_video_url": get_youtube_video_url,
# }