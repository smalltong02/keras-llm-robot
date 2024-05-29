import os
import googlemaps
from langchain_core.tools import tool


def get_gmap_url(address) ->str:

    api_key = os.environ.get("GOOGLE_SEARCH_KEY", "")
    gmaps = googlemaps.Client(key=api_key)

    geocode_result = gmaps.geocode(address)

    if not geocode_result:
        return "Address not found. This error is unrecoverable."
    
    latitude = geocode_result[0]["geometry"]["location"]["lat"]
    longitude = geocode_result[0]["geometry"]["location"]["lng"]

    base_url = "https://www.google.com/maps/search/?api=1&query="
    lat_lon = f"{latitude},{longitude}"

    map_url = f"Get Address URL: {base_url}{lat_lon}"
    print(map_url)
    return map_url

@tool
def get_map_url(address) ->str:
    """Get the URL of a Google Map for the given address.
        Here is an example of calling the function 'get_map_url':
        User: Please help me mark the McDonald's address '20394 88 Ave, Langley Twp, BC V1M 2Y4' on the map and generate its map URL.
        Bot: Okay, I will call the function 'get_map_url' to mark this address on the map.
        {
            "name": "get_map_url",
            "arguments": {
                "address": "20394 88 Ave, Langley Twp, BC V1M 2Y4"
            }
        }
        API Output: "Get Address URL: https://www.google.com/maps/search/?api=1&query=49.1624051,-122.6570233"
        Bot: The URL of the map is https://www.google.com/maps/search/?api=1&query=49.1624051,-122.6570233
    """
    return get_gmap_url(address)

map_toolboxes = [get_map_url]
map_tool_names = {
    "get_map_url": get_map_url,
}

def GetMapFuncallList() ->list:
    funcall_list = []
    for call_tool in map_toolboxes:
        funcall_list.append(call_tool.name)
    return funcall_list

def GetMapFuncallDescription(func_name: str = "") ->str:
    description = ""
    for call_tool in map_toolboxes:
        if func_name == call_tool.name:
            description = call_tool.description
    return description

def is_map_enable() ->bool:
    from WebUI.configs.webuiconfig import InnerJsonConfigWebUIParse
    configinst = InnerJsonConfigWebUIParse()
    tool_boxes = configinst.get("ToolBoxes")
    if not tool_boxes:
        return False
    google_toolboxes = tool_boxes.get("Google ToolBoxes")
    if not google_toolboxes:
        return False
    maps = google_toolboxes.get("Tools").get("Google Maps")
    enable = maps.get("enable", False)
    return enable