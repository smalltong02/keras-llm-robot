import os
import googlemaps
from langchain_core.tools import tool
import google.generativeai as genai

    # import googlemaps
    # import folium
    # from streamlit_folium import st_folium
    # gmaps = googlemaps.Client(key="***")
    # geocode_result = gmaps.geocode("***")
    # location = geocode_result[0]['geometry']['location']
    # print("location: ", location)
    # coordinates = [location['lat'], location['lng']]
    # m = folium.Map(location=coordinates, zoom_start=16)
    # folium.Marker(coordinates, popup="My Home").add_to(m)
    # st_data = st_folium(m, width=725)

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
def get_map_url(address: str) ->str:
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

# for google gemini
get_map_url_gemini = genai.protos.Tool(
    function_declarations=[
      genai.protos.FunctionDeclaration(
        name='get_map_url',
        description="Get the URL of a Google Map for the given address.",
        parameters=genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                'address':genai.protos.Schema(type=genai.protos.Type.STRING, description="This is an address that Maps can recognize."),
            },
            required=['address']
        )
      )
    ])

google_maps_tools = [
    get_map_url_gemini,
]

get_map_url_openai = {
    "type": "function",
    "function": {
        "name": "get_map_url",
        "description": "Get the URL of a Google Map for the given address.",
        "parameters": {
            "type": "object",
            "properties": {
                "address": {"type": "string", "description": "This is an address that Maps can recognize."},
            },
            "required": ["address"],
        },
    }
}

openai_maps_tools = [
    get_map_url_openai,
]

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
    maps = google_toolboxes.get("Tools").get("Google Maps")
    enable = maps.get("enable", False)
    return enable