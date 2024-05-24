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
    """Get the URL of a Google Map for the given address."""
    return get_gmap_url(address)

# map_toolboxes = [get_map_url]
# map_tool_names = {
#     "get_map_url": get_map_url,
# }