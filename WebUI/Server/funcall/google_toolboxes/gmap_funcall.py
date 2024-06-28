import os
from io import BytesIO
from PIL import Image
import googlemaps
from langchain_core.tools import tool
import google.generativeai as genai
from WebUI.configs.basicconfig import GetSearchKeyInGToolBox

def gmap_addressvalidation(address: str)->bool:
    if not address:
        return False
    api_key = GetSearchKeyInGToolBox()
    if not api_key:
        from WebUI.configs.basicconfig import load_env
        load_env()
        api_key = os.environ.get("GOOGLE_SEARCH_KEY", "")
    gmaps = googlemaps.Client(key=api_key)
    result = gmaps.addressvalidation(address)
    return result

def get_gmap_geolocate()->dict:
    api_key = GetSearchKeyInGToolBox()
    if not api_key:
        from WebUI.configs.basicconfig import load_env
        load_env()
        api_key = os.environ.get("GOOGLE_SEARCH_KEY", "")
    gmaps = googlemaps.Client(key=api_key)
    location = {}
    location = gmaps.geolocate()
    return location

def get_gmap_geocode(address: str)->list:
    if not address:
        return []
    api_key = GetSearchKeyInGToolBox()
    if not api_key:
        from WebUI.configs.basicconfig import load_env
        load_env()
        api_key = os.environ.get("GOOGLE_SEARCH_KEY", "")
    gmaps = googlemaps.Client(key=api_key)
    geocode_result = gmaps.geocode(address)
    if not geocode_result:
        return []
    return geocode_result

def get_gmap_reverse_geocode(latitude: float, longitude: float)->list:
    if not latitude or not longitude:
        return []
    api_key = GetSearchKeyInGToolBox()
    if not api_key:
        from WebUI.configs.basicconfig import load_env
        load_env()
        api_key = os.environ.get("GOOGLE_SEARCH_KEY", "")
    gmaps = googlemaps.Client(key=api_key)
    geocode_result = gmaps.reverse_geocode((latitude, longitude))
    if not geocode_result:
        return []
    return geocode_result

def get_gmap_directions(origin: str, destination: str, mode: str="", departure_time: str="", arrival_time: str="", optimize_waypoints=True)->list:
    if not origin and not destination:
        return []
    if not origin:
        location = get_gmap_geolocate()
        if not location:
            return []
        origin = location["location"]
    if not destination:
        location = get_gmap_geolocate()
        if not location:
            return []
        destination = location["location"]
    if not mode:
        mode = "driving"
    if departure_time:
        arrival_time = ""
    api_key = GetSearchKeyInGToolBox()
    if not api_key:
        from WebUI.configs.basicconfig import load_env
        load_env()
        api_key = os.environ.get("GOOGLE_SEARCH_KEY", "")
    gmaps = googlemaps.Client(key=api_key)
    routes = []
    if not departure_time and not arrival_time:
        routes = gmaps.directions(origin, destination, mode=mode)
    else:
        if departure_time:
            routes = gmaps.directions(origin, destination, mode=mode, departure_time=departure_time, optimize_waypoints=optimize_waypoints)
        elif arrival_time:
            routes = gmaps.directions(origin, destination, mode=mode, arrival_time=arrival_time, optimize_waypoints=optimize_waypoints)
    if not routes:
        return []
    return routes

def get_gmap_distance_matrix(origins: list, destinations: list, mode: str="", departure_time: str="", arrival_time: str="")->dict:
    if not origins or not destinations:
        return {}
    if not mode:
        mode = "driving"
    if departure_time:
        arrival_time = ""
    api_key = GetSearchKeyInGToolBox()
    if not api_key:
        from WebUI.configs.basicconfig import load_env
        load_env()
        api_key = os.environ.get("GOOGLE_SEARCH_KEY", "")
    gmaps = googlemaps.Client(key=api_key)
    routes = {}
    if not departure_time and not arrival_time:
        routes = gmaps.distance_matrix(origins, destinations, mode=mode)
    else:
        if departure_time:
            routes = gmaps.distance_matrix(origins, destinations, mode=mode, departure_time=departure_time)
        elif arrival_time:
            routes = gmaps.distance_matrix(origins, destinations, mode=mode, arrival_time=arrival_time)
    if not routes:
        return {}
    return routes

def random_color_generator():
    import random
    while True:
        yield "#{:06x}".format(random.randint(0, 0xFFFFFF))
random_color_gen = random_color_generator()

def get_gmap_static_map(origin: str, destinations: list[str], size: tuple=(600, 400), maptype: str="roadmap", format: str="png")->Image:
    if not origin or not destinations:
        return None
    from googlemaps.maps import StaticMapMarker
    from googlemaps.maps import StaticMapPath
    api_key = GetSearchKeyInGToolBox()
    if not api_key:
        from WebUI.configs.basicconfig import load_env
        load_env()
        api_key = os.environ.get("GOOGLE_SEARCH_KEY", "")
    gmaps = googlemaps.Client(key=api_key)
    path_list =[]
    marker_list =[]
    marker_list.append(StaticMapMarker(location=origin, color=next(random_color_gen), label="S",))
    for destination in destinations:
        path_list.append(StaticMapPath(
            points=[origin, destination],
            weight=5,
            color=next(random_color_gen),
        ))
        marker_list.append(StaticMapMarker(
            location=destination, color=next(random_color_gen), label="D",
        ))

    response = gmaps.static_map(
        size=size,
        zoom=14,
        center=origin,
        maptype=maptype,
        format=format,
        scale=2,
        path=path_list,
        markers=marker_list,
    )
    if response:
        image_data = b"".join(response)
        image = Image.open(BytesIO(image_data))
        return image
    return None

def get_gmap_places(query: str, location: dict={}, radius: int=1000, min_price: int=0, max_price: int=4, open_now: bool=True)->dict:
    if not query:
        return {}
    if not location:
        location = get_gmap_geolocate()
        if not location:
            return {}
        location = location["location"]
    api_key = GetSearchKeyInGToolBox()
    if not api_key:
        from WebUI.configs.basicconfig import load_env
        load_env()
        api_key = os.environ.get("GOOGLE_SEARCH_KEY", "")
    gmaps = googlemaps.Client(key=api_key)
    places = gmaps.places(query=query, location=location, radius=radius, min_price=min_price, max_price=max_price, open_now=open_now)
    if not places:
        return {}
    return places

def get_gmap_nearest_roads():
    pass

def get_gmap_timezone(location: str)->dict:
    if not location:
        return {}
    api_key = GetSearchKeyInGToolBox()
    if not api_key:
        from WebUI.configs.basicconfig import load_env
        load_env()
        api_key = os.environ.get("GOOGLE_SEARCH_KEY", "")
    gmaps = googlemaps.Client(key=api_key)
    geocode_result = gmaps.geocode(location)
    if not geocode_result:
        return {}
    location = geocode_result[0]['geometry']['location']
    results = gmaps.timezone(location=location)
    if not results:
        return {}
    return results

def get_gmap_url(address: str)->str:
    geocode_result = get_gmap_geocode(address)

    if not geocode_result or not isinstance(geocode_result, list):
        return "Address not found. This error is unrecoverable."
    
    latitude = geocode_result[0]["geometry"]["location"]["lat"]
    longitude = geocode_result[0]["geometry"]["location"]["lng"]

    base_url = "https://www.google.com/maps/search/?api=1&query="
    lat_lon = f"{latitude},{longitude}"

    map_url = f"Get Address URL: {base_url}{lat_lon}"
    print(map_url)
    return map_url

@tool
def get_map_url(address: str):
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
    return get_gmap_url(address), {}

@tool
def get_map_directions(origin: str, destination: str, mode: str=""):
    """Get the directions from origin to destination.
        Here is an example of calling the function 'get_gmap_directions', When the origin or destination is "", it means that they are your current position:
        User: On 12 o'clock, I need to go to Vancouver Zoo to attend Jane's birthday party. Please help me plan the route.
        Bot: Okay, I will call the function 'get_gmap_directions' to get the direction from your location to Vancouver Zoo.
        {
            "name": "get_map_directions",
            "arguments": {
                "origin": "",
                "destination": "Vancouver Zoo",
                "mode": "driving"
            }
        }
        API Output: Starting from '20171 92A Ave, Langley Twp, BC V1M 3A5, Canada', ending at 'Greater Vancouver Zoo, 5048 264 St, Aldergrove, BC V4W 1N7, Canada'. the total distance is 17.2 kilometers, and the total time is 17 minutes. The optimal route is to take the 'Trans-Canada Hwy/BC-1 E'.
        Bot: The optimal route from your current location to Vancouver Zoo is to take the "Trans-Canada Hwy/BC-1 E". The total distance is 17.2 kilometers, and the total time is 17 minutes. Given that you need to arrive at your destination by 12:00:00, it is best to depart by 11:40:00.
    """
    directions = get_gmap_directions(origin, destination, mode)
    direction = {}
    if not directions:
        return "No directions found.", {}
    direction = directions[0]
    if not direction:
        return "No directions found.", {}
    summary = direction["summary"]
    start_address = direction["legs"][0]["start_address"]
    start_location = direction["legs"][0]["start_location"]
    end_address = direction["legs"][0]["end_address"]
    end_location = direction["legs"][0]["end_location"]
    distance = direction["legs"][0]["distance"]["text"]
    duration = direction["legs"][0]["duration"]["text"]
    overview_polyline = direction["overview_polyline"]["points"]
    prompt = f"""Starting from '{start_address}', ending at '{end_address}'.
        the total distance is {distance}, and the total time is {duration}.
        The optimal route is to take the '{summary}'."""
    direction_dict = {
        "name": "map_directions",
        "start_address": start_address,
        "start_location": start_location,
        "end_address": end_address,
        "end_location": end_location,
        "distance": distance,
        "duration": duration,
        "overview_polyline": overview_polyline,
    }
    return prompt, direction_dict

@tool
def get_map_places(query: str, location: str="", radius: int=1000, min_price: int=0, max_price: int=4, open_now: bool=True):
    """Get the places that match the query, such as restaurants, libraries, parks, airports, stores, etc.
        Here is an example of calling the function 'get_map_places', When the location is "", it means that you are in the current location:
        User: Are there any good Italian restaurants nearby? Please recommend a few for me.
        Bot: Okay, I will call the function 'get_gmap_places' to get the Italian restaurants near you.
        {
            "name": "get_map_places",
            "arguments": {
                "query": "Italian restaurant",
                "location": ""
            }
        }
        API Output: There are 3 restaurants near you that match the query 'Italian restaurants'.
        Bot: Here are the 3 restaurants that match the query 'Italian restaurants'.
    """
    def GetPrice(price_level: int = 2):
        if price_level == 0:
            return "Free"
        elif price_level == 1:
            return "Cheap"
        elif price_level == 2:
            return "Medium price"
        elif price_level == 3:
            return "Expensive"
        elif price_level == 4:
            return "Very Expensive"
        else:
            return "Unknown"
    location_dict = {}
    geocode_result = get_gmap_geocode(location)
    if geocode_result:
        location_dict = geocode_result[0]["geometry"]["location"]
    places = get_gmap_places(query, location_dict, radius, min_price, max_price, open_now)
    if not geocode_result or not places:
        return "No places found.", []
    place_list = []
    latitude = geocode_result[0]["geometry"]["location"]["lat"]
    longitude = geocode_result[0]["geometry"]["location"]["lng"]
    map_dict = {
            "name": "map_places",
            "start": location,
            "current_location": {
                "lat": latitude,
                "lng": longitude
            }
        }
    prompt = "I found a set of matching places and here are their details:\n\n"
    for place in places["results"]:
        name = place["name"]
        status = place["business_status"]
        address = place["formatted_address"]
        open_now = place["opening_hours"]["open_now"]
        rating = place["rating"]
        price_level = place["price_level"]
        user_ratings_total = place["user_ratings_total"]
        location = place["geometry"]["location"]
        icon = place["icon"]
        price = GetPrice(price_level)
        place_list.append({
            "name": name,
            "status": status,
            "address": address,
            "open_now": open_now,
            "rating": rating,
            "price": price,
            "user_ratings_total": user_ratings_total,
            "icon": icon,
            "location": location,
        })
        prompt += f"""name: {name}
        address: {address}
        business_status: {status}
        open_now: {open_now}
        rating: {rating}
        price: {price}
        user ratings total: {user_ratings_total}
        """ + "\n\n"

    map_dict["locations"] = place_list
    return prompt, map_dict


map_toolboxes = [get_map_url, get_map_directions, get_map_places]
map_tool_names = {
    "get_map_url": get_map_url,
    "get_map_directions": get_map_directions,
    "get_map_places": get_map_places,
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

get_map_directions_gemini = genai.protos.Tool(
    function_declarations=[
      genai.protos.FunctionDeclaration(
        name='get_map_directions',
        description="Get the directions from origin to destination.",
        parameters=genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                'origin':genai.protos.Schema(type=genai.protos.Type.STRING, description="""Starting address, When the location is "", it means that you are in the current location."""),
                'destination':genai.protos.Schema(type=genai.protos.Type.STRING, description="""Destination address, When the location is "", it means that you are in the current location."""),
                'mode':genai.protos.Schema(type=genai.protos.Type.STRING, enum=["driving", "walking", "bicycling", "transit"], description="""Specifies the mode of transport to use when calculating directions. One of "driving", "walking", "bicycling" or "transit"."""),
            },
            required=['origin', 'destination', 'mode']
        )
      )
    ])

get_map_places_gemini = genai.protos.Tool(
    function_declarations=[
      genai.protos.FunctionDeclaration(
        name='get_map_places',
        description="Get the places that match the query, such as restaurants, libraries, parks, airports, stores, etc.",
        parameters=genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                'query':genai.protos.Schema(type=genai.protos.Type.STRING, description="""The text string on which to search, for example: "restaurant"."""),
                'location':genai.protos.Schema(type=genai.protos.Type.STRING, description="""location, When the location is "", it means that you are in the current location."""),
                'radius':genai.protos.Schema(type=genai.protos.Type.INTEGER, description="Distance in meters within which to bias results."),
            },
            required=['query', 'location']
        )
      )
    ])

google_maps_tools = [
    get_map_url_gemini,
    get_map_directions_gemini,
    get_map_places_gemini,
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

get_map_directions_openai = {
    "type": "function",
    "function": {
        "name": "get_map_directions",
        "description": "Get the directions from origin to destination.",
        "parameters": {
            "type": "object",
            "properties": {
                "origin": {"type": "string", "description": "Starting address, if it is a current local address, you can pass an empty string."},
                "destination": {"type": "string", "description": "Destination address, if it is a local address, you can pass an empty string."},
                "mode": {"type": "string", "enum": ["driving", "walking", "bicycling", "transit"], "description": """Specifies the mode of transport to use when calculating directions. One of "driving", "walking", "bicycling" or "transit"."""},
            },
            "required": ["origin", "destination", "mode"],
        },
    }
}

get_map_places_openai = {
    "type": "function",
    "function": {
        "name": "get_map_places",
        "description": "Get the places that match the query, such as restaurants, libraries, parks, airports, stores, etc.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": """The text string on which to search, for example: "restaurant"."""},
                "location": {"type": "string", "description": """location, When the location is "", it means that you are in the current location."""},
                "radius": {"type": "string", "description": "Distance in meters within which to bias results."},
            },
            "required": ["query", "location"],
        },
    }
}

openai_maps_tools = [
    get_map_url_openai,
    get_map_directions_openai,
    get_map_places_openai,
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