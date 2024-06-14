import json
import datetime
import geocoder
from langchain_core.tools import tool
import google.generativeai as genai

@tool
def get_current_location():
    """Get current location information.
        Here is an example of calling the function 'get_current_location':
            User: Where am I?
            Bot: Okay, I will call the function 'get_current_location' to help you get current location.
            {
                "name": get_current_location,
                "arguments": {}
            }
            API Output: "123 Main St, City, State, ZIP"
            Bot: You are currently located at 123 Main St, City, State, ZIP.
    """
    location = geocoder.ip('me')

    if location:
        return location.address
    else:
        return "Location information not available."

@tool
def get_current_time():
    """Get the current local time.
    Here is an example of calling the function 'get_current_time':
        User: Do you know the current time?
        Bot: Okay, I will call the function 'get_current_time' to help you get current time.
        {
            "name": get_current_time,
            "arguments": {}
        }
        API Output: 2024-01-01 12:30:05
        Bot: The current time is 2024-01-01 12:30:05
    """
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")   
    return formatted_time

@tool
def submit_warranty_claim(caption: str, description: str):
    """Submit a repair order for a customer.

    Parameters:
        caption (string): Title of repair order.
        description (string): A detailed description of the damaged goods, including customer information.

    Here is an example of calling the function 'submit_warranty_claim':
        User: My network adapter is broken. Please help me submit a repair request.
        Bot: Sure, please provide the following information:
            1. Your full name.
            2. Email address.
            3. The product serial number.
            4. Date of purchase.
            5. A brief description of the issue.
        User: My name is John Doe. My email address is johndoe@example.com. The product serial number is XYZ123456789. The date of purchase was on January 1, 2022. The issue is that the network adapter is not working properly.
        Bot: Thank you for providing the necessary information. Please wait while we submit your repair request.
            {
                "name": submit_warranty_claim,
                "arguments": {
                    "caption": "My network adapter is broken.",
                    "description": "Customer: John Doe\nEmail: johndoe@example.com\nSerial number: XYZ123456789\nPurchase date: January 1, 2022\nIssue: The network adapter is not working properly.\nAdditional information: None\nWarranty expiration date: None\n"
                }
            }
        API Output: Submitted warranty success. The repair number is: 84905783
        Bot: Hi John Doe, your repair request has been submitted successfully. Your repair number is 84905783. We will get back to you soon.
    """
    warranty_context = f"""
     Title: {caption}

     Description: {description}

     Sent email to support@sony.com
"""
    import random
    repair_number = str(random.randint(100000, 999999))
    return f"""Submitted warranty success. The repair number is: {repair_number}\n
    The content of the email -
    {warranty_context}
    """

@tool
async def search_from_search_engine(se_name: str, query: str):
    """Search for information using a search engine.
    Here is an example of calling the function 'search_from_search_engine':
        User: Can you search for me on Google?
        Bot: Sure, please provide the search query.
        User: Sony TV
        Bot: Here are the top search results for 'Sony TV':
            {
                "name": search_from_search_engine,
                "arguments": {
                    "query": "Sony TV"
                }
            }
    """
    from WebUI.Server.chat.search_engine_chat import lookup_search_engine
    docs = await lookup_search_engine(query, se_name)
    source_documents = [
        f"""from [{inum + 1}] [{doc.metadata["source"]}]({doc.metadata["source"]}) \n\n{doc.page_content}\n\n"""
        for inum, doc in enumerate(docs)
    ]
    context = "\n".join([doc.page_content for doc in docs])
    if not context:
        return None, "", []
    new_query = f"""The user's question has been searched on the internet. Here is all the content retrieved from the search engine:
    {context}\n
"""
    return new_query, se_name, source_documents

@tool
async def search_from_knowledge_base(kb_name: str, query: str):
    """Search for information using a knowledge base.
    Here is an example of calling the function 'search_from_knowledge_base':
        User: Can you search for me on Knowledge Base?
        Bot: Sure, please provide the search query.
        User: Sony TV
        Bot: Here are the top search results for 'Sony TV':
            {
                "name": search_from_knowledge_base,
                "arguments": {
                    "query": "Sony TV"
                }
            }
    """
    from urllib.parse import urlencode
    from WebUI.Server.utils import detect_device
    from fastapi.concurrency import run_in_threadpool
    from WebUI.configs import USE_RERANKER, GetRerankerModelPath
    from WebUI.Server.reranker.reranker import LangchainReranker
    from WebUI.Server.knowledge_base.kb_doc_api import search_docs
    from WebUI.Server.knowledge_base.kb_service.base import KBServiceFactory
    from WebUI.Server.knowledge_base.utils import VECTOR_SEARCH_TOP_K, SCORE_THRESHOLD
    kb = KBServiceFactory.get_service_by_name(kb_name)
    if kb is None:
        return None, "", []
    docs = await run_in_threadpool(search_docs,
            query=query,
            knowledge_base_name=kb_name,
            top_k=VECTOR_SEARCH_TOP_K,
            score_threshold=SCORE_THRESHOLD)
    if USE_RERANKER:
            reranker_model_path = GetRerankerModelPath()
            print("-----------------model path------------------")
            print(reranker_model_path)
            reranker_model = LangchainReranker(top_n=VECTOR_SEARCH_TOP_K,
                                            device=detect_device(),
                                            max_length=1024,
                                            model_name_or_path=reranker_model_path
                                            )
            print(docs)
            docs = reranker_model.compress_documents(documents=docs,
                                                     query=query)
            print("---------after rerank------------------")
            print(docs)
    source_documents = []
    for inum, doc in enumerate(docs):
        filename = doc.metadata.get("source")
        parameters = urlencode({"knowledge_base_name": kb_name, "file_name": filename})
        base_url = "/"
        url = f"{base_url}knowledge_base/download_doc?" + parameters
        text = f"""from [{inum + 1}] [{filename}]({url}) \n\n{doc.page_content}\n\n"""
        source_documents.append(text)
    context = "\n".join([doc.page_content for doc in docs])
    if not context:
        return None, "", []
    new_query = f"""The user's issue has been searched through the knowledge base. Here is all the content retrieved:
    {context}\n
"""
    return new_query, kb_name, source_documents

funcall_tools = [get_current_location, get_current_time, submit_warranty_claim]
tool_names = {
    "get_current_location": get_current_location,
    "get_current_time": get_current_time,
    "submit_warranty_claim": submit_warranty_claim,
}
search_tools = [search_from_search_engine]
search_tool_names = {
    "search_from_search_engine": search_from_search_engine,
}
knowledge_base_tools = [search_from_knowledge_base]
kb_tool_names = {
    "search_from_knowledge_base": search_from_knowledge_base,
}

# for google gemini
get_current_location_func = genai.protos.Tool(
    function_declarations=[
      genai.protos.FunctionDeclaration(
        name='get_current_location',
        description="Get current location information.",
        parameters=None
      )
    ])

get_current_time_func = genai.protos.Tool(
    function_declarations=[
      genai.protos.FunctionDeclaration(
        name='get_current_time',
        description="Get the current local time.",
        parameters=None
      )
    ])

submit_warranty_claim_func = genai.protos.Tool(
    function_declarations=[
      genai.protos.FunctionDeclaration(
        name='submit_warranty_claim',
        description="Submit a repair order for a customer.",
        parameters=genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                'caption':genai.protos.Schema(type=genai.protos.Type.STRING, description="Title of repair order"),
                'description':genai.protos.Schema(type=genai.protos.Type.STRING, description="A detailed description of the damaged goods, including customer information")
            },
            required=['caption','description']
        )
      )
    ])

search_from_search_engine_func = genai.protos.Tool(
    function_declarations=[
      genai.protos.FunctionDeclaration(
        name='search_from_search_engine',
        description="search any information from network when a question exceeds your knowledge scope or when it's beyond the timeframe of your training data.",
        parameters=genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                'query':genai.protos.Schema(type=genai.protos.Type.STRING, description="The questions to look up on the internet."),
            },
            required=['query']
        )
      )
    ])

search_from_knowledge_base_func = genai.protos.Tool(
    function_declarations=[
      genai.protos.FunctionDeclaration(
        name='search_from_knowledge_base',
        description="search any information from knowledge base.",
        parameters=genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                'query':genai.protos.Schema(type=genai.protos.Type.STRING, description="The questions to look up on the knowledge base."),
            },
            required=['query']
        )
      )
    ])

google_funcall_tools = [
        get_current_location_func,
        get_current_time_func,
        submit_warranty_claim_func,
    ]

google_search_tools = [
    search_from_search_engine_func,
]

google_knowledge_base_tools = [
    search_from_knowledge_base_func,
]

# for openai
get_current_location_openai = {
    "type": "function",
    "function": {
        "name": "get_current_location",
        "description": "Get current location information.",
        "parameters": {
            "type": "object",
            "properties": {
            },
        },
    }
}

get_current_time_openai = {
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "Get the current local time.",
        "parameters": {
            "type": "object",
            "properties": {
            },
        },
    }
}

submit_warranty_claim_openai = {
    "type": "function",
    "function": {
        "name": "submit_warranty_claim",
        "description": "Submit a repair order for a customer.",
        "parameters": {
            "type": "object",
            "properties": {
                "caption": {"type": "string", "description": "Title of repair order"},
                "description": {"type": "string", "description": "A detailed description of the damaged goods, including customer information"},
            },
            "required": ["caption", "description"],
        },
    }
}

openai_tools = [
    #get_current_location_openai,
    #get_current_time_openai,
    submit_warranty_claim_openai,

]

def GetFuncallList() ->list:
    funcall_list = []
    for call_tool in funcall_tools:
        funcall_list.append(call_tool.name)
    return funcall_list

def GetFuncallDescription(func_name: str = "") ->str:
    description = ""
    for call_tool in funcall_tools:
        if func_name == call_tool.name:
            description = call_tool.description
    return description

def GetFuncallName(json_data: str) ->str:
    try:
        func = json.loads(json_data)
        func_name = func.get("name", "")
        for call_tool in funcall_tools:
            if func_name == call_tool.name:
                return func_name
    except json.JSONDecodeError:
        return ""
    
def RunNormalFunctionCalling(json_data: str) ->str:
    try:
        func = json.loads(json_data)
        print("func: ", func)
        func_name = func.get("name", "")
        func_arg = func.get("arguments", {})
        if func_name in tool_names:
            result = tool_names[func_name].run(func_arg)
            return func_name, result
        else:
            return "", ""
    except json.JSONDecodeError:
        return "", ""

async def RunFunctionCallingForKnowledgeBase(json_data: str) -> str:
    try:
        func = json.loads(json_data)
        print("func: ", func)
        func_name = func.get("name", "")
        func_arg = func.get("arguments", {})
        if func_name == "search_from_knowledge_base":
            #from WebUI.configs.basicconfig import GetCurrentRunningCfg
            #running_cfg = GetCurrentRunningCfg()
            result = tool_names[func_name].run(func_arg)
            return func_name, result
        else:
            return "", ""
    except json.JSONDecodeError:
        return "", ""

async def RunFunctionCallingForSearchEngine(json_data: str) -> str:
    try:
        func = json.loads(json_data)
        print("func: ", func)
        func_name = func.get("name", "")
        func_arg = func.get("arguments", {})
        if func_name == "search_from_search_engine":
            #from WebUI.configs.basicconfig import GetCurrentRunningCfg
            #running_cfg = GetCurrentRunningCfg()
            result = tool_names[func_name].run(func_arg)
            return func_name, result
        else:
            return "", ""
    except json.JSONDecodeError:
        return "", ""
