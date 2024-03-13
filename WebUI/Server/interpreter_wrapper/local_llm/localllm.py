import re
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from WebUI.Server.chat.utils import History
from WebUI.Server.utils import get_prompt_template
from langchain.prompts.chat import ChatPromptTemplate
from typing import List

class BaseLLM:
    def __init__(self):
        pass

    def chat(self, query):
        pass

class LocalLLM(BaseLLM):
    def __init__(self, 
            model_name : str = "",
            api_base : str = "",
            streaming : bool = True,
            verbose : bool = False,
            temperature : float = 0.7,
            max_tokens : int = 1000,
                 ):
        self.api_base = api_base
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stream = streaming

        self.model = ChatOpenAI(
            streaming=streaming,
            verbose=verbose,
            openai_api_key="EMPTY",
            openai_api_base=self.api_base,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def run(self,
            history : List[dict] = [],
            query : List[dict] = [],
            ):
        
        if "content" in query:
            query = query["content"]
        for his in history:
            if "type" in his:
                del his["type"]

        new_history = [History.from_data(h) for h in history]
        prompt_template = get_prompt_template("llm_chat", "default")
        input_msg = History(role="user", content=prompt_template).to_msg_template(False)
        chat_prompt = ChatPromptTemplate.from_messages(
                [i.to_msg_template() for i in new_history] + [input_msg])
        chain = LLMChain(prompt=chat_prompt, llm=self.model)

        answer = chain.run({"input": query})
        substrings = re.findall(r".*?\n+", answer)
        for line in substrings:
            yield line