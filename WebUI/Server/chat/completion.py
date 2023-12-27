from fastapi import Body
from fastapi.responses import StreamingResponse
from configs import LLM_MODELS, TEMPERATURE
from WebUI.Server.utils import wrap_done, get_OpenAI
from langchain.chains import LLMChain
from langchain.callbacks import AsyncIteratorCallbackHandler
from typing import AsyncIterable, Optional
import asyncio
from langchain.prompts.chat import PromptTemplate
from WebUI.Server.utils import get_prompt_template


async def completion(query: str = Body(..., description="User input: ", examples=["chat"]),
                     stream: bool = Body(False, description="stream output"),
                     echo: bool = Body(False, description="echo"),
                     model_name: str = Body(LLM_MODELS[0], description="LLM model name"),
                     temperature: float = Body(TEMPERATURE, description="LLM Temperature", ge=0.0, le=1.0),
                     max_tokens: Optional[int] = Body(1024, description="max tokens."),
                     prompt_name: str = Body("default",
                                             description=""),
                     ):

    async def completion_iterator(query: str,
                                  model_name: str = LLM_MODELS[0],
                                  prompt_name: str = prompt_name,
                                  echo: bool = echo,
                                  ) -> AsyncIterable[str]:
        callback = AsyncIteratorCallbackHandler()
        model = get_OpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=[callback],
            echo=echo
        )

        prompt_template = get_prompt_template("completion", prompt_name)
        prompt = PromptTemplate.from_template(prompt_template)
        chain = LLMChain(prompt=prompt, llm=model)

        # Begin a task that runs in the background.
        task = asyncio.create_task(wrap_done(
            chain.acall({"input": query}),
            callback.done),
        )

        if stream:
            async for token in callback.aiter():
                # Use server-sent-events to stream the response
                yield token
        else:
            answer = ""
            async for token in callback.aiter():
                answer += token
            yield answer

        await task

    return StreamingResponse(completion_iterator(query=query,
                                                 model_name=model_name,
                                                 prompt_name=prompt_name),
                             media_type="text/event-stream")