from fastapi import Body
from WebUI.Server.utils import BaseResponse
from WebUI.Server.db.repository.chat_history_repository import feedback_chat_history_to_db


def chat_feedback(chat_history_id: str = Body("", max_length=32, description="chat id"),
            score: int = Body(0, max=100, description="user score"),
            reason: str = Body("", description="the reason")
            ):
    try:
        feedback_chat_history_to_db(chat_history_id, score, reason)
    except Exception as e:
        msg = f"error: {e}"
        print(msg)
        return BaseResponse(code=500, msg=msg)

    return BaseResponse(code=200, msg=f"Chat records have been provided: {chat_history_id}")