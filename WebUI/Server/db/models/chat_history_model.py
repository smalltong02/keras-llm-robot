from sqlalchemy import Column, Integer, String, DateTime, JSON, func

from WebUI.Server.db.base import Base, engine


class ChatHistoryModel(Base):
    """
    Chat History Model
    """
    __tablename__ = 'chat_history'
    id = Column(String(32), primary_key=True, comment='History ID')
    # chat/agent_chat
    chat_type = Column(String(50), comment='char type')
    query = Column(String(4096), comment='query by user')
    response = Column(String(4096), comment='response by model')
    # knowledge id, reserved.
    meta_data = Column(JSON, default={})
    feedback_score = Column(Integer, default=-1, comment='score')
    feedback_reason = Column(String(255), default="", comment='reason')
    create_time = Column(DateTime, default=func.now(), comment='create time')

    def __repr__(self):
        return f"<ChatHistory(id='{self.id}', chat_type='{self.chat_type}', query='{self.query}', response='{self.response}',meta_data='{self.meta_data}',feedback_score='{self.feedback_score}',feedback_reason='{self.feedback_reason}', create_time='{self.create_time}')>"

Base.metadata.create_all(bind=engine)