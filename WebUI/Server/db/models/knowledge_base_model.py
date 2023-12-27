from sqlalchemy import Column, Integer, String, DateTime, func

from WebUI.Server.db.base import Base


class KnowledgeBaseModel(Base):
    """
    KnowledgeBase Model
    """
    __tablename__ = 'knowledge_base'
    id = Column(Integer, primary_key=True, autoincrement=True, comment='KnowledgeBase ID')
    kb_name = Column(String(50), comment='KnowledgeBase Name')
    kb_info = Column(String(200), comment='KnowledgeBase Information')
    vs_type = Column(String(50), comment='Vector Type')
    embed_model = Column(String(50), comment='Embedding Model Name')
    file_count = Column(Integer, default=0, comment='File Count')
    create_time = Column(DateTime, default=func.now(), comment='Create Time')

    def __repr__(self):
        return f"<KnowledgeBase(id='{self.id}', kb_name='{self.kb_name}',kb_intro='{self.kb_info} vs_type='{self.vs_type}', embed_model='{self.embed_model}', file_count='{self.file_count}', create_time='{self.create_time}')>"
