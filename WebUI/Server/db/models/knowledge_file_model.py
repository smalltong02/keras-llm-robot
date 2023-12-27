from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, JSON, func

from WebUI.Server.db.base import Base


class KnowledgeFileModel(Base):
    """
    KnowledgeFile Model
    """
    __tablename__ = 'knowledge_file'
    id = Column(Integer, primary_key=True, autoincrement=True, comment='KnowledgeFile ID')
    file_name = Column(String(255), comment='file name')
    file_ext = Column(String(10), comment='file extension')
    kb_name = Column(String(50), comment='KnowledgeBase Name')
    document_loader_name = Column(String(50), comment='Loader Name')
    text_splitter_name = Column(String(50), comment='Splitter Name')
    file_version = Column(Integer, default=1, comment='File Version')
    file_mtime = Column(Float, default=0.0, comment="Modify Time")
    file_size = Column(Integer, default=0, comment="File Size")
    custom_docs = Column(Boolean, default=False, comment="custom docs")
    docs_count = Column(Integer, default=0, comment="Documents count")
    create_time = Column(DateTime, default=func.now(), comment='Create Time')

    def __repr__(self):
        return f"<KnowledgeFile(id='{self.id}', file_name='{self.file_name}', file_ext='{self.file_ext}', kb_name='{self.kb_name}', document_loader_name='{self.document_loader_name}', text_splitter_name='{self.text_splitter_name}', file_version='{self.file_version}', create_time='{self.create_time}')>"


class FileDocModel(Base):
    """
    File Document Model
    """
    __tablename__ = 'file_doc'
    id = Column(Integer, primary_key=True, autoincrement=True, comment='ID')
    kb_name = Column(String(50), comment='KnowledgeBase Name')
    file_name = Column(String(255), comment='File Name')
    doc_id = Column(String(50), comment="Document ID")
    meta_data = Column(JSON, default={})

    def __repr__(self):
        return f"<FileDoc(id='{self.id}', kb_name='{self.kb_name}', file_name='{self.file_name}', doc_id='{self.doc_id}', metadata='{self.meta_data}')>"
