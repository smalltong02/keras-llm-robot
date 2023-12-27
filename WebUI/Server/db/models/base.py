from datetime import datetime
from sqlalchemy import Column, DateTime, String, Integer


class BaseModel:
    """
    base model
    """
    id = Column(Integer, primary_key=True, index=True, comment="Primary ID")
    create_time = Column(DateTime, default=datetime.utcnow, comment="Create Time")
    update_time = Column(DateTime, default=None, onupdate=datetime.utcnow, comment="Update Time")
    create_by = Column(String, default=None, comment="Creator")
    update_by = Column(String, default=None, comment="Updater")
