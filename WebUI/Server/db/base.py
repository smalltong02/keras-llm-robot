from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base, DeclarativeMeta
from sqlalchemy.orm import sessionmaker
import os
import json
from WebUI.configs.basicconfig import GetKbConfig, GetDbRootPath, GetDbUri

kb_config = GetKbConfig()
db_path = os.path.abspath(GetDbRootPath(kb_config))
db_uri = GetDbUri(kb_config) + db_path

engine = create_engine(
    db_uri,
    json_serializer=lambda obj: json.dumps(obj, ensure_ascii=False),
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base: DeclarativeMeta = declarative_base()
