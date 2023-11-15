from pydantic import BaseModel
from typing import Dict, List, Optional

class ApiConfigParams(BaseModel):
    pass

class ApiEmbeddingsParams(ApiConfigParams):
    texts: List[str]
    embed_model: Optional[str] = None
    to_query: bool = False # for minimax