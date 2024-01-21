
from langchain.docstore.document import Document


class DocumentWithVSId(Document):
    """
    Vectorized document
    """
    id: str = None
    score: float = 3.0
