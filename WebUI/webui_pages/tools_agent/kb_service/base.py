import operator
from abc import ABC, abstractmethod

import os
from pathlib import Path
from langchain.embeddings.base import Embeddings
from langchain.docstore.document import Document

