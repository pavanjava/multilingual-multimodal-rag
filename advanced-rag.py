from llama_index.core import VectorStoreIndex, StorageContext, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.qdrant import QdrantVectorStore
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


class MultilingualMultimodalRAG:
    def __init__(self):
        pass
