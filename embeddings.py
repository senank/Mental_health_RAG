from sentence_transformers import SentenceTransformer
from langchain_openai import OpenAIEmbeddings

from pdb import set_trace as pdb


# Open AI
def get_embedding_function_OA():
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    return embeddings

def generate_embedding_OA(text: str) -> list[float]:
    emb = get_embedding_function_OA()
    v = emb.embed_query(text)
    return v

# all-miniLM-L6-v2
def get_embedding_function_HF():
    embeddings = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

def generate_embedding_HF(text: str) -> list[float]:
    emb = get_embedding_function_HF()
    v = emb.encode(text).tolist()
    return v
