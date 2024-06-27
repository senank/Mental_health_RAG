from sentence_transformers import SentenceTransformer
from langchain_openai import OpenAIEmbeddings
from transformers import pipeline

from langchain_openai import ChatOpenAI
from langchain_community.llms.ollama import Ollama
from transformers import AutoTokenizer, AutoModelForCausalLM

from pdb import set_trace as pdb


import os

# Constants
OLLAMA_MODEL = 'mistral'

# Embeddings #
# OA
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

# Responses #
# OA
def get_response_OA_key(prompt):
    model = ChatOpenAI()
    response = model.predict(prompt)
    return response

def get_response_OA_HF(prompt):
    generator = pipeline("text-generation", model="openai-community/gpt2")
    generator(prompt, max_new_tokens=250)

# Ollama
def get_response_Ollama(prompt):
    # Load model and tokenizer
    model_name = "meta-llama/Llama-2-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Encode the prompt
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate a response
    output = model.generate(inputs.input_ids, max_length=150)

    # Decode the generated tokens to a string
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return response

def get_response_Ollama_pipe(prompt):
    generator = pipeline("text-generation", model="meta-llama/Llama-2-7b-chat-hf")
    response = generator({
        "role":"You are a mental health expert with a background in CBT, cognitive science and psychology. \
You have helped many people through mental health episodes and improviing their \
cognitive states.",
        "content":prompt})
    return response
    
def get_response_Ollama_local(prompt):
    # Running Ollama mistral locally
    model = Ollama(model=OLLAMA_MODEL)
    return model.invoke(prompt)