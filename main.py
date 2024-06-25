import pandas as pd

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from langchain_community.embeddings.ollama import OllamaEmbeddings


from langchain_community.vectorstores import Chroma

from embeddings import generate_embedding_HF
import openai

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

import os
from pdb import set_trace as pdb

#DB CONSTANTS
#Chorma DB
CHROMA_PATH = "chroma"
#MongoDB
DB_NAME = 'MentalhealthQA'
COLLECTION_NAME = 'QAs'

# API CONSTANTS
URI = os.getenv('URI')
OA_KEY = os.getenv('OPENAI_API_KEY')
HF_KEY = os.getenv('HF_KEY')

openai.api_key = os.getenv(OA_KEY)



#Connect to DB
client = MongoClient(URI, server_api=ServerApi('1'))
client.admin.command('ping')

# DB Methods
def load_csv_to_mongodb(data, db, collection):    
    db = client[db]
    collection = db[collection]
    
    # Check if the data is already loaded
    if collection.estimated_document_count() > 0:
        print("Data already loaded into the database.")
        return
    else:
        # Insert data into the collection
        collection.insert_many(data)
        generate_embeddings_for_db_OA(db, collection)
        print(f"Data loaded successfully into {db}.{collection}")

def generate_embeddings_for_db(db, collection):
    db = client[db]
    collection = db[collection]
    query = {"question": {"$exists": True, "$ne": ""}, "answer": {"$exists": True, "$ne": ""}}
    x = collection.find(query)
    for data_chunk in collection.find(query):
        comb_emb = generate_embedding_HF(data_chunk['question'] + " [SEP] " + data_chunk['answer'])
        update = {
            '$set': {
                'combined_embedding': comb_emb
            }
        }
        collection.update_one({'_id': data_chunk['_id']}, update)



# Prepare data for Elasticsearch
def generate_data(df):
    for index, row in df.iterrows():
        yield {
            "_index": "mental_health_qa",
            "_type": "record",
            "_id": index,
            "_source": {
                "question": row['Question'],
                "answer": row['Answer']
            }
        }



def STACK_DEFINED_RAG():
    # questions = df['Questions'].tolist()
    # answers = df['Answers'].tolist()

    # data_for_rag = [{'question': q, 'answer': a} for q, a in zip(questions, answers)]
    pass

def save_to_db(data):
    # db = Chroma.from_documents(data_for_rag, OpenAIEmbeddings(), persist_directory = CHROMA_PATH)
    
    # Create a new client and connect to the server
    
    pass

if __name__ == '__main__':
    # Load the CSV file
    df = pd.read_csv('data/Mental_Health_FAQ.csv')
    questions = df['Questions'].tolist()
    answers = df['Answers'].tolist()

    data_for_db = [{'question': q, 'answer': a} for q, a in zip(questions, answers)]
    # load_csv_to_mongodb(data_for_db, DB_NAME, COLLECTION_NAME)

    
    # # Connect to local Elasticsearch instance
    # es = Elasticsearch()

    # # Bulk index the data
    # bulk(es, generate_data(df))
