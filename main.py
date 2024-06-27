import pandas as pd

from embeddings import generate_embedding_HF
from db_helpers import load_csv_to_mongodb
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
SEARCH_I_NAME = "comb_emb_vsearch"

# API CONSTANTS
MONGO_KEY = os.getenv('MONGO_KEY')
OA_KEY = os.getenv('OPENAI_API_KEY')
HF_KEY = os.getenv('HF_KEY')

openai.api_key = os.getenv(OA_KEY)


#Connect to DB
client = MongoClient(MONGO_KEY, server_api=ServerApi('1'))


def get_similarity(query, db, collection):
    db = client[db]
    collection = db[collection]
    
    results = collection.aggregate([
        {"$vectorSearch": {
            "queryVector": generate_embedding_HF(query),
            "path": "combined_embedding",
            "numCandidates": 98,
            "limit": 5,
            "index" : SEARCH_I_NAME,
            }
        }
    ])
    return results

def compare_embeddings_to_input(text, db, collection):
    pass


if __name__ == '__main__':
    # Load the CSV file
    df = pd.read_csv('data/Mental_Health_FAQ.csv')
    questions = df['Questions'].tolist()
    answers = df['Answers'].tolist()

    data_for_db = [{'question': q, 'answer': a} for q, a in zip(questions, answers)]
    load_csv_to_mongodb(data_for_db, client, DB_NAME, COLLECTION_NAME)
    x = get_similarity("depression", DB_NAME, COLLECTION_NAME)