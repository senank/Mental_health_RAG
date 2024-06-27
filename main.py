import pandas as pd

from embeddings import generate_embedding_HF, get_response_Ollama_local
from db_helpers import load_csv_to_mongodb
from prompt import context_from_data, PROMPT_TEMPLATE
import openai

import transformers


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
HF_KEY = os.getenv('HF_TOKEN')

# File constants
FILENAME = 'data/Mental_Health_FAQ.csv'

openai.api_key = os.getenv(OA_KEY)

#Data parsing
def parse_data(filename):
    df = pd.read_csv(filename)
    questions = df['Questions'].tolist()
    answers = df['Answers'].tolist()
    data_for_db = [{'question': q, 'answer': a} for q, a in zip(questions, answers)]
    return data_for_db

def get_similarity(query, collection):
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


def get_response(prompt):
    return get_response_Ollama_local(prompt)

def generate_response(query, db, collection):
    db = client[db]
    collection = db[collection]
    data = list(get_similarity(query, collection))
    prompt = generate_prompt(query, data)
    response = get_response(prompt)
    formatted_response = format_response(response, data)
    return formatted_response
    
    
# Prompt/Response formating
def generate_prompt(query, data):
    context = context_from_data(data)
    prompt = PROMPT_TEMPLATE.format(query=query, context=context)
    return prompt

def format_response(response, data):
    sources = []
    for i in range(len(data)):
        sources.append("ID_{}".format(data[i]['_id']))
    return "{}. \n (Sources: {})".format(response, ", ".join(sources))
    

if __name__ == '__main__':
    #Connect to DB
    client = MongoClient(MONGO_KEY, server_api=ServerApi('1'))
    # transformers.logging.set_verbosity_info()

    # Load the CSV file
    data = parse_data(FILENAME)

    # Load CSV data into DB and add vector embeddings
    load_csv_to_mongodb(data, client, DB_NAME, COLLECTION_NAME)
    
    question_mode = True
    prompt = input("What can I help you with today?\n")
    print(generate_response(prompt, DB_NAME, COLLECTION_NAME))
    while question_mode:
        if input("Is there anything else I can help you with? (Y/N)\n").lower() in ['no', 'n']:
            question_mode = False
            break
        prompt = input("What can I help you with today?\n")
        print(generate_response(prompt, DB_NAME, COLLECTION_NAME))
    
    print("Thank you for using your mental health companion, if there is anything else I can help in the future please don't be shy! :)")
    
    
    