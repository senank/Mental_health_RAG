from load_data import generate_response, DB_NAME, COLLECTION_NAME  
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi 
import os 

MONGO_KEY = os.getenv('MONGO_TOKEN')

if __name__ == '__main__':
    client = MongoClient(MONGO_KEY, server_api=ServerApi('1'))
    question_mode = True
    prompt = input("What can I help you with today?\n")
    print(generate_response(prompt, client, DB_NAME, COLLECTION_NAME))
    while question_mode:
        if input("Is there anything else I can help you with? (Y/N)\n").lower() in ['no', 'n']:
            question_mode = False
            break
        prompt = input("What can I help you with today?\n")
        print(generate_response(prompt, DB_NAME, COLLECTION_NAME))
    
    print("Thank you for using your mental health companion, if there is anything else I can help in the future please don't be shy! :)")