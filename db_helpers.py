from embeddings import generate_embedding_HF

# Data loading
def load_csv_to_mongodb(data, client, db, collection):    
    db = client[db]
    collection = db[collection]
    
    # Check if the data is already loaded
    if collection.estimated_document_count() > 0:
        print("Data already loaded into the database.")
        
    else:
        # Insert data into the collection
        collection.insert_many(data)
        generate_embeddings_for_db(client, db, collection)
        print(f"Data loaded successfully into {db}.{collection}")
    
    #Create Search index
    # if SEARCH_I_NAME in list(collection.list_search_indexes()):
        # create_search_index(collection)
    
    return


# Search index
def get_search_index():
    search_index_model = {
        "mappings": {
            "dynamic": True,
            "fields": {
                "combined_embedding": {
                    "dimensions": 384,
                    "similarity": "dotProduct",
                    "type": "knnVector"
                }
            }
        }
    }

    return search_index_model

def create_search_index(collection, name):
    try:
        definition = get_search_index()
        index = {"definition": definition, "name": name}
        collection.createSearchIndex(index) # Needs a higher version of atlas, done manually through app
    except Exception as e:
        print(e)
        exit()


# Embeddings
def generate_embeddings_for_db(client, db, collection):
    #comment when using as a helper function
    db = client[db]
    collection = db[collection]

    query = {"question": {"$exists": True, "$ne": ""}, "answer": {"$exists": True, "$ne": ""}}
    x = collection.find(query)
    
    for data_chunk in collection.find(query):
        comb_emb = generate_embedding_HF(data_chunk['question'] + " [SEP] " + data_chunk['answer']) # Using HF ATM
        update = {
            '$set': {
                'combined_embedding': comb_emb
            }
        }
        collection.update_one({'_id': data_chunk['_id']}, update)
