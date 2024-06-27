<h1>Mental Health Companion: A RAG-Based Application</h1>

## Overview
This project demonstrates the implementation of a Retrieval-Augmented Generation (RAG) architecture to generate responses based on relevant data retrieved from a MongoDB Atlas database. The data is indexed using vector embeddings generated by the Hugging Face Sentence Transformer model "all-MiniLM-L6-v2". When provided with a user prompt, the application indexes the database for relevant documents, refines the prompt based on this data, and passes it to the open-source LLM (LLaMA) to generate contextually rich responses.

## Features:
- **Data Indexing:** Utilizes MongoDB Atlas for storing and indexing vector embeddings of text data.
- **Vector Embeddings:** Uses all-MiniLM-L6-v2 model from  Hugging Face's SentenceTransformer to generate vector embeddings
- **Dynamic Prompt Crafting:** Generates contextually relevant prompts based on user input and retrieved data.
- **Response Generation:** Uses LLAMA for generating accurate and context-aware responses.

## System Requirements
- Python 3.12+
- MongoDB Atlas account
- Hugging Face transformers
- PyTorch
- Ollama running locally 

## Installation and Setup
1. Clone the repository:
   ```
   git clone https://github.com/senank/Mental_health_RAG.git
   ```
2. Install required packages:
   ```
   pip install -r requirements.txt
   ```
3. Add HuggingFace and MongoDB Atlas tokens as environment variables (```$ HF_TOKEN``` and ```$ MONGO_TOKEN``` respectively), or change them in ```main.py``` under ```API Constants```.

## Usage
1. Start the MongoDB Atlas cluster; ensure to create a database and collection that have the same name as the constants defined in ```main.py``` i.e. ```DB_NAME, COLLECTION_NAME```.
2. Start Ollama and run:
   ```
   ollama run mistral
   ```
4. Run main.py to populate the database with data and embeddings, and run the application:
   ```
   python main.py
   ```
