import redis
from redis.commands.search.field import VectorField, TextField, NumericField
from redis.commands.search.query import Query
from redis.commands.search.indexDefinition import IndexDefinition, IndexType

import openai
import os
import requests
import csv
import numpy as np
from utils import vstack_if_exists
from keys import OPENAI_KEY
from config import TEXT_EMBED_DIM

# OpenAI API key
openai.api_key = OPENAI_KEY

def get_embedding(text):
	# Embed a line of text
	response = openai.Embedding.create(
    	model= "text-embedding-ada-002",
    	input=text
	)
	# Extract the AI output embedding as a list of floats
	embedding = response["data"][0]["embedding"]
    
	return np.array(embedding).astype(np.float32)

def get_embeddings_from_csv(file):
    embeddings = None
    labels = []
    all_data = []

    # Open the CSV file
    with open(file, 'r') as csvfile:
        # Create a CSV reader
        csvreader = csv.reader(csvfile)
        
        # Skip the header row if it exists
        #next(csvreader, None)
        
        # Iterate through the rows and extract the labels (first column)
        for row in csvreader:
            label = row[0]  # Assuming label is in the first column
            data = row[1]
            labels.append(label)
            all_data.append(data)
            embedding = get_embedding(label)
            embeddings = vstack_if_exists(embeddings, embedding)
    print(embeddings.shape)
    np.save('co2_embeddings', embeddings)
    return labels, embeddings, all_data

def upload_embeddings(file):


    
    # Connect to the Redis server
    conn = redis.Redis(host='localhost', port=6379, encoding='utf8', decode_responses=True)

    labels, embeddings, co2 = get_embeddings_from_csv(file)
    p = conn.pipeline(transaction=False)
    

    for i, (label, vector, carbon) in enumerate(zip(labels, embeddings, co2)):
        byte_vector = vector.tobytes()
        # Create a new hash with url and embedding
        post_hash = {
            "label": label,
            "embedding": byte_vector,
            "co2": carbon
        }

    
        # create hash
        conn.hset(name=f"co2_embedding:{i}", mapping=post_hash)
    
    p.execute()


def index_db():

    openai.api_key = OPENAI_KEY

    # Connect to the Redis server
    conn = redis.Redis(host='localhost', port=6379, encoding='utf8', decode_responses=True)
    co2_schema = [
        TextField("label"),
        VectorField("embedding", "HNSW", {"TYPE": "FLOAT32", "DIM": TEXT_EMBED_DIM, "DISTANCE_METRIC": "COSINE"}),
        NumericField("co2"),
    ]

    try:
        conn.ft("ingredients_co2").create_index(fields=co2_schema, definition=IndexDefinition(prefix=["ingredient:"], index_type=IndexType.HASH))
    except Exception as e:
        print("Index already exists")
        print(e)

if __name__ == "__main__":
    get_embeddings_from_csv('co2.csv')
    #index_db()