import openai
import pandas as pd
import torch
import numpy as np
import os
import csv
from keys import OPENAI_KEY
from utils import vstack_if_exists

openai.api_key = OPENAI_KEY

def get_embedding(text):
	# Embed a line of text
	response = openai.Embedding.create(
    	model= "text-embedding-ada-002",
    	input=text
	)
	# Extract the AI output embedding as a list of floats
	embedding = response["data"][0]["embedding"]
    
	return np.array(embedding)


def get_embeddings(text, fname):

    # avoid unnecessary API calls
    if os.path.exists(f'{fname}_embeddings.csv'):
        embeddings = np.genfromtxt(f'{fname}_embeddings.csv', delimiter=',',dtype=float)

        text = []
        with open(f'{fname}_names.csv', mode='r') as file:
            reader = csv.reader(file)
            
            # Read the data row by row and append it to the list
            for row in reader:
                text.extend(row)

        return embeddings, text
    else:
        embeddings = None
        for item in text:
            embedding = get_embedding(item)
            embeddings = vstack_if_exists(embeddings, embedding)
        
        np.savetxt(f'{fname}_embeddings.csv', embeddings, delimiter=',')
        with open(f'{fname}_names.csv', mode='w') as file:
            writer = csv.writer(file)
            writer.writerow(text)
        return embeddings, text




def cosine_distance(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)) 

def get_labels(sentence_embs, class_embs, classes):

    sentence_embs  = sentence_embs / np.linalg.norm(sentence_embs, axis=1)[:, np.newaxis]
    class_embs = class_embs / np.linalg.norm(class_embs, axis=1)[:, np.newaxis]
    sims = sentence_embs @ class_embs.T # compute cosine similarity

    labels = np.argmax(sims, axis=1)
    str_labels = [classes[i] for i in labels]
    return labels, str_labels



ingredients = ["1 cup flour",
                  "2 tablespoons sugar",
                  "3 eggs",
                  "1/2 teaspoon salt",
                  "2 cups milk",
                  "1 tablespoon olive oil",
                  "2 lbs bone-in dark meat chicken"]

if __name__ == "__main__":
    #queries = ['flour', 'sugar', 'eggs', 'butter', 'chicken', 'salt']

    measures = ['cups', 'tablespoons', 'teaspoons', 'pounds', 'grams', 'milliliters', 'other']

    ie, ingredients = get_embeddings(ingredients, 'output/ingredients')
    ce, queries = get_embeddings(measures, 'output/measures')

    labels, str_labels = get_labels(ie, ce, queries)

    print(str_labels)