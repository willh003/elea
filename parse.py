import streamlit as st
from bs4 import BeautifulSoup
from embed import get_embedding, cosine_distance
from utils import vstack_if_exists, class_data_from_csv
import requests
from config import *
import numpy as np
from fractions import Fraction

@st.cache_data
def get_ingredient_embeddings_from_url(url, use_model=True):
    """
    Inputs:
        url: a valid url to a recipe site (string)
        use_model: false to generate random embeddings (avoid model query)
    Returns:
        numpy array of embeddings (size N x embed_dim)
    """

    html = get_html_from_url(url)
    names = get_ingredient_names(html)
    amounts, units = get_ingredient_amounts(html)
    print(list(zip(amounts,units)))
    if use_model:
        embeddings = get_embeddings(names)
    else:
        embeddings = np.random.random((len(names),TEXT_EMBED_DIM))
    return names, embeddings, amounts, units

def get_html_from_url(url):
    r = requests.get(url)
    return r.text

def get_ingredient_names(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find all <span> elements with data-ingredient-name="true" attribute
    ingredient_spans = soup.find_all('span', attrs={'data-ingredient-name': 'true'})

    # Extract the text from each <span> element
    ingredient_names = [span.text for span in ingredient_spans]
    return ingredient_names

def get_ingredient_amounts(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find all <span> elements with data-ingredient-name="true" attribute
    quantity_spans = soup.find_all('span', attrs={'data-ingredient-quantity': 'true'})
    unit_spans = soup.find_all('span', attrs={'data-ingredient-unit': 'true'})

    # Extract the text from each <span> element
    ingredient_quantities = [span.text for span in quantity_spans]

    ingredient_units = [span.text for span in unit_spans]
    return ingredient_quantities, ingredient_units

@st.cache_data
def get_embeddings(text):
    embeddings = None
    for item in text:
        embedding = get_embedding(item)
        embeddings = vstack_if_exists(embeddings, embedding)
    return embeddings

def get_ingredients_kg(names, embeddings, quantities, units):
    kgs = []
    for n, e, q, u in zip(names, embeddings, quantities, units):
        if q == '':
            kgs.append(0)
        elif u == '':
            print(q)
            kgs.append(parse_amount(q,n,e))
        else:
            print(q)
            kgs.append(parse_amount(q,n,e, u))
    return kgs

def parse_amount(quantity,item,item_embedding, measure=None):
    """
    Given a quantity and item, get the equivalent kg of the item. If there is a measure for the item, do it in terms of the measure. Otherwise, do it in terms of discrete numbers of that item
    """
    try:
        if len(quantity) == 1:
            clean_quantity = fraction_char(quantity)
        else:
            if '/' in quantity:
                clean_quantity = convert_mixed_fraction_to_float(mixed_fraction)
            else:
                clean_quantity = sum([fraction_char(c) for c in quantity.split(' ')])
        if measure:
            return parse_measurable(clean_quantity, measure, item, item_embedding)
        else:
            # get nearest ingredient from list of most common discrete ingredients
            return parse_discrete(clean_quantity, item, item_embedding)
    except Exception as e:
        print(str(e))
        return 0
    
def convert_mixed_fraction_to_float(mixed_fraction):
    parts = mixed_fraction.split()
    if len(parts) == 1:
        return float(parts[0])
    elif len(parts) == 2:
        whole_part = float(parts[0])
        fractional_part = Fraction(parts[1])
        return whole_part + fractional_part


def parse_measurable(quantity, measure, item, item_embedding):
    """
    Calculates the equivalent kg of an item given by its volume or weight
    Inputs:
        quantity: float, representing the amount (in units of measure) of the item
        measure
        item
        item_embedding

    """

    # common volumes from https://en.wikibooks.org/wiki/Cookbook:Units_of_measurement 
    abbr = {"tsp": "teaspoon", "tbsp": "tablespoon", "fl oz": "fluid ounce",
            "c": "cup", "pt": "pint", "qt": "quart", "gal": "gallon", "l": "liter", "ml": "milliliter"}
    volumes = ["tsp", "teaspoon", "tbsp", "tablespoon", "fl oz", "fluid ounce",
                "c", "cup", "pt", "pint", "qt", "quart", "gal", "gallon", "ml", "milliliter", "l", "liter"]
    
    weights_to_g = {"mg": .001, "kg": 1000, "kilogram": 1000, "kilo": 1000, "milligram": 1000, "pound": 453, "lb":453, "ounce": 28, "oz":28}
    
    if measure[-1] == "s":
        measure = measure[:-1]
    if measure in abbr:
        measure = abbr[measure]

    
    liquid_volumes_to_g = {"teaspoon": 5, "tablespoon": 14.8, "fluid ounce": 30,
             "cup": 200, "pint": 450, "quart": 900, "gallon": 3500, "milliliter": 1, "liter":1000}
    
    # average dry densities from https://www.engineeringtoolbox.com/foods-materials-bulk-density-d_1819.html were .75 g/cm^3
    dry_volumes_to_g = {k: v * .75 for (k, v) in zip(liquid_volumes_to_g.keys(), liquid_volumes_to_g.values())}

    # get if liquid or dry ingredient using embedding

    if measure in volumes:
        if get_is_liquid(item_embedding, get_liquid_embeddings()):
            return liquid_volumes_to_g[measure] * quantity / 1000
        else: 
            return dry_volumes_to_g[measure] * quantity / 1000
    elif measure in weights_to_g:
        return weights_to_g[measure] * quantity / 1000
    else:
        print(f"WARNING: MEASURE {measure} NOT FOUND FOR ITEM {item}")
        return parse_discrete(quantity, item, item_embedding)

def parse_discrete(quantity, item, item_embedding):
    """
    Calculates the total amount in kg of some quantity of a discrete item (i.e., 2 tomatoes)
    Inputs:
        quantity
        item
        item_embedding
    """

    discrete_embeddings, discrete_items = get_discrete_item_embeddings()
    item_embedding /= np.linalg.norm(item_embedding)
    discrete_embeddings = discrete_embeddings / np.linalg.norm(discrete_embeddings, axis=1)[:, np.newaxis]
    similarities = discrete_embeddings @ item_embedding

    closest_idx = np.argmax(similarities)
    closest_item = list(discrete_items.keys())[closest_idx]
    return discrete_items[closest_item] * quantity / 1000

@st.cache_data
def get_liquid_embeddings():
    out = np.load(LIQUID_EMBEDDINGS_FILE)
    return out

@st.cache_data
def get_discrete_item_embeddings():
    return np.load(DISCRETE_ITEM_EMBEDDINGS_FILE), class_data_from_csv(DISCRETE_ITEM_FILE)

def get_is_liquid(item_embedding, liquid_embeddings):
    """
    Inputs:
        item_embedding: the embedding of the query item
        liquid_embeddings: np array, where the first element is the embedding for a liquid item and the second element is the embedding for a dry item
    Returns:
        whether the item should be classified as liquid or dry
    """
    return cosine_distance(item_embedding, liquid_embeddings[0]) < cosine_distance(item_embedding, liquid_embeddings[1])


def fraction_char(c):
    """
    Given a string c representing a number, if it is a special character for 1/x, return that number
    """
    if len(c) == 0:
        return 0
    if len(c) == 1:
        if ord(c) == 188:
            return 1/4
        elif ord(c) == 189:
            return 1/2
        elif ord(c) == 190:
            return 3/4
        elif ord(c) == 8531:
            return 1/3
        elif ord(c) == 8532:
            return 2/3 
        if 48 <= ord(c) <= 57: # if it is a digit
            return int(c)
        return 0
    return int(c)