import streamlit as st
import numpy as np
import pandas as pd
import validators

from keys import OPENAI_KEY
from config import *
from utils import class_data_from_csv
from parse import get_ingredient_embeddings_from_url, get_ingredients_kg
from embed import get_labels


def get_user_input(text):
    inp = st.text_input(text)
    return inp

@st.cache_data
def load_class_embeddings():
    """
    Load the class embeddings (either from a database or text file)
    DO NOT MUTATE OUTPUT OF THIS 
    """
    class_embeddings = np.load(CO2_EMBEDDINGS_FILE)
    return class_embeddings

def data_per_label(labels, data):
    label_data = []
    for label in labels:
        
        label_data.append(data[label])
    return label_data


def calculate_co2():
    url = st.session_state.url_input
    if not validators.url(url):
        st.warning('Cannot Find URL')
        return

    ingredient_names, ingredient_embeddings, ingredient_quantities, ingredient_units = get_ingredient_embeddings_from_url(url, use_model=True)
    if type(ingredient_embeddings) != np.ndarray:
        st.warning('Not a valid recipe')
        return

    ingredient_kgs = get_ingredients_kg(ingredient_names, ingredient_embeddings, ingredient_quantities, ingredient_units)

    class_embeddings = load_class_embeddings()
    class_co2_data = class_data_from_csv(CO2_DATA_FILE)
    _, labels = get_labels(ingredient_embeddings, class_embeddings, 
        list(class_co2_data.keys()))

    co2_per_label = data_per_label(labels, class_co2_data)

    co2_per_ingredient = [weight * co2 for weight, co2 in zip (ingredient_kgs, co2_per_label)]

    total = sum(co2_per_ingredient)
    st.text(f"Total Kilograms C02: {total}")

    data = {"Ingredient": ingredient_names, "CO2 Usage": co2_per_ingredient}
    df = pd.DataFrame(data)
    st.bar_chart(df.set_index("Ingredient"))
    print(list(zip(ingredient_names,labels)))


def main():

    st.text_input("Enter the recipe url: ", key="url_input")

    if st.button('Calculate CO2 Usage'):
        calculate_co2()


if __name__=="__main__":
    main()
