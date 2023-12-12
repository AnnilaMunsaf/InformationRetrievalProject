import pandas as pd
from movies_retrieval_system.functions import preprocess, create_inverted_index
import streamlit as st

@st.cache_data
def process_movie_data(file_path='./wiki_movie_plots_deduped.csv'):
    # Read the CSV file
    movies_df = pd.read_csv(file_path, usecols=['Release Year', 'Title', 'Plot', 'Genre'])

    # Preprocess the plot column
    movies_df['Preprocessed_Title'] = movies_df['Title'].apply(preprocess)

    # Create inverted index
    inverted_index_list = create_inverted_index(movies_df)

    return movies_df, inverted_index_list
