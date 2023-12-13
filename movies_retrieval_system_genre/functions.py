from sklearn.metrics.pairwise import cosine_similarity
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import Levenshtein as lev
import pandas as pd
import numpy as np
from sklearn.metrics import jaccard_score
from sklearn.feature_extraction.text import CountVectorizer
from itertools import chain, combinations

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

# Function to preprocess a list of words
def preprocess(text):
    # Tokenize and remove stopwords
    tokens = word_tokenize(text)
    filtered_tokens = [token for token in tokens if token.lower() not in STOPWORDS]

    # Remove tokens shorter than 3 characters
    result = [token for token in filtered_tokens if len(token) > 2]

    # Lemmatize remaining tokens using WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = ' '.join([lemmatizer.lemmatize(token, pos='v') for token in result])

    return lemmatized_tokens


# Function to calculate jaccard similarity
def calculate_jaccard_similarity(user_vector, movie_vectors):
    # Convert sparse matrices to dense arrays
    user_array = user_vector.toarray().flatten()
    movie_array = movie_vectors.toarray()

    # Calculate Jaccard similarity
    intersection = np.sum(np.minimum(user_array, movie_array), axis=1)
    union = np.sum(np.maximum(user_array, movie_array), axis=1)
    jaccard_similarity = intersection / union

    return jaccard_similarity

def levenstein_distances(preprocessed_user_input, movie_df):
    # Calculate Levenshtein distance for each movie title
    distances = [lev.distance(preprocessed_user_input, title) for title in movie_df['Preprocessed_title']]

    # Add distances to the DataFrame
    movie_df['Levenstein distance'] = distances
    return movie_df


# Function to create inverted index
def create_inverted_index(df):
    inverted_index = defaultdict(list)

    for idx, row in df.iterrows():
        preprocessed_plot = preprocess(row['Title'])
        tokens = word_tokenize(preprocessed_plot)

        for token in set(tokens):
            inverted_index[token].append(idx)

    return inverted_index

def calculate_similarity_with_tfid(user_input_value, movie_df):
    # Preprocess movie plots
    movie_df['Preprocessed_title'] = movie_df['Title'].apply(preprocess)

    #Apply levenstein distance
    levenstein_distances(user_input_value, movie_df)

    # Apply TF-IDF vectorization
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(movie_df['Preprocessed_title'])

    # Vectorize user input
    user_vector = vectorizer.transform([user_input_value])

    # Calculate cosine similarity
    similarity_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()

    # Add similarity scores to the DataFrame
    similarity_column = f'Similarity (TF-IDF)'
    movie_df[similarity_column] = similarity_scores

    # Sort by similarity scores in descending order
    movie_df = movie_df.sort_values(by=[similarity_column, 'Levenstein distance'], ascending=[False, True])

    return movie_df[['Release Year', 'Title', 'Genre', similarity_column, 'Levenstein distance', 'Preprocessed_title']]


def calculate_similarity_with_jaccard(user_input_value, movie_df):
    # Preprocess movie titles
    movie_df['Preprocessed_title'] = movie_df['Title'].apply(preprocess)

    # Apply levenstein distance
    levenstein_distances(user_input_value, movie_df)

    # Convert the preprocessed text into binary vectors (1 if the word is present, 0 otherwise)
    vectorizer = CountVectorizer(binary=True)
    user_vector = vectorizer.fit_transform([user_input_value])
    movie_vectors = vectorizer.transform(movie_df['Preprocessed_title'])

    # Calculate Jaccard similarity
    similarity_scores = calculate_jaccard_similarity(user_vector, movie_vectors)

    # Add similarity scores to the DataFrame
    similarity_column = 'Similarity (Jaccard)'
    movie_df[similarity_column] = similarity_scores

    # Sort by similarity scores in descending order
    movie_df = movie_df.sort_values(by=[similarity_column, 'Levenstein distance'], ascending=[False, True])

    return movie_df[['Release Year', 'Title', 'Genre', similarity_column, 'Levenstein distance', 'Preprocessed_title']]


def get_similar_movies(selected_movie_title, release_year, results):
    selected_movie_title = selected_movie_title.strip()
    selected_movie_genre = results.loc[(results['Title'] == selected_movie_title) &
                                       (results['Release Year'] == int(release_year))].iloc[0]['Genre']

    selected_genres = [genre.strip() for genre in selected_movie_genre.split(',')]

    # Generate all possible combinations of genres
    all_combinations = list(powerset(selected_genres))

    # Convert each combination back to a string
    all_combination_strings = [', '.join(combination) for combination in reversed(all_combinations) if combination]

    movies_genre_df = pd.DataFrame()
    for combination_string in all_combination_strings:
        movie = results[results["Genre"].str.contains(combination_string, case=False, na=False)]
        movies_genre_df = pd.concat([movies_genre_df, movie], ignore_index=True)

    movies_genre_df = movies_genre_df.drop(movies_genre_df[
                                               (movies_genre_df['Title'] == selected_movie_title) & (
                                                       movies_genre_df['Release Year'] == int(release_year))
                                               ].index)

    return movies_genre_df.head(10)