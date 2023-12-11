from sklearn.metrics.pairwise import cosine_similarity
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

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

# Function to create inverted index
def create_inverted_index(df):
    inverted_index = defaultdict(list)

    for idx, row in df.iterrows():
        preprocessed_plot = preprocess(row['Plot'])
        tokens = word_tokenize(preprocessed_plot)

        for token in set(tokens):
            inverted_index[token].append(idx)

    return inverted_index

def calculate_similarity_with_tfid(user_input_value, movie_df):
    # Preprocess user input
    preprocessed_user_input = preprocess(user_input_value)

    # Preprocess movie plots
    movie_df['Preprocessed_plot'] = movie_df['Plot'].apply(preprocess)

    # Apply TF-IDF vectorization
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(movie_df['Preprocessed_plot'])

    # Vectorize user input
    user_vector = vectorizer.transform([preprocessed_user_input])

    # Calculate cosine similarity
    similarity_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()

    # Add similarity scores to the DataFrame
    similarity_column = f'Similarity (TF-IDF)'
    movie_df[similarity_column] = similarity_scores

    # Sort by similarity scores in descending order
    movie_df = movie_df.sort_values(by=similarity_column, ascending=False)

    return movie_df[['Release Year', 'Title', similarity_column, 'Preprocessed_plot']]
def calculate_similarity_with_tfid_using_inverted_index(user_input_value, movie_df,inverted_index):
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()

    # Preprocess user input
    preprocessed_user_input = preprocess(user_input_value)

    # Preprocess Plot
    movie_df['Preprocessed_plot'] = movie_df['Plot'].apply(preprocess)

    # Retrieve relevant indices from inverted index
    user_tokens = word_tokenize(preprocessed_user_input)
    relevant_indices = set(idx for token in user_tokens for idx in inverted_index.get(token, []))

    # Extract corresponding rows from the DataFrame
    relevant_movies = movie_df.loc[list(relevant_indices)].reset_index(drop=True)

    # Fit the vectorizer on the relevant subset of the data
    vectorizer.fit(relevant_movies['Preprocessed_plot'])
    # Vectorize user input
    user_vector = vectorizer.transform([preprocessed_user_input])

    # Calculate similarity using TF-IDF
    similarity_scores = cosine_similarity(user_vector,
                                          vectorizer.transform(relevant_movies['Preprocessed_plot'])).flatten()

    # Add similarity scores to the DataFrame
    similarity_column = f'Similarity (TF-IDF)'
    relevant_movies[similarity_column] = similarity_scores

    # Sort by similarity scores in descending order
    relevant_movies = relevant_movies.sort_values(by=similarity_column, ascending=False)

    # Prepare the result string
    result_str = ""
    for idx, (_, movie) in enumerate(relevant_movies.iterrows(), start=1):
        if idx > 10:
            break  # Stop after displaying the top 10 movies

        year = movie['Release Year']
        title = movie['Title']
        score = movie['Similarity (TF-IDF)']
        result_str += f"{idx}. {title} [{year}] - Similarity: {score:.4f}\n"

    return result_str


