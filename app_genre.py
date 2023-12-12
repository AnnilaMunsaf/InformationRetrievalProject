import streamlit as st
from movies_retrieval_system_genre.functions import calculate_similarity_with_tfid, get_similar_movies
from movies_retrieval_system_genre.data_processing import process_movie_data
import requests

# Load processed data
movies_df, inverted_index_list = process_movie_data()

# Define your information retrieval function
def retrieve_movies_cosine_similarity(user_query):
    with st.spinner("Loading..."):
        result = calculate_similarity_with_tfid(user_query, movies_df)
    return result

def retrieve_movies_jaccard_similarity(user_query):
    with st.spinner("Loading..."):
        result = calculate_similarity_with_tfid(user_query, movies_df)
    return result

def retrieve_movies_solr(user_query):
    return ["Solr Movie 1", "Solr Movie 2", "Solr Movie 3", "Solr Movie 4", "Solr Movie 5", "Solr Movie 6", "Solr Movie 7", "Solr Movie 8", "Solr Movie 9", "Solr Movie 10"]

# Streamlit UI
st.title("Movie Information Retrieval")

# User input
query = st.text_input("Enter Query:")
retrieval_method = st.radio("Select Ranking Method:", ["Solr", "Cosine Similarity", "Jaccard Similarity"])

# Search button
search_button = st.button("Search")

# Retrieve movies based on user input and selected method when the button is clicked
if search_button:
    if not query:
        st.warning("Please enter a query.")
    elif retrieval_method == "Solr":
        # Implement Solr retrieval logic here if needed
        # results = retrieve_movies_solr(query)
        pass
    elif retrieval_method == "Jaccard Similarity":
        # Retrieve movies using Jaccard Similarity
        # Retrieve movies based on user input
        results = retrieve_movies_jaccard_similarity(query)
        # Retrieve movies using Cosine Similarity
        # Retrieve movies based on user input
    else:
        results = retrieve_movies_jaccard_similarity(query)


    if query:
        # Display results
        if not results.empty:
            movie_titles_year = results.head(20)
            movie_titles_year['MovieInfo'] = results['Title'] + ' (' + results['Release Year'].astype(str) + ')'
            # Create a dropdown for the user to select a movie
            selected_movie = st.selectbox("Select a movie:", movie_titles_year['MovieInfo'].tolist())

            # Display details of the selected movie
            if selected_movie:
                selected_movie_title = selected_movie.split(' (')[0]
                selected_movie_details = movie_titles_year[movie_titles_year['Title'] == selected_movie_title].iloc[0]
                release_year = selected_movie_details['Release Year']
                # similarity_score = selected_movie_details._3

                st.write(f"Selected Movie: {selected_movie}")
                st.header("10 Similar Recommended Movies:")
                similar_movies = get_similar_movies(selected_movie_title, release_year, results)
                for idx, movie in enumerate(similar_movies.itertuples(), start=1):
                    if idx > 10:
                        break  # Stop after displaying the top 10 movies

                    release_year = movie._1
                    title = getattr(movie, 'Title', '')

                    st.write(f"{idx}. {title} ({release_year})")
                        # f"{idx}. {title} ({release_year}) - Similarity: {similarity_score:.4f}")


    else:
            st.warning("No movies found.")


