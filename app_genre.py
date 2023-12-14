import streamlit as st
from movies_retrieval_system_genre.functions import calculate_similarity_with_tfid, get_similar_movies
from movies_retrieval_system_genre.data_processing import process_movie_data
import requests
import pandas as pd

from functools import lru_cache
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
    with st.spinner("Loading..."):
        keywords = user_query.replace(" ", "%2C")

        solr_url = "http://localhost:8983/solr/movies_retrieval_core/select?fl=*%2Cscore&indent=true&q.op=OR&q=Title%3A" + \
              keywords + "&rows=10&useParams="

        result = requests.get(url=solr_url).json()
        movies = result.get('response', {}).get('docs', [])

        result_list = []
        for idx, movie in enumerate(movies[:10], start=1):
            if idx > 10:
                break  # Stop after displaying the top 10 movies

            title = movie.get('Title', [''])[0]
            year = str(movie.get('Release_Year', [''])[0])
            score = float(movie.get('score', 0.0))  # Assuming score is a numerical value

            result_list.append({'Release Year': year, 'Title': title, 'score': score})

        return pd.DataFrame(result_list)
# Streamlit UI
st.title("Movie Information Retrieval")

# User input
st.session_state.query = st.text_input("Enter Query:")
retrieval_method = st.radio("Select Ranking Method:", ["Solr", "Cosine Similarity", "Jaccard Similarity"])

# Search button

def get_movies(query):
    # Retrieve movies based on user input and selected method when the button is clicked
    if not query:
        st.warning("Please enter a query.")
    elif retrieval_method == "Solr":
        # Implement Solr retrieval logic here if needed
        st.session_state.results = retrieve_movies_solr(query)
        pass
    elif retrieval_method == "Jaccard Similarity":
        # Retrieve movies using Jaccard Similarity
        st.session_state.results = retrieve_movies_jaccard_similarity(query)
    else:
        st.session_state.results = retrieve_movies_jaccard_similarity(query)


def show():
    st.session_state.results = None
    st.button("Search", on_click=get_movies(st.session_state.query))

    if st.session_state.results is not None:
        print(st.session_state.results)
        movie_titles_year = st.session_state.results.head(20)
        movie_titles_year['MovieInfo'] = st.session_state.results['Title'] + ' (' + st.session_state.results['Release Year'].astype(str) + ')'

    # Create a dropdown for the user to select a movie
        st.session_state.selected_movie = st.selectbox("Select a movie:", movie_titles_year['MovieInfo'].tolist(), index=None, placeholder="Select a movie",)
        st.write(f"Selected Movie: {st.session_state.selected_movie}")
        # Display details of the selected movie
        if st.button("Show Similar Movies") and st.session_state.selected_movie is not None:
            selected_movie_title, release_year_str = st.session_state.selected_movie.split('(')
            selected_movie_title = selected_movie_title.strip()
            release_year = release_year_str.rstrip(')').strip()

            # selected_movie_details = movie_titles_year[(movie_titles_year['Release Year'] == int(release_year)) &
            #                                            (movie_titles_year['Title'] == selected_movie_title.strip())].iloc[0]

            st.header("10 Similar Recommended Movies:")

            # Check if similar movies list is not empty
            similar_movies = get_similar_movies(selected_movie_title, release_year, st.session_state.results)
            if not similar_movies.empty:
                for idx, movie in enumerate(similar_movies.itertuples(), start=1):
                    if idx > 10:
                        break  # Stop after displaying the top 10 movies

                    release_year = movie._1
                    title = getattr(movie, 'Title', '')

                    st.write(f"{idx}. {title} ({release_year})")
            else:
                st.info("No similar movies found.")



show()
