import streamlit as st
from movies_retrieval_system.functions import calculate_similarity_with_tfid_using_inverted_index
from movies_retrieval_system.data_processing import process_movie_data
import requests
# Load processed data
movies_df, inverted_index_list = process_movie_data()

# Define your information retrieval function
def retrieve_movies_cosine_similarity(user_query):
    with st.spinner("Loading..."):
        result = calculate_similarity_with_tfid_using_inverted_index(user_query, movies_df, inverted_index_list)
    return result

def retrieve_movies_solr(user_query):
    with st.spinner("Loading..."):
        keywords = user_query.replace(" ", "%2C")

        #print(keywords)

        solr_url = "http://localhost:8983/solr/movies_retrieval_core/select?fl=*%2Cscore&indent=true&q.op=OR&q=Plot%3A" + \
              keywords + "&rows=10&useParams="

        #print(solr_url)

        result = requests.get(url=solr_url).json()
        movies = result.get('response', {}).get('docs', [])

        result_list = ""
        for idx, movie in enumerate(movies[:10], start=1):
            if idx > 10:
                break  # Stop after displaying the top 10 movies

            title = movie.get('Title', [''])[0]
            year = str(movie.get('Release_Year', [''])[0])
            score = str(movie.get('score', ''))
            result_list += f"{idx}. {title} [{year}] - Similarity: {score}\n"

    return result_list


# Streamlit UI
st.title("Movie Information Retrieval")

# User input
query = st.text_input("Enter Query:")
retrieval_method = st.radio("Select Ranking Method:", ["Solr", "Cosine Similarity"])

# Search button
search_button = st.button("Search")

# Retrieve movies based on user input and selected method when the button is clicked
if search_button:
    if not query:
        st.warning("Please enter a query.")
    else:
        # Retrieve movies based on the selected method
        if retrieval_method == "Solr":
            results = retrieve_movies_solr(query)
        else:
            results = retrieve_movies_cosine_similarity(query)

        # Display results
        st.header("Top 10 Movies:")
        if results:
            st.write(results)
        else:
            st.warning("No movies found.")
