import streamlit as st
from movies_retrieval_system.functions import calculate_similarity_with_tfid_using_inverted_index
from movies_retrieval_system.data_processing import process_movie_data


# Load processed data
movies_df, inverted_index_list = process_movie_data()

# Define your information retrieval function
def retrieve_movies_cosine_similarity(user_query):
    with st.spinner("Loading..."):
        result = calculate_similarity_with_tfid_using_inverted_index(user_query, movies_df, inverted_index_list)
    return result

def retrieve_movies_solr(user_query):
    return ["Solr Movie 1", "Solr Movie 2", "Solr Movie 3", "Solr Movie 4", "Solr Movie 5", "Solr Movie 6", "Solr Movie 7", "Solr Movie 8", "Solr Movie 9", "Solr Movie 10"]

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
    elif retrieval_method == "Solr":
        # Implement Solr retrieval logic here if needed
        # results = retrieve_movies_solr(query)
        pass
    else:
        # Retrieve movies using Cosine Similarity
        # Retrieve movies based on user input
        results = retrieve_movies_cosine_similarity(query)

    if query:
        # Display results
        st.header("Top 10 Movies:")
        if not results.empty:
            for idx, movie in enumerate(results.itertuples(), start=1):
                if idx > 10:
                    break  # Stop after displaying the top 10 movies

                release_year = movie._1
                title = getattr(movie, 'Title', '')
                similarity_score = movie._3

                st.write(
                    f"{idx}. {title} ({release_year}) - Similarity: {similarity_score:.4f}")


    else:
            st.warning("No movies found.")

