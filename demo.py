# I want to create a content-based recommendation system for movies and books. 

# 1. Load CSV files for movies (columns: title, genres, user_ratings) and books (columns: title, author, genre, ratings). 
# 2. Clean the data: remove missing values and duplicates. 
# 3. Use TF-IDF vectorization to convert genres/genre into numerical features. 
# 4. Compute cosine similarity for both movies and books. 
# 5. Create two functions: 
#    - get_movie_recommendations(title) → returns top 10 similar movies. 
#    - get_book_recommendations(title) → returns top 10 similar books. 
# 6. Make a simple Streamlit interface: 
#    - User can choose between Movie or Book. 
#    - Input a title and see top 10 recommendations. 
# 7. Handle exceptions if the title is not found. 
# 8. Print recommendations on the web page. 
# 9. Ensure the code runs end-to-end without errors. 
import streamlit as st
import pandas as pd
from recommendation_system import get_movie_recommendations, get_book_recommendations
st.title("Movie and Book Recommendation System")
option = st.selectbox("Choose Recommendation Type", ("Movie", "Book"))
title_input = st.text_input("Enter the Title:")
if st.button("Get Recommendations"):
    if option == "Movie":
        try:
            recommendations = get_movie_recommendations(title_input)
            st.write("Top 10 Movie Recommendations:")
            for i, rec in enumerate(recommendations):
                st.write(f"{i+1}. {rec}")
        except IndexError:
            st.error("Movie title not found. Please check the title and try again.")
    else:
        try:
            recommendations = get_book_recommendations(title_input)
            st.write("Top 10 Book Recommendations:")
            for i, rec in enumerate(recommendations):
                st.write(f"{i+1}. {rec}")
        except IndexError:
            st.error("Book title not found. Please check the title and try again.")
  
