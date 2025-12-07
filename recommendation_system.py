
##Create a complete academic prototype for a Movie and Book Recommendation System using Python.

# Project requirements:
# - This is a university AI/ML project, not a production web app.
# - Focus on data collection, preprocessing, model training, and recommendation output.
# - Use publicly available datasets (MovieLens for movies, Kaggle/Goodreads-style dataset for books).
# - Code must be simple, clean, and easy to explain for a viva/oral exam.

# Technical requirements:
# 1. Load movie and book datasets using pandas.
# 2. Perform data cleaning (remove null values, duplicates).
# 3. Select relevant features:
#    - Movies: title, genres, user ratings
#    - Books: title, author, genre, ratings
# 4. Convert text features into numerical form using TF-IDF.
# 5. Build a Content-Based Recommendation System using cosine similarity.
# 6. Train the model on the prepared dataset.
# 7. Generate top-N recommendations based on a given movie or book.
# 8. Print sample recommendation outputs in the console.

# Project structure:
# - data/ (datasets)
# - recommendation_system.py or notebook.ipynb
# - requirements.txt
# - README.md explaining aim, objectives, dataset, model, and workflow

# Additional instructions:
# - Add clear comments explaining each step.
# - Keep the math and ML concepts simple.
# - Ensure the output proves that the AI model works.
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
# Load datasets
movies = pd.read_csv('data/movies.csv')  # MovieLens dataset  
books = pd.read_csv('data/books.csv')    # Goodreads-style dataset
# Data Cleaning
movies.dropna(subset=['title', 'genres', 'user_ratings'], inplace=True)
movies.drop_duplicates(inplace=True)
books.dropna(subset=['title', 'author', 'genre', 'ratings'], inplace=True)
books.drop_duplicates(inplace=True)
# Feature Selection
movie_features = movies[['title', 'genres', 'user_ratings']]
book_features = books[['title', 'author', 'genre', 'ratings']]
# Text to Numerical Conversion using TF-IDF
tfidf_movies = TfidfVectorizer(stop_words='english')
movie_features['genres'] = movie_features['genres'].fillna('')
tfidf_matrix_movies = tfidf_movies.fit_transform(movie_features['genres'])
tfidf_books = TfidfVectorizer(stop_words='english')
book_features['genre'] = book_features['genre'].fillna('')
tfidf_matrix_books = tfidf_books.fit_transform(book_features['genre'])
# Content-Based Recommendation System using Cosine Similarity
cosine_sim_movies = cosine_similarity(tfidf_matrix_movies, tfidf_matrix_movies)
cosine_sim_books = cosine_similarity(tfidf_matrix_books, tfidf_matrix_books)
# Function to get movie recommendations
def get_movie_recommendations(title, cosine_sim=cosine_sim_movies):
    idx = movie_features[movie_features['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Get top 10 recommendations
    movie_indices = [i[0] for i in sim_scores]
    return movie_features['title'].iloc[movie_indices]
# Function to get book recommendations  
def get_book_recommendations(title, cosine_sim=cosine_sim_books):
    idx = book_features[book_features['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Get top 10 recommendations
    book_indices = [i[0] for i in sim_scores]
    return book_features['title'].iloc[book_indices]
# Sample Outputs
print("Movie Recommendations for 'The Matrix':") # Example movie title
print(get_movie_recommendations('The Matrix'))  
print("\nBook Recommendations for 'The Great Gatsby':")
print(get_book_recommendations('The Great Gatsby'))
print("\nRecommendation system executed successfully.")
