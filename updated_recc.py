# I have a Python-based recommendation system for Books and Movies.
# Currently, the user selects a category (Book/Movie), types a title, and gets recommendations based on similarity.

# Here is my current code:
# [PASTE YOUR CURRENT CODE HERE]

# Please refactor my code to add two specific features required for my final project.

# Feature 1: Advanced Filtering (Genre & Rating)
# Modify the recommendation logic to filter the dataset BEFORE generating recommendations.
# - Add input prompts for the user to specify a "Preferred Genre" and "Minimum Rating" (e.g., 1-10).
# - If the user provides these filters, narrow down the candidate pool to only items that match the genre and meet the rating threshold.
# - Then, perform the similarity search on this filtered data.

# Feature 2: Automated Data Ingestion (Simulation)
# Create a function called `fetch_new_data()` to simulate an automatic update pipeline.
# - Since I cannot use a real live API during the demo, write a function that "mock fetches" a dictionary of new movies/books (e.g., create a dummy list of 2-3 new items inside the function).
# - The function should append these new items to the main DataFrame/dataset automatically.
# - Add a print statement: "System successfully updated with X new entries from external source."

# Please keep the code clean and comment the new sections clearly so I can explain them during my presentation.
"""
Refactored recommendation system with:
- Feature 1: Advanced Filtering (Genre & Minimum Rating) applied BEFORE similarity search.
- Feature 2: Automated Data Ingestion (Simulation) via fetch_new_data().
- Clear comments for presentation.

Provides:
- load_data(movies_path, books_path)
- fetch_new_data(simulate_movies=True, simulate_books=True)
- get_movie_recommendations(title, preferred_genre=None, min_rating=None, top_n=10)
- get_book_recommendations(title, preferred_genre=None, min_rating=None, top_n=10)
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Global datasets (loaded by load_data())
# ---------------------------------------------------------------------------
movies = pd.DataFrame()
books = pd.DataFrame()


# ---------------------------------------------------------------------------
# Data loading & cleaning (call load_data at startup)
# ---------------------------------------------------------------------------
def load_data(movies_path="data/movies.csv", books_path="data/books.csv"):
    """
    Load datasets from CSVs and perform initial cleaning. Call once at app start.
    Returns (movies_df, books_df).
    """
    global movies, books

    # Load (wrap in try/except so module import doesn't fail if files missing)
    try:
        movies = pd.read_csv(movies_path)
    except Exception:
        movies = pd.DataFrame(columns=["title", "genres", "user_ratings"])

    try:
        books = pd.read_csv(books_path)
    except Exception:
        books = pd.DataFrame(columns=["title", "author", "genre", "ratings"])

    # Basic cleaning for movies
    movies = movies.copy()
    for col in ["title", "genres", "user_ratings"]:
        if col not in movies.columns:
            movies[col] = np.nan

    movies.dropna(subset=["title"], inplace=True)
    movies["title"] = movies["title"].astype(str)
    movies["genres"] = movies["genres"].fillna("").astype(str)
    movies["user_ratings"] = pd.to_numeric(movies["user_ratings"], errors="coerce")
    movies.drop_duplicates(subset=["title", "genres"], inplace=True)
    movies = movies.reset_index(drop=True)

    # Basic cleaning for books
    books = books.copy()
    for col in ["title", "author", "genre", "ratings"]:
        if col not in books.columns:
            books[col] = np.nan

    books.dropna(subset=["title"], inplace=True)
    books["title"] = books["title"].astype(str)
    books["genre"] = books["genre"].fillna("").astype(str)
    books["ratings"] = pd.to_numeric(books["ratings"], errors="coerce")
    books.drop_duplicates(subset=["title", "author"], inplace=True)
    books = books.reset_index(drop=True)

    return movies, books


# Attempt initial load (safe if files missing)
try:
    load_data()
except Exception:
    # If CSVs not available, keep empty dataframes; fetch_new_data can add demo items.
    pass


# ---------------------------------------------------------------------------
# Feature 1: Advanced Filtering (Genre & Rating)
# Filtering happens BEFORE similarity computation.
# ---------------------------------------------------------------------------
def _filter_candidates(df, title_col, genre_col, rating_col, preferred_genre=None, min_rating=None):
    """
    Return a filtered copy of df according to preferred_genre (substring, case-insensitive)
    and min_rating (numeric threshold). If None for either, no filtering for that attribute.
    """
    candidates = df.copy()

    if preferred_genre:
        pattern = str(preferred_genre).strip()
        # Use str.contains to support combined genres like "Action|Sci-Fi" or comma separated
        candidates = candidates[candidates[genre_col].str.contains(pattern, case=False, na=False)]

    if min_rating is not None:
        # Keep only rows where rating_col can be converted to number and >= min_rating
        candidates = candidates[pd.to_numeric(candidates[rating_col], errors="coerce").notna()]
        candidates = candidates[candidates[rating_col].astype(float) >= float(min_rating)]

    # Reset index to match TF-IDF matrix row indices
    candidates = candidates.reset_index(drop=True)
    return candidates


def _get_similar_titles_from_candidates(candidates, title, genre_col, top_n=10):
    """
    Compute TF-IDF over the genre_col in candidates, find the title row,
    compute cosine similarity, and return top_n similar titles (excluding the requested title).
    """
    # Find exact title match (case-insensitive)
    title_matches = candidates[candidates["title"].str.lower() == title.strip().lower()]
    if title_matches.empty:
        raise IndexError("Title not found in the (possibly filtered) candidate set.")

    # Build TF-IDF on the candidate genre column
    tfidf = TfidfVectorizer(stop_words="english")
    text_data = candidates[genre_col].fillna("").astype(str)
    tfidf_matrix = tfidf.fit_transform(text_data)

    # Use the first matching index
    idx = int(title_matches.index[0])

    # Compute cosine similarities
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    # Enumerate and sort results
    sim_scores = list(enumerate(cosine_sim))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Exclude the item itself
    sim_scores = [s for s in sim_scores if s[0] != idx]

    # Pick top_n
    top_scores = sim_scores[:top_n]
    indices = [i[0] for i in top_scores]

    return candidates["title"].iloc[indices].tolist()


def get_movie_recommendations(title, preferred_genre=None, min_rating=None, top_n=10):
    """
    Get movie recommendations for a given title.
    preferred_genre: optional string to filter candidates BEFORE similarity calculation.
    min_rating: optional numeric threshold for user_ratings.
    Returns list of titles (strings). Returns empty list if no candidates match filters.
    Raises IndexError if title not found in filtered candidate set.
    """
    global movies
    candidates = _filter_candidates(
        movies,
        title_col="title",
        genre_col="genres",
        rating_col="user_ratings",
        preferred_genre=preferred_genre,
        min_rating=min_rating,
    )

    if candidates.empty:
        return []

    return _get_similar_titles_from_candidates(candidates, title, genre_col="genres", top_n=top_n)


def get_book_recommendations(title, preferred_genre=None, min_rating=None, top_n=10):
    """
    Get book recommendations for a given title.
    preferred_genre: optional string to filter candidates BEFORE similarity calculation.
    min_rating: optional numeric threshold for ratings.
    Returns list of titles (strings). Returns empty list if no candidates match filters.
    Raises IndexError if title not found in filtered candidate set.
    """
    global books
    candidates = _filter_candidates(
        books,
        title_col="title",
        genre_col="genre",
        rating_col="ratings",
        preferred_genre=preferred_genre,
        min_rating=min_rating,
    )

    if candidates.empty:
        return []

    return _get_similar_titles_from_candidates(candidates, title, genre_col="genre", top_n=top_n)


# ---------------------------------------------------------------------------
# Feature 2: Automated Data Ingestion (Simulation)
# ---------------------------------------------------------------------------
def fetch_new_data(simulate_movies=True, simulate_books=True):
    """
    Simulate fetching new data and append entries to the global movies and books DataFrames.

    Returns (movies_added, books_added).
    Prints: "System successfully updated with X new entries from external source."
    """
    global movies, books

    new_movies = []
    new_books = []

    if simulate_movies:
        new_movies = [
            {
                "title": "Mock Movie: Starlight Run",
                "genres": "Sci-Fi|Adventure",
                "user_ratings": 8.1,
            },
            {
                "title": "Mock Movie: Silent Echoes",
                "genres": "Drama|Mystery",
                "user_ratings": 7.2,
            },
        ]

    if simulate_books:
        new_books = [
            {
                "title": "Mock Book: The Last Archive",
                "author": "A. Demo",
                "genre": "Science Fiction|Thriller",
                "ratings": 4.3,
            },
            {
                "title": "Mock Book: Whispers of Rain",
                "author": "B. Sample",
                "genre": "Romance|Contemporary",
                "ratings": 3.9,
            },
        ]

    movies_added = 0
    for entry in new_movies:
        # Avoid exact-title duplicates (case-insensitive)
        if not ((movies["title"].astype(str).str.lower() == entry["title"].lower()).any()):
            movies = pd.concat([movies, pd.DataFrame([entry])], ignore_index=True, sort=False)
            movies_added += 1

    books_added = 0
    for entry in new_books:
        if not ((books["title"].astype(str).str.lower() == entry["title"].lower()).any()):
            books = pd.concat([books, pd.DataFrame([entry])], ignore_index=True, sort=False)
            books_added += 1

    total_added = movies_added + books_added
    print(f"System successfully updated with {total_added} new entries from external source.")
    return movies_added, books_added


# ---------------------------------------------------------------------------
# Demo when run as a script
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Initial counts -> Movies:", len(movies), "Books:", len(books))
    m_added, b_added = fetch_new_data()
    print("After ingestion -> Movies:", len(movies), "Books:", len(books))

    # Try a sample recommendation if possible
    if not movies.empty:
        sample_title = movies.iloc[0]["title"]
        print(f"\nSample movie recommendations for '{sample_title}':")
        try:
            recs = get_movie_recommendations(sample_title, preferred_genre=None, min_rating=None, top_n=5)
            print(recs)
        except IndexError as e:
            print("Movie sample error:", e)

    if not books.empty:
        sample_title = books.iloc[0]["title"]
        print(f"\nSample book recommendations for '{sample_title}':")
        try:
            recs = get_book_recommendations(sample_title, preferred_genre=None, min_rating=None, top_n=5)
            print(recs)
        except IndexError as e:
            print("Book sample error:", e)