"""
TMDB-backed dynamic lookup + local-DB recommendation helpers.

Provides:
- get_recommendations(title, top_n=10)
- search_and_add_movie_from_api(title)

Assumptions:
- A global pandas DataFrame called `movies` exists (or will be bound) with columns:
  ['title', 'genres', 'vote_average', 'plot_summary']
- A TMDB API key is available via environment variable TMDB_API_KEY (or set TMDB_API_KEY constant).
- A TF-IDF vectorizer over movies['genres'] is used for content-similarity and stored
  in globals `tfidf_movies` and `tfidf_matrix_movies`. This code will initialize/fit them
  on demand if they are missing.

Notes:
- This implementation avoids refitting the TF-IDF on the entire corpus when adding a movie.
  Instead it transforms the new movie's genre text using the existing vectorizer (if fitted)
  and appends the vector to the TF-IDF matrix. If no vectorizer exists yet, it fits one.
- For TMDB genre names, we call the movie details endpoint to retrieve genre names.
"""

import os
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import vstack
from scipy import sparse

# -----------------------------
# Globals (expected to be present or will be created on first call)
# -----------------------------
# Example placeholder. In your code, you likely already have `movies` DataFrame.
# movies = pd.read_csv("data/movies.csv")  # <-- your real dataset load happens elsewhere
try:
    movies  # if already defined in module where this file is imported, use it
except NameError:
    movies = pd.DataFrame(columns=["title", "genres", "vote_average", "plot_summary"])

# TF-IDF related globals
tfidf_movies = None           # TfidfVectorizer instance (fitted)
tfidf_matrix_movies = None    # sparse matrix of TF-IDF vectors (rows align with movies DataFrame)


# -----------------------------
# Helper: ensure TF-IDF is ready and consistent with `movies`
# -----------------------------
def _ensure_tfidf():
    """
    Make sure tfidf_movies and tfidf_matrix_movies exist and align with `movies`.
    If tfidf_movies is not fitted yet or tfidf_matrix_movies is None, fit on current movies['genres'].
    """
    global tfidf_movies, tfidf_matrix_movies, movies

    # Prepare genre texts
    genre_texts = movies["genres"].fillna("").astype(str).tolist()

    if tfidf_movies is None or tfidf_matrix_movies is None:
        # Fit a new vectorizer on existing genres
        tfidf_movies = TfidfVectorizer(stop_words="english")
        if len(genre_texts) > 0:
            tfidf_matrix_movies = tfidf_movies.fit_transform(genre_texts)
        else:
            # No items yet; create empty tfidf_matrix (0 x vocab)
            tfidf_matrix_movies = None
    else:
        # We assume vectorizer already fitted and tfidf_matrix aligned.
        # If mismatch in rows is detected, re-fit as a fallback.
        if tfidf_matrix_movies.shape[0] != len(genre_texts):
            tfidf_movies = TfidfVectorizer(stop_words="english")
            tfidf_matrix_movies = tfidf_movies.fit_transform(genre_texts)


# -----------------------------
# Function: search TMDB and append single movie to local DataFrame
# -----------------------------
def search_and_add_movie_from_api(title):
    """
    Search TMDB for `title`. If a movie is found, extract fields and append a
    single-row DataFrame to the global `movies` DataFrame and update TF-IDF structures.

    Returns:
        new_title (str) if movie added (normalized title string),
        None if no result found or on error.
    """
    global movies, tfidf_movies, tfidf_matrix_movies

    TMDB_API_KEY = os.getenv("TMDB_API_KEY")
    if not TMDB_API_KEY:
        print("TMDB_API_KEY not set in environment. Cannot query TMDB.")
        return None

    session = requests.Session()
    base = "https://api.themoviedb.org/3"

    # 1) Search endpoint
    search_url = f"{base}/search/movie"
    params = {"api_key": TMDB_API_KEY, "query": title, "page": 1, "include_adult": False}
    try:
        resp = session.get(search_url, params=params, timeout=10)
        resp.raise_for_status()
        results = resp.json().get("results", [])
    except Exception as e:
        print(f"TMDB search request failed: {e}")
        return None

    if not results:
        print(f"No TMDB results for title: {title}")
        return None

    # Use the top search result
    top = results[0]
    movie_id = top.get("id")
    if not movie_id:
        print("TMDB search result missing movie id.")
        return None

    # 2) Get movie details for genre names and full overview
    details_url = f"{base}/movie/{movie_id}"
    params = {"api_key": TMDB_API_KEY}
    try:
        details_resp = session.get(details_url, params=params, timeout=10)
        details_resp.raise_for_status()
        details = details_resp.json()
    except Exception as e:
        print(f"TMDB details request failed: {e}")
        return None

    # Extract desired fields (safe access with fallbacks)
    tmdb_title = details.get("title") or top.get("title") or title
    genres_list = details.get("genres") or []  # list of {"id":..., "name":...}
    genres_names = [g.get("name", "") for g in genres_list if g.get("name")]
    genres_str = ", ".join(genres_names) if genres_names else (top.get("genre_ids") and "") or ""
    vote_average = details.get("vote_average") if details.get("vote_average") is not None else top.get("vote_average")
    try:
        vote_average = float(vote_average) if vote_average is not None else None
    except Exception:
        vote_average = None
    plot_summary = details.get("overview") or top.get("overview") or ""

    # Build the new single-row DataFrame
    new_entry = {
        "title": str(tmdb_title).strip(),
        "genres": genres_str,
        "vote_average": vote_average,
        "plot_summary": plot_summary,
    }

    # Check for duplicate by title (case-insensitive)
    if ((movies["title"].astype(str).str.lower() == new_entry["title"].lower()).any()):
        print(f"Movie '{new_entry['title']}' already exists in local DB (case-insensitive match). Will not add.")
        # Find and return the existing normalized title instead
        existing_title = movies.loc[movies["title"].astype(str).str.lower() == new_entry["title"].lower(), "title"].iloc[0]
        return existing_title

    # Append the new entry to the DataFrame
    movies = pd.concat([movies, pd.DataFrame([new_entry])], ignore_index=True, sort=False)

    # Update TF-IDF structures:
    # - If tfidf has been fitted before, transform new genres and vstack to tfidf_matrix_movies.
    # - If not, fit anew on the updated movies['genres'].
    try:
        if tfidf_movies is None or tfidf_matrix_movies is None:
            # Fit on the full corpus now that movie is appended
            _ensure_tfidf()
        else:
            # Transform just the new movie's genre text (uses existing vocabulary)
            new_text = pd.Series([new_entry["genres"]]).fillna("").astype(str).tolist()
            new_vec = tfidf_movies.transform(new_text)  # returns sparse matrix with 1 row
            if tfidf_matrix_movies is None:
                tfidf_matrix_movies = new_vec
            else:
                # Append (vstack) the new row so indices align with `movies`
                tfidf_matrix_movies = vstack([tfidf_matrix_movies, new_vec])
    except Exception as e:
        # On any failure, fallback to re-fitting fully so matrix remains consistent
        print(f"Warning: incremental TF-IDF update failed ({e}), refitting TF-IDF on whole corpus.")
        tfidf_movies = TfidfVectorizer(stop_words="english")
        tfidf_matrix_movies = tfidf_movies.fit_transform(movies["genres"].fillna("").astype(str).tolist())

    print(f"Added movie '{new_entry['title']}' to local DB from TMDB (id={movie_id}).")
    return new_entry["title"]


# -----------------------------
# Function: get_recommendations (refactored to use dynamic TMDB lookup)
# -----------------------------
def get_recommendations(title, top_n=10):
    """
    Return a list of recommended movie titles for `title`.

    Workflow:
      1. Check if `title` exists in `movies` (case-insensitive).
      2. If not found, attempt to fetch via TMDB using search_and_add_movie_from_api(title).
      3. If the movie still isn't available, return None.
      4. Compute cosine similarity between the title row and the rest of the corpus
         using TF-IDF over `genres` and return top_n titles (excluding the input title).

    Returns:
      list[str] of recommended titles (may be shorter than top_n),
      or None if the movie could not be located/fetched.
    """
    global movies, tfidf_movies, tfidf_matrix_movies

    # Normalize input for case-insensitive matching
    title_norm = str(title).strip()
    if not title_norm:
        print("Empty title provided.")
        return None

    # Ensure TF-IDF structures exist or can be built
    _ensure_tfidf()

    # Try to find the title in the local DataFrame (case-insensitive)
    matches = movies[movies["title"].astype(str).str.lower() == title_norm.lower()]
    if matches.empty:
        # Not found locally -> attempt to fetch from TMDB and add
        added_title = search_and_add_movie_from_api(title_norm)
        if not added_title:
            print(f"Could not find '{title}' locally or via TMDB.")
            return None
        # Recompute matches (title now present)
        matches = movies[movies["title"].astype(str).str.lower() == added_title.strip().lower()]
        if matches.empty:
            print("Unexpected: movie added but cannot be found in local DataFrame.")
            return None

    # Now we have at least one matching row; use the first match
    idx = int(matches.index[0])

    # Ensure tfidf_matrix_movies aligns with movies rows
    _ensure_tfidf()
    if tfidf_matrix_movies is None or tfidf_matrix_movies.shape[0] != len(movies):
        # Fall back to building TF-IDF from scratch to ensure alignment
        tfidf_movies = TfidfVectorizer(stop_words="english")
        tfidf_matrix_movies = tfidf_movies.fit_transform(movies["genres"].fillna("").astype(str).tolist())

    # Compute cosine similarities between the chosen movie row and all movies
    # Use sparse matrices efficiently: tfidf_matrix_movies[idx] is 1xN sparse
    try:
        target_vec = tfidf_matrix_movies[idx]
        sims = cosine_similarity(target_vec, tfidf_matrix_movies).flatten()  # 1D array
    except Exception as e:
        print(f"Similarity computation failed: {e}")
        return None

    # Enumerate indices and similarities, sort descending
    sim_scores = list(enumerate(sims))
    # Exclude the item itself
    sim_scores = [s for s in sim_scores if s[0] != idx]
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Pick top_n and map to titles
    top_indices = [i for i, score in sim_scores[:top_n]]
    recommendations = movies["title"].iloc[top_indices].tolist()

    return recommendations