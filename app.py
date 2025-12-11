"""
Streamlit app using the refactored recommendation system.

UI additions:
- Preferred Genre input
- Minimum Rating input
- Button to simulate automated data ingestion (fetch_new_data)
- Uses filtered candidate pool for recommendations
"""

import streamlit as st
from updated_recc import (
    load_data,
    fetch_new_data,
    get_movie_recommendations,
    get_book_recommendations,
)

# Load data once at app start
load_data()

st.title("Movie and Book Recommendation System (Advanced Demo)")

option = st.selectbox("Choose Recommendation Type", ("Movie", "Book"))
title_input = st.text_input("Enter the Title (exact title required):")

# Feature 1: Advanced Filtering inputs
preferred_genre = st.text_input("Preferred Genre (optional, e.g., 'Drama' or 'Sci-Fi'):")
# Use 0.0 as sentinel for "no filter" in the UI; convert to None later if unchanged.
min_rating = st.number_input("Minimum Rating (optional, 0 means no filter)", min_value=0.0, max_value=10.0, value=0.0, step=0.1)

st.markdown("### Data Ingestion (Simulation)")
if st.button("Fetch New Data (simulate external update)"):
    movies_added, books_added = fetch_new_data()
    st.success(f"Fetched and added {movies_added} new movies and {books_added} new books.")
    # NOTE: Do NOT call load_data() here because fetch_new_data() appends directly to the module's global DataFrames.
    # Calling load_data() again would re-read CSVs and may overwrite the in-memory simulation additions.

st.write("---")

if st.button("Get Recommendations"):
    if not title_input.strip():
        st.error("Please enter a title.")
    else:
        st.caption("Title matching is case-insensitive but must match exactly one of the titles in the current dataset (after filters).")
        # Interpret min_rating 0.0 as no filter
        min_rating_filter = None if min_rating == 0.0 else min_rating
        try:
            if option == "Movie":
                recommendations = get_movie_recommendations(
                    title_input,
                    preferred_genre=preferred_genre if preferred_genre.strip() else None,
                    min_rating=min_rating_filter,
                    top_n=10,
                )
                if not recommendations:
                    st.info("No candidate movies matched your filters. Try removing filters or lowering the minimum rating.")
                else:
                    st.write("Top Movie Recommendations:")
                    for i, rec in enumerate(recommendations):
                        st.write(f"{i+1}. {rec}")
                    st.success("Recommendations generated successfully")
            else:
                recommendations = get_book_recommendations(
                    title_input,
                    preferred_genre=preferred_genre if preferred_genre.strip() else None,
                    min_rating=min_rating_filter,
                    top_n=10,
                )
                if not recommendations:
                    st.info("No candidate books matched your filters. Try removing filters or lowering the minimum rating.")
                else:
                    st.write("Top Book Recommendations:")
                    for i, rec in enumerate(recommendations):
                        st.write(f"{i+1}. {rec}")
                    st.success("Recommendations generated successfully")

        except IndexError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Unexpected error: {e}")