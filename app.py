# app.py
import streamlit as st
from model import load_data, vectorize_genres, build_knn_model, recommend_movie_knn

# Load and prepare data
movies = load_data()
genre_vectors = vectorize_genres(movies)
model_knn = build_knn_model(genre_vectors)

# Streamlit app
st.title('🎬 Movie Recommendation System')
st.write('Enter a movie title or a genre, and we will recommend similar ones!')

# Text input for the user to type the movie title or genre
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""  # Initialize session state if not set

user_input = st.text_input('Enter a movie title or genre', value=st.session_state.user_input)

# Update session state whenever user input changes
if user_input != st.session_state.user_input:
    st.session_state.user_input = user_input

if user_input:
    # Convert input to lowercase for case-insensitive comparison
    user_input_lower = user_input.lower()
    
    # Check if the input is a movie title or genre
    if user_input_lower in movies['title'].dropna().str.lower().values:
        # If it's a movie title, proceed with recommendations based on title
        movie_to_recommend = [movie for movie in movies['title'].dropna() if user_input_lower in movie.lower()]
        
        if movie_to_recommend:
            movie_to_recommend = movie_to_recommend[0]
            if st.button('Recommend'):
                recommendations = recommend_movie_knn(model_knn, genre_vectors, movies, movie_to_recommend)
                st.write('### Recommended Movies:')
                for rec in recommendations:
                    st.write(f"🎥 {rec}")
        else:
            st.write(f"Sorry, no movie found matching '{user_input}'. Try a different name or check the spelling.")
    else:
        # If it's not a title, try matching it as a genre
        matching_genre_movies = movies[movies['genres'].str.contains(user_input_lower, case=False, na=False)]
        
        if not matching_genre_movies.empty:
            if st.button('Recommend'):
                st.write('### Recommended Movies based on Genre:')
                for _, row in matching_genre_movies.iterrows():
                    st.write(f"🎥 {row['title']}")
        else:
            st.write(f"Sorry, no movies found with the genre '{user_input}'. Try a different genre.")
