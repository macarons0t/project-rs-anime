# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import joblib
import requests
import json

# Add the current directory to path to import our modules
sys.path.append(os.path.dirname(__file__))

# Import our custom modules
try:
    from cleaning import clean_and_split_data 
    from recommender_model import UserBasedCollaborativeFilter, train_and_save_model
except ImportError as e:
    st.error(f"Failed to import custom modules. Please ensure 'cleaning.py' and 'recommender_model.py' are in the same directory as this app. Error: {e}")
    st.stop() 

# Streamlit app configuration
st.set_page_config(
    page_title="Anime Recommendation System",
    page_icon="🎌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Configure Ollama ---
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2:latest" # Or whatever model you intend to use and have pulled  # You can change this to any model you have installed

def check_ollama_connection():
    """Check if Ollama is running and accessible."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, None
    except requests.exceptions.RequestException:
        return False, None

def get_available_models():
    """Get list of available Ollama models."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models_data = response.json()
            return [model["name"] for model in models_data.get("models", [])]
        return []
    except requests.exceptions.RequestException:
        return []

@st.cache_data(ttl=86400)  # Cache trivia for a day
def get_anime_trivia_ollama(anime_name: str, anime_genre: str, model_name: str = DEFAULT_MODEL) -> str:
    """
    Queries Ollama to get a fun trivia fact about the anime.
    """
    prompt = (
        f"Provide one concise, fun, and interesting trivia fact about the anime titled '{anime_name}' "
        f"which has the genre(s) '{anime_genre}'. "
        "Keep it to one sentence, around 20-30 words. Be factual and avoid speculation."
    )
    
    try:
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 150  # Limit response length
            }
        }
        
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "No trivia available.").strip()
        else:
            return f"Error: Unable to fetch trivia (Status: {response.status_code})"
            
    except requests.exceptions.Timeout:
        return "Timeout: Trivia request took too long."
    except requests.exceptions.RequestException as e:
        return f"Network error: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

def load_and_clean_data_cached():
    """
    Loads and cleans the anime and rating data.
    This function now strictly relies on the presence of data files.
    """
    anime_path = './data/anime.csv'
    rating_path = './data/rating.csv'
    
    # Check if the data directory exists
    DATA_DIR = './data'
    if not os.path.exists(DATA_DIR):
        st.error(f"Error: Data directory '{DATA_DIR}' not found. Please create it and place your 'anime.csv' and 'rating.csv' files inside.")
        st.stop() 

    with st.spinner("Loading and cleaning data..."):
        train_df, test_df, anime_df = clean_and_split_data(
            anime_path, rating_path,
            min_ratings_per_user=5,  
            min_ratings_per_anime=10
        )
    
    if train_df is None:
        st.error("Failed to load or clean data. This usually means 'anime.csv' or 'rating.csv' are missing, empty, or problematic after filtering. Please ensure they are in the './data/' directory.")
        st.stop() 
    
    st.success("Data loaded and cleaned successfully!")
    return train_df, test_df, anime_df

@st.cache_resource
def load_or_train_model_cached(train_df, anime_df):
    """Load existing model or train a new one."""
    model_path = 'anime_recommender_model.pkl'
    
    model = UserBasedCollaborativeFilter.load_model(model_path)
    
    if model is None:
        st.info("No existing model found or failed to load. Training new model...")
        with st.spinner("Training recommendation model... This may take a moment..."):
            model = train_and_save_model(train_df, anime_df, model_path)
            if model: 
                st.success("Model trained and saved successfully!")
            else:
                st.error("Failed to train the recommendation model.")
                st.stop() 
    
    return model

def display_recommendations(recommendations, title, anime_full_df: pd.DataFrame, selected_model: str):
    """
    Display recommendations in a nice format, including trivia for the top 2.
    
    Args:
        recommendations (List[Tuple]): List of (anime_id, anime_name, rating)
        title (str): Title for the recommendation section
        anime_full_df (pd.DataFrame): The full anime dataframe to get genre information
        selected_model (str): The Ollama model to use for trivia
    """
    if not recommendations:
        st.write("No recommendations available.")
        return
    
    st.subheader(title)
    
    rec_data = []
    for i, (anime_id, anime_name, rating) in enumerate(recommendations, 1):
        rec_data.append({
            '#': i, 
            'Anime Name': anime_name, 
            'Predicted Rating / Avg Rating': f'{rating:.2f}'
        })
    
    df = pd.DataFrame(rec_data)
    
    # Display the DataFrame
    st.dataframe(
        df, 
        use_container_width=True,
        hide_index=True,
        column_config={
            "#": st.column_config.NumberColumn(
                "#",
                width="small"
            ),
            "Anime Name": st.column_config.TextColumn(
                "Anime Name",
                width="large"
            ),
            "Predicted Rating / Avg Rating": st.column_config.TextColumn(
                "Rating",
                width="medium"
            )
        }
    )

    # Add Fun Trivia for the first two recommendations
    st.markdown("---")
    st.markdown("#### ✨ Fun Anime Trivia!")
    
    # Check if Ollama is connected before trying to get trivia
    is_connected, _ = check_ollama_connection()
    
    if not is_connected:
        st.warning("⚠️ Ollama is not running or accessible. Please start Ollama to get anime trivia.")
        st.info("To start Ollama, run: `ollama serve` in your terminal")
        return
    
    if len(recommendations) > 0:
        for i in range(min(2, len(recommendations))):
            anime_id, anime_name, _ = recommendations[i]
            
            # Get genre from the full anime_df
            anime_info = anime_full_df[anime_full_df['anime_id'] == anime_id]
            anime_genre = anime_info['genre'].iloc[0] if not anime_info.empty else "Unknown"

            with st.spinner(f"Fetching trivia for {anime_name}..."):
                trivia = get_anime_trivia_ollama(anime_name, anime_genre, selected_model)
                st.markdown(f"**{i+1}. {anime_name}:** {trivia}")
    else:
        st.info("No recommendations to fetch trivia for.")

def main():
    st.title("🎌 Anime Recommendation System")
    st.markdown("Get personalized anime recommendations based on collaborative filtering!")
    
    # Check Ollama connection and show status
    is_connected, models_info = check_ollama_connection()
    
    if is_connected:
        available_models = get_available_models()
        if available_models:
            st.success(f"✅ Ollama is running! Available models: {', '.join(available_models)}")
        else:
            st.warning("⚠️ Ollama is running but no models found. Please install a model first.")
    else:
        st.error("❌ Ollama is not running. Please start Ollama to enable trivia features.")
        st.info("To start Ollama: `ollama serve` in terminal, then install a model: `ollama pull llama3.1:8b`")
    
    train_df, test_df, anime_df = load_and_clean_data_cached() 
    
    model = load_or_train_model_cached(train_df, anime_df)
    
    st.sidebar.header("Recommendation Options")
    
    # Model selection in sidebar
    if is_connected:
        available_models = get_available_models()
        if available_models:
            selected_model = st.sidebar.selectbox(
                "Select Ollama Model for Trivia",
                available_models,
                index=0 if DEFAULT_MODEL not in available_models else available_models.index(DEFAULT_MODEL)
            )
        else:
            selected_model = DEFAULT_MODEL
            st.sidebar.warning("No models available. Please install a model.")
    else:
        selected_model = DEFAULT_MODEL
        st.sidebar.error("Ollama not connected")
    
    num_users_to_show = 50 
    available_users = sorted(train_df['user_id'].unique())
    displayed_users = available_users[:num_users_to_show]
    
    available_genres = ['Action', 'Adventure', 'Comedy', 'Drama', 'Fantasy', 'Horror', 'Mystery', 'Romance', 'Sci-Fi', 'Slice of Life', 'Sports', 'Supernatural', 'Thriller'] 
    
    rec_type = st.sidebar.selectbox(
        "Choose Recommendation Type",
        ["User-Based Recommendations", "Popular Anime", "Genre-Based Recommendations", "Browse All Anime"]
    )
    
    n_recs = st.sidebar.slider("Number of Recommendations", 5, 20, 10)
    
    if rec_type == "User-Based Recommendations":
        st.header("🎯 Personalized Recommendations")
        
        input_col, button_col = st.columns([3, 1])
        
        with input_col:
            selected_user_label = st.selectbox(
                "Select a User ID (or choose 'New User' for general recommendations)",
                ["New User"] + [f"User {uid}" for uid in displayed_users]
            )
        
        with button_col:
            st.write("") 
            st.write("") 
            get_recs = st.button("Get Recommendations", type="primary", use_container_width=True)
        
        if get_recs:
            if selected_user_label == "New User":
                user_id = 999999999 
            else:
                user_id = int(selected_user_label.split()[-1])
            
            with st.spinner("Generating recommendations..."):
                recommendations = model.get_user_recommendations(user_id, n_recs)
                
            display_recommendations(recommendations, f"Recommendations for {selected_user_label}", anime_df, selected_model)
            
            if selected_user_label != "New User":
                st.subheader(f"📊 {selected_user_label}'s Recent Rating History")
                user_ratings = train_df[train_df['user_id'] == user_id].merge(
                    anime_df[['anime_id', 'name']], on='anime_id'
                ).sort_values('rating', ascending=False)
                
                if not user_ratings.empty:
                    st.write("Top 5 recent ratings:")
                    for _, row in user_ratings.head(5).iterrows():
                        st.write(f"• **{row['name']}** - {row['rating']}/10")
                else:
                    st.write("No rating history found for this user in the training data.")
    
    elif rec_type == "Popular Anime":
        st.header("🔥 Most Popular Anime")
        
        if st.button("Get Popular Anime", type="primary"):
            with st.spinner("Finding popular anime..."):
                popular_recs = model.get_popular_recommendations(n_recs)
                display_recommendations(popular_recs, "Popular Anime (Based on Average Ratings)", anime_df, selected_model)
    
    elif rec_type == "Genre-Based Recommendations":
        st.header("🎭 Browse by Genre")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            selected_genre = st.selectbox("Choose a Genre", available_genres)
        with col2:
            st.write("") 
            st.write("") 
            get_genre_recs = st.button("Get Genre Recommendations", type="primary", use_container_width=True)
        
        if get_genre_recs:
            with st.spinner(f"Finding {selected_genre} anime..."):
                genre_recs = model.get_anime_by_genre(selected_genre, n_recs)
                display_recommendations(genre_recs, f"Top {selected_genre} Anime", anime_df, selected_model)
    
    elif rec_type == "Browse All Anime":
        st.header("📚 Browse All Available Anime")
        
        search_term = st.text_input("Search anime by name:", placeholder="Enter anime name...").strip()
        
        if search_term:
            filtered_anime = anime_df[
                anime_df['name'].str.contains(search_term, case=False, na=False)
            ].sort_values('rating', ascending=False) 
        else:
            filtered_anime = anime_df.sort_values('rating', ascending=False) 
            if len(filtered_anime) > 500:
                filtered_anime = filtered_anime.head(500) 
        
        if not filtered_anime.empty:
            st.subheader("Available Anime")
            
            display_df = filtered_anime[['name', 'genre', 'type', 'episodes', 'rating']].copy()
            display_df.columns = ['Anime Name', 'Genre', 'Type', 'Episodes', 'Rating']
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.write("No anime found matching your search.")
    
    # Data insights sidebar
    with st.sidebar:
        st.markdown("---")
        st.header("📈 Data Insights")
        
        if train_df is not None and anime_df is not None:
            st.metric("Total Users", train_df['user_id'].nunique())
            st.metric("Total Anime", anime_df.shape[0])
            st.metric("Total Ratings", len(train_df))
            st.metric("Avg Rating", f"{train_df['rating'].mean():.2f}")
            
            st.subheader("Rating Distribution")
            rating_counts = train_df['rating'].value_counts().sort_index()
            st.bar_chart(rating_counts)
        
        # Ollama status in sidebar
        st.markdown("---")
        st.header("🤖 AI Status")
        if is_connected:
            st.success("Ollama Connected")
            if available_models:
                st.info(f"Using: {selected_model}")
        else:
            st.error("Ollama Disconnected")
            st.markdown("**Setup Instructions:**")
            st.code("# Install Ollama\nbrew install ollama\n\n# Start Ollama\nollama serve\n\n# Pull a model\nollama pull llama3.1:8b")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### About This Recommendation System
    
    This system uses **User-Based Collaborative Filtering** to provide personalized anime recommendations. Here's how it works:
    
    - **User-Based**: Finds users with similar tastes and recommends anime they liked
    - **Popular Anime**: Shows the highest-rated anime across all users
    - **Genre-Based**: Filters recommendations by your preferred genre
    - **Collaborative Filtering**: Uses the wisdom of the crowd to make predictions
    - **AI Trivia**: Uses local Ollama models to provide fun facts about recommended anime
    
    The system handles new users by providing popular recommendations and genre-based suggestions.
    """)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.info("Please make sure all required files ('cleaning.py', 'recommender_model.py', and data files in './data/') are present and properly formatted.")
        
        with st.expander("Debug Information"):
            st.write("Error details:", str(e))
            st.write("Current working directory:", os.getcwd())
            st.write("Files in current directory:", os.listdir('.'))