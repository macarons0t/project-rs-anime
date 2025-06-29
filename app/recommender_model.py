# 2-recommender_model.py
import pandas as pd
import numpy as np
import joblib
import sys
import os
from typing import List, Tuple, Optional

# Import the clean_and_split_data function from your 1-cleaning.py script
# Make sure 1-cleaning.py is in the same directory or accessible via Python path
from cleaning import clean_and_split_data 

class UserBasedCollaborativeFilter:
    """
    Enhanced User-Based Collaborative Filtering model that predicts ratings
    and generates anime recommendations with actual anime names.
    """
    def __init__(self, train_matrix: pd.DataFrame = None, anime_df: pd.DataFrame = None):
        """
        Initializes the UserBasedCollaborativeFilter.

        Args:
            train_matrix (pd.DataFrame, optional): The user-item matrix
            anime_df (pd.DataFrame, optional): DataFrame with anime information including names
        """
        self.train_matrix = train_matrix
        self.anime_df = anime_df
        self.global_mean_rating = None
        
    def train(self, train_df: pd.DataFrame, anime_df: pd.DataFrame):
        """
        Trains the model by creating the user-item matrix and storing anime data.

        Args:
            train_df (pd.DataFrame): DataFrame containing user_id, anime_id, and rating
            anime_df (pd.DataFrame): DataFrame containing anime information
        """
        print("Training model: Creating user-item matrix...")
        self.train_matrix = train_df.pivot(index='user_id', columns='anime_id', values='rating')
        self.anime_df = anime_df
        self.global_mean_rating = train_df['rating'].mean()
        print(f"User-item matrix created with shape: {self.train_matrix.shape}")
        print(f"Global mean rating: {self.global_mean_rating:.2f}")
        print("Training complete.")

    def predict_rating(self, user_id: int, anime_id: int) -> float:
        """
        Predicts the rating for a given user and anime.
        
        Args:
            user_id (int): The ID of the user
            anime_id (int): The ID of the anime

        Returns:
            float: The predicted rating
        """
        if self.train_matrix is None:
            return self.global_mean_rating if self.global_mean_rating else 7.0

        # If anime doesn't exist in training data, return global mean
        if anime_id not in self.train_matrix.columns:
            return self.global_mean_rating if self.global_mean_rating else 7.0

        # Get all ratings for this anime
        anime_ratings = self.train_matrix[anime_id].dropna()
        
        if anime_ratings.empty:
            return self.global_mean_rating if self.global_mean_rating else 7.0
            
        # If user exists in training data, use user-based approach
        if user_id in self.train_matrix.index:
            # Simple approach: return mean of all users who rated this anime
            return anime_ratings.mean()
        else:
            # For new users, return the mean rating of the anime
            return anime_ratings.mean()

    def get_user_recommendations(self, user_id: int, n_recommendations: int = 10) -> List[Tuple[int, str, float]]:
        """
        Generate anime recommendations for a user.
        
        Args:
            user_id (int): The ID of the user
            n_recommendations (int): Number of recommendations to return
            
        Returns:
            List[Tuple[int, str, float]]: List of (anime_id, anime_name, predicted_rating)
        """
        if self.train_matrix is None or self.anime_df is None:
            return []
            
        recommendations = []
        
        # Get all anime IDs from the training matrix
        all_anime_ids = self.train_matrix.columns.tolist()
        
        # If user exists in training data, exclude anime they've already rated
        if user_id in self.train_matrix.index:
            user_ratings = self.train_matrix.loc[user_id]
            unrated_anime_ids = user_ratings[user_ratings.isna()].index.tolist()
        else:
            # For new users, consider all anime
            unrated_anime_ids = all_anime_ids
            
        # Generate predictions for unrated anime
        for anime_id in unrated_anime_ids:
            predicted_rating = self.predict_rating(user_id, anime_id)
            
            # Get anime name from anime dataframe
            anime_info = self.anime_df[self.anime_df['anime_id'] == anime_id]
            if not anime_info.empty:
                anime_name = anime_info.iloc[0]['name']
                recommendations.append((anime_id, anime_name, predicted_rating))
        
        # Sort by predicted rating (descending) and return top N
        recommendations.sort(key=lambda x: x[2], reverse=True)
        return recommendations[:n_recommendations]
    
    def get_popular_recommendations(self, n_recommendations: int = 10) -> List[Tuple[int, str, float]]:
        """
        Get popular anime recommendations based on average ratings.
        
        Args:
            n_recommendations (int): Number of recommendations to return
            
        Returns:
            List[Tuple[int, str, float]]: List of (anime_id, anime_name, avg_rating)
        """
        if self.train_matrix is None or self.anime_df is None:
            return []
            
        # Calculate average rating for each anime
        anime_avg_ratings = self.train_matrix.mean(axis=0).dropna()
        
        # Filter anime with at least 5 ratings for reliability
        anime_rating_counts = self.train_matrix.count(axis=0)
        reliable_anime = anime_rating_counts[anime_rating_counts >= 5].index
        
        popular_anime = []
        for anime_id in reliable_anime:
            if anime_id in anime_avg_ratings.index:
                avg_rating = anime_avg_ratings[anime_id]
                
                # Get anime name
                anime_info = self.anime_df[self.anime_df['anime_id'] == anime_id]
                if not anime_info.empty:
                    anime_name = anime_info.iloc[0]['name']
                    popular_anime.append((anime_id, anime_name, avg_rating))
        
        # Sort by average rating and return top N
        popular_anime.sort(key=lambda x: x[2], reverse=True)
        return popular_anime[:n_recommendations]
    
    def get_anime_by_genre(self, genre: str, n_recommendations: int = 10) -> List[Tuple[int, str, float]]:
        """
        Get anime recommendations by genre.
        
        Args:
            genre (str): The genre to filter by
            n_recommendations (int): Number of recommendations to return
            
        Returns:
            List[Tuple[int, str, float]]: List of (anime_id, anime_name, avg_rating)
        """
        if self.anime_df is None:
            return []
            
        # Filter anime by genre (assuming genre column contains comma-separated values)
        genre_anime = self.anime_df[
            self.anime_df['genre'].str.contains(genre, case=False, na=False)
        ]
        
        recommendations = []
        for _, anime_row in genre_anime.iterrows():
            anime_id = anime_row['anime_id']
            anime_name = anime_row['name']
            
            # Get average rating from training matrix if available
            if self.train_matrix is not None and anime_id in self.train_matrix.columns:
                avg_rating = self.train_matrix[anime_id].mean()
                if not pd.isna(avg_rating):
                    recommendations.append((anime_id, anime_name, avg_rating))
            else:
                # Use the rating from anime dataframe as fallback
                rating = anime_row.get('rating', 7.0)
                recommendations.append((anime_id, anime_name, rating))
        
        # Sort by rating and return top N
        recommendations.sort(key=lambda x: x[2], reverse=True)
        return recommendations[:n_recommendations]

    def save_model(self, file_path: str):
        """
        Saves the trained model to a file using joblib.
        
        Args:
            file_path (str): The path to save the model file
        """
        if self.train_matrix is None:
            print("Warning: No model to save. Train the model first.")
            return

        model_data = {
            'train_matrix': self.train_matrix,
            'anime_df': self.anime_df,
            'global_mean_rating': self.global_mean_rating
        }
        
        print(f"Saving model to {file_path}...")
        joblib.dump(model_data, file_path)
        print("Model saved successfully.")

    @staticmethod
    def load_model(file_path: str):
        """
        Loads a trained model from a file.
        
        Args:
            file_path (str): The path to the saved model file
            
        Returns:
            UserBasedCollaborativeFilter: An instance of the loaded model
        """
        print(f"Loading model from {file_path}...")
        try:
            model_data = joblib.load(file_path)
            
            model = UserBasedCollaborativeFilter()
            model.train_matrix = model_data['train_matrix']
            model.anime_df = model_data['anime_df']
            model.global_mean_rating = model_data.get('global_mean_rating', 7.0)
            
            print("Model loaded successfully.")
            return model
        except FileNotFoundError:
            print(f"Error: Model file not found at {file_path}")
            return None
        except Exception as e:
            print(f"Error loading model: {e}")
            return None


# Updated training script
def train_and_save_model(train_df: pd.DataFrame, anime_df: pd.DataFrame, model_path: str = 'anime_recommender_model.pkl'):
    """
    Train and save the anime recommendation model.
    
    Args:
        train_df (pd.DataFrame): Training data with user_id, anime_id, rating
        anime_df (pd.DataFrame): Anime information data
        model_path (str): Path to save the trained model
    """
    print("Starting model training...")
    
    # Initialize and train the model
    model = UserBasedCollaborativeFilter()
    model.train(train_df, anime_df)
    
    # Save the model
    model.save_model(model_path)
    
    print("Model training and saving completed!")
    return model


if __name__ == '__main__':
    print("--- Enhanced Anime Recommendation System ---")
    
    # Define data directory and paths
    DATA_DIR = './data'
    ANIME_DATA_PATH = os.path.join(DATA_DIR, 'anime.csv')
    RATING_DATA_PATH = os.path.join(DATA_DIR, 'rating.csv')

    # Ensure the data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory '{DATA_DIR}' not found. Please create it and place your data files inside.")
        sys.exit(1)

    # Use the clean_and_split_data function to get the processed dataframes
    print("\n--- Starting Data Loading and Cleaning ---")
    train_df, test_df, anime_df = clean_and_split_data(
        anime_path=ANIME_DATA_PATH,
        rating_path=RATING_DATA_PATH,
        min_ratings_per_user=5,  # Adjust these thresholds as needed for your actual data
        min_ratings_per_anime=10
    )

    # Check if data loading and cleaning were successful
    if train_df is None or anime_df is None:
        print("Script terminated because data loading or cleaning failed.")
        sys.exit(1)
    
    print("\n--- Data Loading and Cleaning Successful ---")

    # Train the model using the loaded and cleaned data
    model = train_and_save_model(train_df, anime_df)
    
    # Test recommendations
    print("\n--- Testing Recommendations ---")
    
    # Example: Get recommendations for a sample user from your training data
    # You might need to find an existing user_id from your actual train_df
    if not train_df.empty:
        sample_user_id = train_df['user_id'].iloc[0] # Take the first user from the training data
        print(f"\nGetting recommendations for User ID: {sample_user_id}")
        user_recs = model.get_user_recommendations(user_id=sample_user_id, n_recommendations=5)
        print(f"\nRecommendations for User {sample_user_id}:")
        if user_recs:
            for anime_id, anime_name, rating in user_recs:
                print(f"  {anime_name} (ID: {anime_id}) - Predicted Rating: {rating:.2f}")
        else:
            print(f"  No recommendations found for User {sample_user_id}. They might have rated all popular anime, or there's insufficient unrated anime.")

    # Get popular recommendations
    popular_recs = model.get_popular_recommendations(n_recommendations=5)
    print(f"\nPopular Anime:")
    if popular_recs:
        for anime_id, anime_name, rating in popular_recs:
            print(f"  {anime_name} (ID: {anime_id}) - Avg Rating: {rating:.2f}")
    else:
        print("  No popular recommendations found.")
    
    # Get recommendations by genre (e.g., 'Action')
    action_recs = model.get_anime_by_genre('Action', n_recommendations=5)
    print(f"\nAction Anime:")
    if action_recs:
        for anime_id, anime_name, rating in action_recs:
            print(f"  {anime_name} (ID: {anime_id}) - Rating: {rating:.2f}")
    else:
        print("  No Action anime recommendations found.")