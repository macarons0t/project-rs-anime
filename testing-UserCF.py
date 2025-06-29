# 3-testing-CollaborativeFiltering.py
import pandas as pd
import numpy as np
import os
import sys
from sklearn.metrics import mean_squared_error
from recommender_model import UserBasedCollaborativeFilter
from cleaning import clean_and_split_data
from tqdm import tqdm # Import tqdm

def evaluate_model(model: UserBasedCollaborativeFilter, test_df: pd.DataFrame):
    """
    Evaluates the trained recommendation model on the test dataset.

    Args:
        model (UserBasedCollaborativeFilter): The trained recommendation model.
        test_df (pd.DataFrame): The test DataFrame containing user_id, anime_id, and actual ratings.
    """
    print("\n--- Starting Model Evaluation ---")
    predictions = []
    actual_ratings = []
    
    # Sample 10% of the test_df for faster evaluation
    sampled_test_df = test_df.sample(frac=0.1, random_state=42)
    print(f"Evaluating on 10% of test data: {len(sampled_test_df)} samples.")
    
    # Iterate over the sampled test set and make predictions with a progress bar
    # Use tqdm to wrap the iteration for the progress bar
    for index, row in tqdm(sampled_test_df.iterrows(), total=len(sampled_test_df), desc="Predicting ratings"):
        user_id = int(row['user_id'])
        anime_id = int(row['anime_id'])
        actual_rating = row['rating']
        
        predicted_rating = model.predict_rating(user_id, anime_id)
        
        predictions.append(predicted_rating)
        actual_ratings.append(actual_rating)

    # Calculate performance metrics
    if not predictions:
        print("No predictions were made. The test set might be empty or encounter issues during prediction.")
        return

    # Convert lists to numpy arrays for calculation
    predictions = np.array(predictions)
    actual_ratings = np.array(actual_ratings)

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(actual_ratings, predictions))
    print(f"Root Mean Squared Error (RMSE) on test set: {rmse:.4f}")

    # You can add other metrics here if desired, e.g., MAE
    mae = np.mean(np.abs(predictions - actual_ratings))
    print(f"Mean Absolute Error (MAE) on test set: {mae:.4f}")

    print("--- Model Evaluation Completed ---")


if __name__ == '__main__':
    print("--- Evaluating Anime Recommendation System Performance ---")

    # Define data directory and paths
    DATA_DIR = './data'
    ANIME_DATA_PATH = os.path.join(DATA_DIR, 'anime.csv')
    RATING_DATA_PATH = os.path.join(DATA_DIR, 'rating.csv')
    MODEL_PATH = 'anime_recommender_model.pkl'

    # Ensure the data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory '{DATA_DIR}' not found. Please create it and place your data files inside.")
        sys.exit(1)

    # Load and clean data to get the test set
    print("\n--- Loading and Cleaning Data for Evaluation ---")
    _, test_df, anime_df = clean_and_split_data(
        anime_path=ANIME_DATA_PATH,
        rating_path=RATING_DATA_PATH,
        min_ratings_per_user=5,
        min_ratings_per_anime=10
    )

    if test_df is None:
        print("Script terminated because data loading or cleaning failed, or no test data available.")
        sys.exit(1)

    # Load the trained model
    print(f"\n--- Loading Trained Model from {MODEL_PATH} ---")
    model = UserBasedCollaborativeFilter.load_model(MODEL_PATH)

    if model is None:
        print("Script terminated because the model could not be loaded.")
        sys.exit(1)

    # Evaluate the loaded model on the test data
    evaluate_model(model, test_df)