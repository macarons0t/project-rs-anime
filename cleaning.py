# 1-cleaning.py
"""
This script loads anime and rating data, cleans both datasets, and splits the rating data into
training and testing sets. It includes improved filtering for better recommendations.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import numpy as np
import sys

def clean_and_split_data(anime_path='./data/anime.csv', rating_path='./data/rating.csv', 
                        test_size: float = 0.2, random_state: int = 42, min_ratings_per_user: int = 5,
                        min_ratings_per_anime: int = 10):
    """
    Loads anime and rating data, cleans both datasets, and splits rating data into
    training and testing sets with improved filtering for better recommendations.

    Args:
        anime_path (str): Path to the anime.csv file
        rating_path (str): Path to the rating.csv file
        test_size (float): The proportion of the dataset to include in the test split
        random_state (int): Controls the shuffling applied to the data before applying the split
        min_ratings_per_user (int): Minimum number of ratings per user to keep
        min_ratings_per_anime (int): Minimum number of ratings per anime to keep

    Returns:
        tuple: A tuple containing:
            - train_df (pd.DataFrame): The training DataFrame
            - test_df (pd.DataFrame): The testing DataFrame
            - anime (pd.DataFrame): The cleaned anime DataFrame
            - None, None, None if file not found or data is empty after filtering
    """
    print(f"Loading anime data from: {anime_path}")
    try:
        anime = pd.read_csv(anime_path, delimiter=',')
        print(f"Loaded anime data with shape: {anime.shape}")
    except FileNotFoundError:
        print(f"Error: Anime data file not found at '{anime_path}'. Please ensure it's in the correct directory or adjust the path.")
        return None, None, None

    print(f"Loading rating data from: {rating_path}")
    try:
        rating = pd.read_csv(rating_path, delimiter=',')
        print(f"Loaded rating data with shape: {rating.shape}")
    except FileNotFoundError:
        print(f"Error: Rating data file not found at '{rating_path}'. Please ensure it's in the correct directory or adjust the path.")
        return None, None, None

    # Clean anime data
    print("\n=== Cleaning Anime Data ===")
    print("Original anime data shape:", anime.shape)
    
    # Remove anime with missing names
    anime_clean = anime.dropna(subset=['name']).copy()
    print(f"After removing anime with missing names: {anime_clean.shape}")
    
    # Handle missing genres - fill with 'Unknown'
    anime_clean['genre'] = anime_clean['genre'].fillna('Unknown')
    
    # Handle missing ratings - fill with median rating
    if 'rating' in anime_clean.columns:
        median_rating = anime_clean['rating'].median()
        anime_clean['rating'] = anime_clean['rating'].fillna(median_rating)
        print(f"Filled missing anime ratings with median: {median_rating}")
    
    # Handle missing member counts
    if 'members' in anime_clean.columns:
        anime_clean['members'] = anime_clean['members'].fillna(0)
    
    # Clean rating data
    print("\n=== Cleaning Rating Data ===")
    print("Original rating data shape:", rating.shape)
    
    # Remove ratings with value -1 (unknown ratings)
    rating_clean = rating[rating['rating'] != -1].copy()
    print(f"After removing -1 ratings: {rating_clean.shape}")
    
    # Remove duplicate (user_id, anime_id) combinations by keeping the mean rating
    print("Removing duplicate (user_id, anime_id) combinations...")
    rating_clean = rating_clean.groupby(['user_id', 'anime_id']).agg({'rating': 'mean'}).reset_index()
    print(f"After handling duplicates: {rating_clean.shape}")
    
    # Filter users and anime with sufficient ratings for better recommendations
    print(f"\n=== Filtering for Quality ===")
    
    # Count ratings per user and per anime
    user_rating_counts = rating_clean['user_id'].value_counts()
    anime_rating_counts = rating_clean['anime_id'].value_counts()
    
    print(f"Users with at least {min_ratings_per_user} ratings: {sum(user_rating_counts >= min_ratings_per_user)}")
    print(f"Anime with at least {min_ratings_per_anime} ratings: {sum(anime_rating_counts >= min_ratings_per_anime)}")
    
    # Keep only users with sufficient ratings
    active_users = user_rating_counts[user_rating_counts >= min_ratings_per_user].index
    rating_filtered = rating_clean[rating_clean['user_id'].isin(active_users)].copy()
    print(f"After filtering users: {rating_filtered.shape}")
    
    # Keep only anime with sufficient ratings
    popular_anime = anime_rating_counts[anime_rating_counts >= min_ratings_per_anime].index
    rating_filtered = rating_filtered[rating_filtered['anime_id'].isin(popular_anime)].copy()
    print(f"After filtering anime: {rating_filtered.shape}")
    
    # Ensure we still have anime data for the filtered anime IDs
    anime_clean = anime_clean[anime_clean['anime_id'].isin(rating_filtered['anime_id'].unique())].copy()
    print(f"Final anime data shape: {anime_clean.shape}")
    
    # Split into train and test
    print(f"\n=== Splitting Data ===")
    print(f"Splitting data into training ({1-test_size:.0%}) and testing ({test_size:.0%}) sets...")
    
    if len(rating_filtered) == 0:
        print("Error: No data remaining after filtering. Consider lowering the minimum rating thresholds.")
        return None, None, None
    
    train_df, test_df = train_test_split(rating_filtered, test_size=test_size, 
                                        random_state=random_state, stratify=None)
    
    print(f"Train data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    
    # Print some statistics
    print(f"\n=== Data Statistics ===")
    print(f"Unique users in training: {train_df['user_id'].nunique()}")
    print(f"Unique anime in training: {train_df['anime_id'].nunique()}")
    print(f"Rating range: {train_df['rating'].min():.1f} - {train_df['rating'].max():.1f}")
    print(f"Average rating: {train_df['rating'].mean():.2f}")
    print(f"Available anime names: {anime_clean['name'].nunique()}")

    return train_df, test_df, anime_clean

if __name__ == '__main__':
    # Define data directory and paths
    DATA_DIR = './data'
    ANIME_DATA_PATH = os.path.join(DATA_DIR, 'anime.csv')
    RATING_DATA_PATH = os.path.join(DATA_DIR, 'rating.csv')

    # Check if the data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory '{DATA_DIR}' not found. Please create it and place your data files inside.")
        sys.exit(1)

    # Call the clean_and_split_data function, which now handles file loading and errors
    train_df_example, test_df_example, anime_df_example = clean_and_split_data(
        anime_path=ANIME_DATA_PATH,
        rating_path=RATING_DATA_PATH,
        min_ratings_per_user=5,  # Set your desired thresholds for real data
        min_ratings_per_anime=10
    )
    
    # Check if the function returned None, indicating a failure (e.g., file not found or no data after filtering)
    if train_df_example is None or test_df_example is None or anime_df_example is None:
        print("Script terminated due to data loading or cleaning issues. Check the error messages above.")
        sys.exit(1)
    else:
        print("\n=== Data Cleaning and Splitting Successful! ===")
        print("\nCleaned training data sample:")
        print(train_df_example.head())
        print("\nCleaned testing data sample:")
        print(test_df_example.head())
        print("\nCleaned anime data sample:")
        print(anime_df_example.head())