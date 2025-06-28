# project-rs-anime
A collaborative Streamlit-based anime recommender system project for course demo.

## Business Use Case

This project addresses the challenge of content discovery in the vast and ever-growing world of anime. For both casual viewers and dedicated fans, navigating the immense catalogue of anime series and movies can be overwhelming. Our recommender system solves this by providing personalized suggestions, helping users efficiently discover new anime titles aligned with their interests.

By curating recommendations, we aim to:
- Enhance user satisfaction
- Reduce decision fatigue
- Foster a deeper, more enjoyable engagement with the anime medium

### MVP Description

The Minimum Viable Product (MVP) is an interactive Streamlit web application. It allows users to input a user ID and view personalized anime recommendations.

The MVP showcases:
- Collaborative filtering (user-based or item-based)
- Content-based filtering (based on genres, type, etc.)
- Neural collaborative filtering (hybrid)

The app visually compares different strategies, helping users understand their behaviour and performance.

## Project Folder Structure
```
project-rs-anime/
├── app/            # Streamlit files (.py)
├── data/           # All CSV datasets
├── notebooks/      # All EDA and model notebooks (.ipynb)
├── README.md       # Required project overview
└── requirements.txt # Required to show dependencies
