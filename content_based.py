import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_data(path="data/anime.csv"):
    df = pd.read_csv(path)
    df = df[['name', 'genre', 'rating']]
    df = df.dropna(subset=['name'])
    df['genre'] = df['genre'].fillna('')
    return df

def compute_similarity(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['genre'])
    sim_matrix = cosine_similarity(tfidf_matrix)
    return sim_matrix

def recommend(title, df, sim_matrix, top_n=5):
    idx_map = pd.Series(df.index, index=df['name']).drop_duplicates()
    if title not in idx_map:
        return f"'{title}' not found."

    idx = idx_map[title]
    scores = list(enumerate(sim_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    rec_idx = [i[0] for i in scores[1:top_n+1]]

    return df[['name', 'rating']].iloc[rec_idx].reset_index(drop=True)

# Example usage
if __name__ == "__main__":
    df = load_data()
    sim = compute_similarity(df)
    print(recommend("Naruto", df, sim))