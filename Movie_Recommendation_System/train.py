import os
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Create models folder
os.makedirs("models", exist_ok=True)

# Load datasets
movies = pd.read_csv("data/movies.csv")
tags = pd.read_csv("data/tags.csv")

# Combine tags for each movie
tags = tags.groupby("movieId")["tag"].apply(
    lambda x: " ".join(x.astype(str))
).reset_index()

# Merge with movies
movies = movies.merge(tags, on="movieId", how="left")

# Fill missing tags
movies["tag"] = movies["tag"].fillna("")

# Clean genres
movies["genres"] = movies["genres"].str.replace("|", " ", regex=False)

# Create combined text
movies["content"] = movies["genres"] + " " + movies["tag"]

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words="english")

tfidf_matrix = tfidf.fit_transform(movies["content"])

# Train Nearest Neighbors model
knn = NearestNeighbors(
    metric="cosine",
    algorithm="brute",
    n_neighbors=11
)

knn.fit(tfidf_matrix)

# Save everything
with open("models/movies.pkl", "wb") as f:
    pickle.dump(movies, f)

with open("models/tfidf.pkl", "wb") as f:
    pickle.dump(tfidf, f)

with open("models/knn_model.pkl", "wb") as f:
    pickle.dump(knn, f)

print("Training Complete!")
print(f"Movies: {len(movies)}")