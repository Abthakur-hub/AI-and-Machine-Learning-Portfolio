import pickle

movies = pickle.load(open("models/movies.pkl", "rb"))
tfidf = pickle.load(open("models/tfidf.pkl", "rb"))
knn = pickle.load(open("models/knn_model.pkl", "rb"))

tfidf_matrix = tfidf.transform(movies["content"])


def recommend(movie_name, top_n=10):
    movie = movies[movies["title"].str.lower() == movie_name.lower()]

    if movie.empty:
        return None

    idx = movie.index[0]

    distances, indices = knn.kneighbors(
        tfidf_matrix[idx],
        n_neighbors=top_n + 1
    )

    recommendations = []

    for i in indices.flatten()[1:]:
        recommendations.append(movies.iloc[i]["title"])

    return recommendations