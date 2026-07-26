from flask import Flask, render_template, request
from recommendation import recommend

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/recommend", methods=["POST"])
def get_recommendation():

    movie = request.form["movie"]

    recommendations = recommend(movie)

    return render_template(
        "result.html",
        movie=movie,
        recommendations=recommendations
    )


if __name__ == "__main__":
    app.run(debug=True)