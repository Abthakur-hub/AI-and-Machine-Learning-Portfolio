from recommendation import recommend

movie = input("Enter movie name: ")

result = recommend(movie)

if result is None:
    print("Movie not found!")
else:
    print("\nRecommended Movies:\n")

    for i, movie in enumerate(result, start=1):
        print(f"{i}. {movie}")