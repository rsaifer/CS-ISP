# Importing pandas and numpy for reading csv and array manupilation
import pandas as pd
import numpy as np

# Importing sys for argument parsing
import sys

# Importing sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Reads tmdb_5000_credits and tmdb_5000_movies
# Also defines column names
# Returns a merged dataframe
def get_merged_dataframe():

    # Reading files
    df1 = pd.read_csv("./tmdb_5000_credits.csv")
    df2 = pd.read_csv("./tmdb_5000_movies.csv")

    # Defining column names for our credits data frame
    df1.columns = ["id", "tittle", "cast", "crew"]

    # merging both data frames together
    df2 = df2.merge(df1, on="id")
    return df2

# Retunrs a TF-IDF matrix
def get_tfidf_matrix(merged_df):
    # Define a TF-IDF Vectorizer Object.
    # Remove all english stop words such as 'the', 'a' so the data is more "clean"
    tfidf = TfidfVectorizer(stop_words="english")

    # Replace NaN with an empty string
    merged_df["overview"] = merged_df["overview"].fillna("")

    # Construct the required TF-IDF matrix by fitting and transforming the data
    tfidf_matrix = tfidf.fit_transform(merged_df["overview"])
    return tfidf_matrix


# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title ,merged_df,cosine_sim,indices):
    # Get the index of the movie that matches the title
    try:
        idx = indices[title]

        # Get the pairwsie similarity scores of all movies with that movie
        sim_scores = list(enumerate(cosine_sim[idx]))

        # Sort the movies based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the scores of the 10 most similar movies
        sim_scores = sim_scores[1:11]

        # Get the movie indices
        movie_indices = [i[0] for i in sim_scores]

        # Return the top 10 most similar movies
        return merged_df["title"].iloc[movie_indices]

    except KeyError:
        print("Couldn't find the movie in the db :(")
        return  pd.DataFrame({'A' : []})


def run(movie):
    merged_df = get_merged_dataframe()
    
    tfidf_matrix = get_tfidf_matrix(merged_df)

    # Compute the cosine similarity matrix
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Construct a reverse map of indices and movie titles
    indices = pd.Series(merged_df.index, index=merged_df["title"]).drop_duplicates()

    return get_recommendations(movie,merged_df,cosine_sim,indices)

if __name__ == "__main__":
    run()
