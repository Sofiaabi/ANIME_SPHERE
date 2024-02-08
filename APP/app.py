# several helpful packages to load
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import streamlit as st # Web app

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')



# Loading the dataset
df = pd.read_csv('C:/Users/Sofia/Documents/ANIME_SPHERE/DATA/anime.csv')


# observing the first few observations
print(df.head((10)))



def get_anime_recommendations(title, df):
    # Create binary genre columns
    genre_columns = df['Genre'].str.get_dummies(sep=', ').columns
    
    # Create DataFrame with binary genre columns
    genre_df = df['Genre'].str.get_dummies(sep=', ')[genre_columns]
    
    # Merge the genre dataframe with the original dataframe
    df = pd.concat([df, genre_df], axis=1)
    
    # Fill NaN values with zeros
    df.fillna(0, inplace=True)
    
    # Scale the rating and members columns
    scaler = StandardScaler()
    df[['Score', 'Members']] = scaler.fit_transform(df[['Score', 'Members']])
    
    # Select the columns to use for similarity
    similarity_cols = ['Score', 'Members'] + list(genre_columns)
    
    # Compute the cosine similarity matrix
    anime_data = df[similarity_cols]
    anime_sim_matrix = cosine_similarity(anime_data)
    
    # Get the index of the anime title
    title_index = df[df['Title'] == title].index[0]
    
    # Get the similarity values for the given anime title
    sim_values = anime_sim_matrix[title_index].argsort()[::-1][1:]
    
    # Get the top 10 anime titles with the highest similarity values
    top_anime_titles = df.iloc[sim_values]['Title'].tolist()[:10]
    
    # Create a numbered list of recommendations
    recommendations = "\n".join([f"{i+1}. {anime_title}" for i, anime_title in enumerate(top_anime_titles)])
    
    return recommendations

# Get anime recommendations for "Koe no Katachi"
anime_recommendations = get_anime_recommendations("Koe no Katachi", df)

# Print the recommendations
print(anime_recommendations)
