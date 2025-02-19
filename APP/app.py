import numpy as np
import pandas as pd
import streamlit as st
import requests
import base64
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import warnings

# Hide warnings
warnings.filterwarnings('ignore')

# Function to download a file from Google Drive
def download_file_from_google_drive(file_id, destination):
    URL = f"https://drive.google.com/uc?export=download&id={file_id}"
    session = requests.Session()
    response = session.get(URL, stream=True)
    
    with open(destination, "wb") as f:
        for chunk in response.iter_content(1024):
            if chunk:
                f.write(chunk)

# Download the dataset from Google Drive
download_file_from_google_drive('1rE1-_6LvV9Hyp_jaNok8XwqiBQxLzj4y', 'anime.csv')

# Download the background image from Google Drive
download_file_from_google_drive('1ZvEB01G9fTEs7Mf_Q6K4fJAMLAO-ZOtG', 'background.jpg')

# Load the dataset
df = pd.read_csv('anime.csv')

# Function to set background image
def set_background_image(image_path):
    # Convert image to base64
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    img_b64 = base64.b64encode(img_bytes).decode()

    # CSS for background image
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url('data:image/jpeg;base64,{img_b64}');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """, unsafe_allow_html=True
    )

# Function to get anime recommendations
def get_anime_recommendations(selected_anime, anime_count, df):
    # Create binary genre columns
    genre_columns = df['Genre'].str.get_dummies(sep=', ').columns
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
    
    # Initialize list to store recommendations
    all_recommendations = []
    
    # Get the indices of selected anime
    selected_indices = df[df['Title'].isin(selected_anime)].index
    
    # Get recommendations for each selected anime
    for index in selected_indices:
        # Get the similarity values for the selected anime title
        sim_values = anime_sim_matrix[index].argsort()[::-1][1:]
        
        # Get the top anime titles with the highest similarity values
        top_anime_titles = df.iloc[sim_values]['Title'].tolist()[:anime_count]
        
        all_recommendations.extend(top_anime_titles)
    
    # Remove duplicates and limit the number of recommendations
    all_recommendations = list(set(all_recommendations))[:anime_count]
    
    return all_recommendations

# Streamlit app
def main():
    # Set background image
    set_background_image('background.jpg')

    # Sidebar Design
    st.sidebar.subheader('Explore Anime Recommendations with Ease!')
    app_mode = st.sidebar.radio("Navigation", ["Anime Recommender", "About", "Contact"], index=0)
    
    # Main Content
    if app_mode == "Anime Recommender":
        st.title('Anime Recommendation Engine 🎥')
        
        st.markdown("""Select your favorite anime and get personalized recommendations based on genres, ratings, and more!""")
        
        selected_anime = st.multiselect('Pick Your Favorite Anime 🎬', df['Title'], default=[df['Title'].iloc[0]])
        anime_count = st.slider('How many recommendations do you want?', min_value=1, max_value=10, value=5)
        
        with st.spinner('Generating recommendations...'):
            if st.button('Get Recommendations 🌟'):
                anime_recommendations = get_anime_recommendations(selected_anime, anime_count, df)
                st.subheader('✨ Top Anime Recommendations')
                for i, anime in enumerate(anime_recommendations):
                    st.markdown(f"{i+1}. **{anime}**")

    elif app_mode == "About":
        st.title("About This App 👨‍💻")
        st.markdown("""Welcome to the **Anime Sphere**! This app uses a **cosine similarity algorithm** to recommend anime based on genres, ratings, and user interaction.""")
        st.write("Made with 💙 by Sofia.")

    elif app_mode == "Contact":
        st.title("Get in Touch 📬")
        st.markdown("""If you have any questions or feedback, feel free to reach out to me via email/linkedIn: - **Email**: [sofiaabielmi@gmail.com](mailto:sofiaabielmi@gmail.com)""")

# Run the app
if __name__ == '__main__':
    main()
