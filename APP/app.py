import numpy as np
import pandas as pd
import streamlit as st
import requests
import base64
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings('ignore')

# --- Cache the downloaded files ---
@st.cache_resource
def download_and_load_file(file_id, filename):
    URL = f"https://drive.google.com/uc?export=download&id={file_id}"
    session = requests.Session()
    response = session.get(URL, stream=True)

    with open(filename, "wb") as f:
        for chunk in response.iter_content(1024):
            if chunk:
                f.write(chunk)
    return filename

# --- Improved CSS for background and text readability ---
@st.cache_data
def get_background_style(image_path):
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    img_b64 = base64.b64encode(img_bytes).decode()
    return f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

        .stApp {{
            background-image: linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.6)), url("data:image/jpeg;base64,{img_b64}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-position: center;
            font-family: 'Roboto', sans-serif;
        }}

        h1, h2, h3, h4, h5, h6,
        .stMarkdown, .stText, .css-18e3th9, .css-1v0mbdj {{
            color: #FFAD5B !important;
            font-weight: bold !important;
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.8);
        }}

        p, span {{
            color: #20EFEC !important;
            font-weight: bold !important;
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.8);
        }}

        .stButton button {{
            color: #ffa500 !important;
            font-weight: bold;
            text-shadow: none;
            background-color: rgba(0,0,0,0.4);
            border: 1px solid #ffa500;
        }}
        </style>
    """

# --- Cache dataset and preprocessed values ---
@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv('anime.csv')
    genre_columns = df['Genre'].str.get_dummies(sep=', ').columns
    genre_df = df['Genre'].str.get_dummies(sep=', ')[genre_columns]
    df = pd.concat([df, genre_df], axis=1)
    df.fillna(0, inplace=True)

    scaler = StandardScaler()
    df[['Score', 'Members']] = scaler.fit_transform(df[['Score', 'Members']])
    similarity_cols = ['Score', 'Members'] + list(genre_columns)
    anime_sim_matrix = cosine_similarity(df[similarity_cols])

    return df, anime_sim_matrix

# --- Load files only once ---
download_and_load_file('1rE1-_6LvV9Hyp_jaNok8XwqiBQxLzj4y', 'anime.csv')
download_and_load_file('1ZvEB01G9fTEs7Mf_Q6K4fJAMLAO-ZOtG', 'background.jpg')

# --- Main function ---
def main():
    st.markdown(get_background_style('background.jpg'), unsafe_allow_html=True)

    # --- Custom Top-Right Navigation ---
    nav_options = ["🎥 Anime Recommender", "ℹ️ About", "📬 Contact"]
    if "app_mode" not in st.session_state:
        st.session_state.app_mode = nav_options[0]

    col1, col2, col3, col4 = st.columns([4, 3, 2, 2])
    with col2:
        if st.button("🎥 Recommender"):
            st.session_state.app_mode = "🎥 Anime Recommender"
    with col3:
        if st.button("ℹ️ About"):
            st.session_state.app_mode = "ℹ️ About"
    with col4:
        if st.button("📬 Contact"):
            st.session_state.app_mode = "📬 Contact"

    if st.session_state.app_mode not in nav_options:
        st.session_state.app_mode = nav_options[0]

    # --- Main App Logic ---
    if st.session_state.app_mode == "🎥 Anime Recommender":
        st.title('Anime Recommender Engine 🎥')
        st.markdown("Select your favorite anime and get personalized recommendations.")

        df, anime_sim_matrix = load_and_prepare_data()
        selected_anime = st.multiselect('Pick Your Favorite Anime 🎬', df['Title'], default=[df['Title'].iloc[0]])
        anime_count = st.slider('How many recommendations do you want?', 1, 10, 5)

        if st.button('Get Recommendations 🌟'):
            with st.spinner('Generating recommendations...'):
                indices = df[df['Title'].isin(selected_anime)].index
                all_recommendations = []

                for index in indices:
                    sim_scores = anime_sim_matrix[index].argsort()[::-1][1:]
                    recommended = df.iloc[sim_scores]['Title'].tolist()
                    all_recommendations.extend(recommended)

                unique_recommendations = list(set(all_recommendations))[:anime_count]
                st.subheader('Top Anime Recommendations')
                for i, anime in enumerate(unique_recommendations):
                    st.markdown(f"{i+1}. **{anime}**")

    elif st.session_state.app_mode == "ℹ️ About":
        st.title("About This App 👩‍💻")
        st.markdown("Welcome to **Anime Sphere**! A simple, beginner friendly recommendation system using cosine similarity.")

    elif st.session_state.app_mode == "📬 Contact":
        st.title("Get in Touch 📬")
        st.markdown("📩 Email: [sofiaabielmi@gmail.com](mailto:sofiaabielmi@gmail.com)")
        st.markdown("💻 GitHub: [github.com/sofiaabielmi](https://github.com/Sofiaabi)")

if __name__ == '__main__':
    main()
