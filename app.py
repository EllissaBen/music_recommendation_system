# app.py
import streamlit as st
from recommend import df, recommend_songs

# ---------------------------
# Streamlit page config
# ---------------------------
st.set_page_config(
    page_title="Music Recommender 🎵",
    page_icon="🎧",
    layout="centered"
)

st.title("🎶 Instant Music Recommender")

# Dropdown to select a song
song_list = sorted(df['song'].dropna().unique())
selected_song = st.selectbox("🎵 Select a song:", song_list)

# Recommend button
if st.button("🚀 Recommend Similar Songs"):
    with st.spinner("Finding similar songs..."):
        recommendations = recommend_songs(selected_song)
        if recommendations is None:
            st.warning("❌ Sorry, song not found.")
        else:
            st.success("Top similar songs:")
            st.table(recommendations)
