import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# SPOTIFY MUSIC TRENDS DASHBOARD
# Dataset expected file name: train.csv
# Place this Python file and train.csv in the same folder.
# Run using: streamlit run spotify_dashboard_app.py
# ============================================================

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Spotify Music Trends Dashboard",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Custom CSS Styling
# Primary colors: green, black, and white
# -----------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #050505 0%, #101010 50%, #062b16 100%);
        color: #FFFFFF;
    }

    [data-testid="stSidebar"] {
        background-color: #050505;
        border-right: 1px solid #1DB954;
    }

    [data-testid="stSidebar"] * {
        color: #FFFFFF;
    }

    .main-title {
        font-size: 56px;
        font-weight: 900;
        color: #1DB954;
        margin-bottom: 0px;
        letter-spacing: -1px;
    }

    .sub-title {
        font-size: 18px;
        color: #D8D8D8;
        margin-top: 0px;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 34px;
        color: #1DB954;
        font-weight: 800;
        margin-top: 20px;
        margin-bottom: 12px;
    }

    .small-note {
        color: #CFCFCF;
        font-size: 14px;
    }

    .kpi-card {
        background-color: #0B0B0B;
        border: 1px solid #1DB954;
        border-radius: 18px;
        padding: 22px;
        box-shadow: 0 0 18px rgba(29, 185, 84, 0.20);
        min-height: 125px;
    }

    .kpi-label {
        color: #BDBDBD;
        font-size: 15px;
        font-weight: 600;
        margin-bottom: 4px;
    }

    .kpi-value {
        color: #FFFFFF;
        font-size: 31px;
        font-weight: 900;
    }

    .insight-box {
        background-color: #0B0B0B;
        border-left: 8px solid #1DB954;
        border-radius: 14px;
        padding: 20px 24px;
        color: #FFFFFF;
        font-size: 17px;
        line-height: 1.7;
        box-shadow: 0 0 16px rgba(29, 185, 84, 0.18);
    }

    .footer-box {
        background-color: #080808;
        border-top: 1px solid #1DB954;
        border-radius: 12px;
        padding: 16px 20px;
        color: #D6D6D6;
        font-size: 14px;
        margin-top: 28px;
    }

    div[data-testid="stMetric"] {
        background-color: #0B0B0B;
        border: 1px solid #1DB954;
        padding: 18px;
        border-radius: 18px;
        box-shadow: 0 0 15px rgba(29, 185, 84, 0.16);
    }

    div[data-testid="stMetricLabel"] {
        color: #BDBDBD;
    }

    div[data-testid="stMetricValue"] {
        color: #FFFFFF;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Load and Clean Data
# -----------------------------
@st.cache_data
def load_data(file_path="train.csv"):
    df = pd.read_csv(file_path)

    # Remove unnecessary index column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Standardize column names for safer use
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Remove rows missing important display fields
    df = df.dropna(subset=["track_name", "artists", "album_name", "track_genre"])

    # Remove exact duplicate rows only.
    # We do not drop duplicate track_id because some tracks may appear under different genres.
    df = df.drop_duplicates()

    # Convert duration from milliseconds to minutes
    df["duration_min"] = df["duration_ms"] / 60000

    # Create a cleaner explicit label
    df["explicit_label"] = np.where(df["explicit"] == True, "Explicit", "Clean")

    # Create popularity category for easier dashboard interpretation
    df["popularity_category"] = pd.cut(
        df["popularity"],
        bins=[-1, 30, 60, 100],
        labels=["Low Popularity", "Medium Popularity", "High Popularity"]
    )

    # Create mood category using valence
    df["mood_category"] = pd.cut(
        df["valence"],
        bins=[-0.01, 0.33, 0.66, 1.00],
        labels=["Low Valence", "Balanced Valence", "High Valence"]
    )

    # Clean text values
    df["track_genre"] = df["track_genre"].astype(str).str.title()
    df["artists"] = df["artists"].astype(str)
    df["track_name"] = df["track_name"].astype(str)
    df["album_name"] = df["album_name"].astype(str)

    return df

try:
    df = load_data("train.csv")
except FileNotFoundError:
    st.error("File not found. Please make sure train.csv is in the same folder as this Streamlit file.")
    st.stop()

# -----------------------------
# Header
# -----------------------------
st.markdown("<p class='main-title'>Spotify Music Trends Dashboard 🎧</p>", unsafe_allow_html=True)
st.markdown(
    "<p class='sub-title'>Exploring track popularity, genres, artists, and audio features using Spotify track data.</p>",
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class='small-note'>
    <b>Team Members:</b> Ralph Victor L. Ilarde, Member 2, Member 3<br>
    <b>Dashboard Theme:</b> Music analytics, listener trends, and audio feature exploration
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.markdown("# 🎵 Dashboard Filters")
st.sidebar.markdown("Use these filters to explore Spotify music trends.")

all_genres = sorted(df["track_genre"].dropna().unique())
selected_genres = st.sidebar.multiselect(
    "Select Genre(s)",
    options=all_genres,
    default=all_genres[:8]
)

popularity_range = st.sidebar.slider(
    "Popularity Range",
    int(df["popularity"].min()),
    int(df["popularity"].max()),
    (int(df["popularity"].min()), int(df["popularity"].max()))
)

energy_range = st.sidebar.slider(
    "Energy Range",
    float(df["energy"].min()),
    float(df["energy"].max()),
    (float(df["energy"].min()), float(df["energy"].max()))
)

danceability_range = st.sidebar.slider(
    "Danceability Range",
    float(df["danceability"].min()),
    float(df["danceability"].max()),
    (float(df["danceability"].min()), float(df["danceability"].max()))
)

explicit_choice = st.sidebar.multiselect(
    "Track Type",
    options=sorted(df["explicit_label"].unique()),
    default=sorted(df["explicit_label"].unique())
)

search_text = st.sidebar.text_input("Search Track or Artist")

# Apply filters
filtered_df = df[
    (df["track_genre"].isin(selected_genres)) &
    (df["popularity"].between(popularity_range[0], popularity_range[1])) &
    (df["energy"].between(energy_range[0], energy_range[1])) &
    (df["danceability"].between(danceability_range[0], danceability_range[1])) &
    (df["explicit_label"].isin(explicit_choice))
].copy()

if search_text:
    search_text_lower = search_text.lower()
    filtered_df = filtered_df[
        filtered_df["track_name"].str.lower().str.contains(search_text_lower, na=False) |
        filtered_df["artists"].str.lower().str.contains(search_text_lower, na=False)
    ]

if filtered_df.empty:
    st.warning("No records found based on the selected filters. Please adjust your filter settings.")
    st.stop()

# -----------------------------
# KPI Section
# -----------------------------
st.markdown("<p class='section-title'>Key Performance Indicators</p>", unsafe_allow_html=True)

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

total_tracks = len(filtered_df)
avg_popularity = filtered_df["popularity"].mean()
avg_danceability = filtered_df["danceability"].mean()
avg_energy = filtered_df["energy"].mean()
avg_tempo = filtered_df["tempo"].mean()

kpi1.metric("Total Tracks", f"{total_tracks:,}")
kpi2.metric("Avg Popularity", f"{avg_popularity:.2f}")
kpi3.metric("Avg Danceability", f"{avg_danceability:.2f}")
kpi4.metric("Avg Energy", f"{avg_energy:.2f}")
kpi5.metric("Avg Tempo", f"{avg_tempo:.0f} BPM")

st.text("")

# -----------------------------
# Top Track Highlight
# -----------------------------
top_track = filtered_df.sort_values("popularity", ascending=False).iloc[0]
st.markdown(
    f"""
    <div class='insight-box'>
    <b>Top Track in Current Selection:</b> {top_track['track_name']} by {top_track['artists']}<br>
    <b>Genre:</b> {top_track['track_genre']} &nbsp; | &nbsp;
    <b>Popularity:</b> {top_track['popularity']} &nbsp; | &nbsp;
    <b>Energy:</b> {top_track['energy']:.2f} &nbsp; | &nbsp;
    <b>Danceability:</b> {top_track['danceability']:.2f}
    </div>
    """,
    unsafe_allow_html=True
)

st.text("")

# -----------------------------
# Chart Theme Helper
# -----------------------------
def apply_spotify_layout(fig, title=None):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#050505",
        plot_bgcolor="#0B0B0B",
        font=dict(color="#FFFFFF"),
        title=dict(text=title if title else fig.layout.title.text, font=dict(size=22, color="#1DB954")),
        legend=dict(bgcolor="rgba(0,0,0,0)")
    )
    fig.update_xaxes(gridcolor="#2A2A2A")
    fig.update_yaxes(gridcolor="#2A2A2A")
    return fig

spotify_green_scale = ["#0B0B0B", "#0E7A35", "#1DB954", "#A7F3C1", "#FFFFFF"]

# -----------------------------
# Charts Row 1
# -----------------------------
st.markdown("<p class='section-title'>Genre and Popularity Overview</p>", unsafe_allow_html=True)

left_col, right_col = st.columns(2)

with left_col:
    genre_pop = (
        filtered_df.groupby("track_genre", as_index=False)
        .agg(avg_popularity=("popularity", "mean"), total_tracks=("track_id", "count"))
        .sort_values("avg_popularity", ascending=False)
        .head(15)
    )

    fig_genre = px.bar(
        genre_pop,
        x="avg_popularity",
        y="track_genre",
        orientation="h",
        color="avg_popularity",
        color_continuous_scale=spotify_green_scale,
        title="Top Genres by Average Popularity",
        labels={"avg_popularity": "Average Popularity", "track_genre": "Genre"},
        hover_data=["total_tracks"]
    )
    fig_genre.update_layout(yaxis={"categoryorder": "total ascending"})
    fig_genre = apply_spotify_layout(fig_genre)
    st.plotly_chart(fig_genre, use_container_width=True)

with right_col:
    fig_hist = px.histogram(
        filtered_df,
        x="popularity",
        nbins=30,
        color="popularity_category",
        color_discrete_sequence=["#0E7A35", "#1DB954", "#FFFFFF"],
        title="Distribution of Track Popularity",
        labels={"popularity": "Popularity Score", "popularity_category": "Popularity Category"}
    )
    fig_hist = apply_spotify_layout(fig_hist)
    st.plotly_chart(fig_hist, use_container_width=True)

# -----------------------------
# Charts Row 2
# -----------------------------
st.markdown("<p class='section-title'>Audio Features and Popularity</p>", unsafe_allow_html=True)

left_col2, right_col2 = st.columns(2)

with left_col2:
    sample_df = filtered_df.sample(min(5000, len(filtered_df)), random_state=42)
    fig_scatter = px.scatter(
        sample_df,
        x="danceability",
        y="popularity",
        color="track_genre",
        size="energy",
        hover_name="track_name",
        hover_data=["artists", "energy", "tempo", "valence"],
        title="Danceability vs Popularity",
        labels={"danceability": "Danceability", "popularity": "Popularity"}
    )
    fig_scatter = apply_spotify_layout(fig_scatter)
    st.plotly_chart(fig_scatter, use_container_width=True)

with right_col2:
    fig_bubble = px.scatter(
        sample_df,
        x="energy",
        y="valence",
        size="popularity",
        color="explicit_label",
        hover_name="track_name",
        hover_data=["artists", "track_genre", "danceability", "tempo"],
        color_discrete_sequence=["#1DB954", "#FFFFFF"],
        title="Energy, Mood, and Popularity Bubble View",
        labels={"energy": "Energy", "valence": "Valence / Mood", "explicit_label": "Track Type"}
    )
    fig_bubble = apply_spotify_layout(fig_bubble)
    st.plotly_chart(fig_bubble, use_container_width=True)

# -----------------------------
# Charts Row 3
# -----------------------------
st.markdown("<p class='section-title'>Top Artists and Track Rankings</p>", unsafe_allow_html=True)

left_col3, right_col3 = st.columns(2)

with left_col3:
    artist_rank = (
        filtered_df.groupby("artists", as_index=False)
        .agg(avg_popularity=("popularity", "mean"), track_count=("track_id", "count"))
    )

    # Keep artists with at least 2 tracks in the current filter to avoid one-track bias
    artist_rank = artist_rank[artist_rank["track_count"] >= 2]

    if artist_rank.empty:
        artist_rank = (
            filtered_df.groupby("artists", as_index=False)
            .agg(avg_popularity=("popularity", "mean"), track_count=("track_id", "count"))
        )

    artist_rank = artist_rank.sort_values("avg_popularity", ascending=False).head(15)

    fig_artist = px.bar(
        artist_rank,
        x="avg_popularity",
        y="artists",
        orientation="h",
        color="avg_popularity",
        color_continuous_scale=spotify_green_scale,
        title="Top Artists by Average Popularity",
        labels={"avg_popularity": "Average Popularity", "artists": "Artist"},
        hover_data=["track_count"]
    )
    fig_artist.update_layout(yaxis={"categoryorder": "total ascending"})
    fig_artist = apply_spotify_layout(fig_artist)
    st.plotly_chart(fig_artist, use_container_width=True)

with right_col3:
    top_tracks = filtered_df.sort_values("popularity", ascending=False).head(15)
    fig_tracks = px.bar(
        top_tracks,
        x="popularity",
        y="track_name",
        orientation="h",
        color="popularity",
        color_continuous_scale=spotify_green_scale,
        title="Top Tracks by Popularity",
        labels={"popularity": "Popularity", "track_name": "Track"},
        hover_data=["artists", "track_genre", "album_name"]
    )
    fig_tracks.update_layout(yaxis={"categoryorder": "total ascending"})
    fig_tracks = apply_spotify_layout(fig_tracks)
    st.plotly_chart(fig_tracks, use_container_width=True)

# -----------------------------
# Audio Feature Comparison
# -----------------------------
st.markdown("<p class='section-title'>Average Audio Feature Profile</p>", unsafe_allow_html=True)

feature_cols = ["danceability", "energy", "speechiness", "acousticness", "instrumentalness", "liveness", "valence"]
feature_avg = filtered_df[feature_cols].mean().reset_index()
feature_avg.columns = ["Audio Feature", "Average Value"]

fig_features = px.bar(
    feature_avg,
    x="Audio Feature",
    y="Average Value",
    color="Average Value",
    color_continuous_scale=spotify_green_scale,
    title="Average Audio Features of Selected Tracks",
    text="Average Value"
)
fig_features.update_traces(texttemplate="%{text:.2f}", textposition="outside")
fig_features.update_yaxes(range=[0, 1])
fig_features = apply_spotify_layout(fig_features)
st.plotly_chart(fig_features, use_container_width=True)

# -----------------------------
# Correlation Heatmap
# -----------------------------
st.markdown("<p class='section-title'>Relationship Between Numerical Features</p>", unsafe_allow_html=True)

corr_cols = [
    "popularity", "duration_min", "danceability", "energy", "loudness",
    "speechiness", "acousticness", "instrumentalness", "liveness",
    "valence", "tempo"
]
corr_matrix = filtered_df[corr_cols].corr(numeric_only=True)

fig_corr = go.Figure(
    data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale=[[0, "#FFFFFF"], [0.5, "#0B0B0B"], [1, "#1DB954"]],
        zmin=-1,
        zmax=1,
        colorbar=dict(title="Correlation")
    )
)
fig_corr.update_layout(
    title="Correlation Heatmap of Spotify Audio Features",
    template="plotly_dark",
    paper_bgcolor="#050505",
    plot_bgcolor="#0B0B0B",
    font=dict(color="#FFFFFF"),
    title_font=dict(size=22, color="#1DB954")
)
st.plotly_chart(fig_corr, use_container_width=True)

# -----------------------------
# Data Preview
# -----------------------------
st.markdown("<p class='section-title'>Cleaned Data Preview</p>", unsafe_allow_html=True)
preview_cols = [
    "track_name", "artists", "album_name", "track_genre", "popularity",
    "duration_min", "explicit_label", "danceability", "energy", "valence", "tempo"
]
st.dataframe(
    filtered_df[preview_cols].sort_values("popularity", ascending=False).head(50),
    use_container_width=True,
    hide_index=True
)

# -----------------------------
# Final Insight
# -----------------------------
st.markdown("<p class='section-title'>💡 Key Insight</p>", unsafe_allow_html=True)

avg_pop_high_energy = filtered_df[filtered_df["energy"] >= filtered_df["energy"].median()]["popularity"].mean()
avg_pop_low_energy = filtered_df[filtered_df["energy"] < filtered_df["energy"].median()]["popularity"].mean()

st.markdown(
    f"""
    <div class='insight-box'>
    Based on the current dashboard filters, the selected Spotify tracks have an average popularity score of
    <b>{avg_popularity:.2f}</b>. The dashboard shows that music popularity is not explained by one feature only.
    Energy, danceability, genre, artist presence, and mood-related features all help describe how tracks differ from
    one another. In this selection, higher-energy tracks have an average popularity score of
    <b>{avg_pop_high_energy:.2f}</b>, while lower-energy tracks have an average popularity score of
    <b>{avg_pop_low_energy:.2f}</b>. This suggests that audio characteristics can support music trend analysis,
    but genre and artist recognition should also be considered when interpreting popularity.
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Footer / Attribution
# -----------------------------
st.markdown(
    """
    <div class='footer-box'>
    <b>Data Source Attribution:</b> Spotify Tracks Dataset from Kaggle. The dataset contains Spotify track metadata,
    genre labels, and audio features such as popularity, danceability, energy, tempo, valence, and acousticness.
    Used for educational dashboard development and music trend analysis.<br><br>
    <b>Note:</b> This dashboard is for academic and educational purposes only. The analysis describes patterns in the
    provided dataset and should not be interpreted as official Spotify platform statistics.
    </div>
    """,
    unsafe_allow_html=True
)
