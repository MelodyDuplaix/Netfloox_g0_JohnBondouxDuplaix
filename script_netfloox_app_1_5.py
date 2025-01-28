import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FunctionTransformer
from sklearn.preprocessing import MultiLabelBinarizer

# Étape 1 : Charger les fichiers TSV
def load_tsv(file_path):
    try:
        data = pd.read_csv(file_path, sep='\t', low_memory=False)
        if 'streamlit' in globals(): st.write(f"Fichier chargé : {file_path}")
        if 'streamlit' in globals(): st.write(f"Colonnes disponibles : {list(data.columns)}")
        return data
    except FileNotFoundError:
        st.error(f"Fichier non trouvé : {file_path}")
        return None
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier {file_path} : {e}")
        return None

# Charger les fichiers nécessaires
def load_data():
    basics = load_tsv(r"data/title.basics-10k.tsv")
    ratings = load_tsv(r"data/title.ratings-10k.tsv")
    crew = load_tsv(r"data/title.crew-10k.tsv")
    principals = load_tsv(r"data/title.principals-10k.tsv")
    names = load_tsv(r"data/name.basics-10k.tsv")
    return basics, ratings, crew, principals, names

# Étape 2 : Préparer et fusionner les données
def prepare_data():
    basics, ratings, crew, principals, names = load_data()

    if basics is None or ratings is None or principals is None or names is None:
        st.stop()

    required_columns = {
        "basics": ['tconst', 'primaryTitle', 'genres'],
        "ratings": ['tconst', 'averageRating', 'numVotes'],
        "principals": ['tconst', 'nconst', 'category'],
        "names": ['nconst', 'primaryName']
    }

    for dataset_name, columns in required_columns.items():
        dataset = locals()[dataset_name]
        for col in columns:
            if col not in dataset.columns:
                st.error(f"La colonne '{col}' est absente dans le fichier {dataset_name}.")
                st.stop()

    df = pd.merge(basics, ratings, on='tconst', how='inner')
    principals = pd.merge(principals, names[['nconst', 'primaryName']], on='nconst', how='left')
    df = pd.merge(df, principals[['tconst', 'primaryName', 'category']], on='tconst', how='left')

    df = df.rename(columns={
        'primaryTitle': 'title',
        'averageRating': 'rating',
        'numVotes': 'votes',
        'genres': 'genre'
    })

    final_columns = ['tconst', 'title', 'genre', 'rating', 'votes', 'primaryName', 'category']
    for col in final_columns:
        if col not in df.columns:
            st.error(f"La colonne requise '{col}' est absente après la fusion.")
            st.stop()

    return df[final_columns]

# Étape 3 : Analyse et visualisation des données
def visualisation_data(df):
    st.subheader("Visualisation des données")

    st.write("### Répartition des genres")
    genre_counts = df["genre"].str.split(',').explode().value_counts().head(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=genre_counts.values, y=genre_counts.index, orient='h')
    plt.title("Top 10 des genres les plus fréquents")
    plt.xlabel("Nombre de films")
    plt.ylabel("Genres")
    st.pyplot(plt)

    st.write("### Répartition des notes moyennes")
    plt.figure(figsize=(10, 6))
    sns.histplot(df['rating'], bins=20, kde=True)
    plt.title("Répartition des notes moyennes")
    plt.xlabel("Note moyenne")
    plt.ylabel("Nombre de films")
    st.pyplot(plt)

    st.write("### Top 10 des acteurs/actrices les plus actifs")
    actors = df[df['category'].str.contains('actor|actress', na=False)]['primaryName'].value_counts().head(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=actors.values, y=actors.index, orient='h')
    plt.title("Top 10 des acteurs/actrices les plus actifs")
    plt.xlabel("Nombre de films")
    plt.ylabel("Acteurs/Actrices")
    st.pyplot(plt)

    st.write("### Top 10 des réalisateurs les plus actifs")
    directors = df[df['category'] == 'director']['primaryName'].value_counts().head(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=directors.values, y=directors.index, orient='h')
    plt.title("Top 10 des réalisateurs les plus actifs")
    plt.xlabel("Nombre de films")
    plt.ylabel("Réalisateurs")
    st.pyplot(plt)

# Système de recommandation
def recommend_movies(df, movie_title):
    movie = df[df['title'].str.contains(movie_title, case=False, na=False)]
    if movie.empty:
        st.warning("Aucun film trouvé avec ce titre.")
        return None

    genre = movie['genre'].str.split(",").explode().unique() if not movie.empty else []
    recommendations = df[(df['genre'].str.contains('|'.join(genre))) & (~df['title'].str.contains(movie_title, case=False, na=False))]
    recommendations = recommendations.sort_values(by='rating', ascending=False).drop_duplicates(subset=['title']).head(5)

    if recommendations.empty:
        st.warning("Aucune recommandation trouvée.")
    else:
        st.write("### Films recommandés")
        st.dataframe(recommendations[['title', 'genre', 'rating', 'votes']])

# Préparer les données pour le modèle prédictif
def prepare_model_data(df):
    features = ['genre', 'primaryName', 'category', 'rating', 'votes']
    target = 'votes'

    df = df.dropna(subset=features)
    X = df[features]
    y = df[target]

    return X, y

class MultiLabelBinarizerWrapper(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mlb = MultiLabelBinarizer()

    def fit(self, X, y=None):
        self.mlb.fit(X)
        return self

    def transform(self, X):
        return self.mlb.transform(X)

def train_popularity_model(X, y):
    categorical_features = ['primaryName', 'category']
    genre_features = ['genre']

    preprocessor = ColumnTransformer(
        transformers=[
            ('genre_encoder', Pipeline([
                ('splitter', FunctionTransformer(lambda x: x["genre"].str.split(','))),
                ('mlb', MultiLabelBinarizerWrapper())
            ]), genre_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)

    return model, X_test, y_test

# Fonction principale
def main():
    st.title("Système d'analyse et de recommandation de films")

    menu = ["Accueil", "Visualisation des données", "Recommandations", "Estimation de la popularité"]
    choice = st.sidebar.selectbox("Menu principal", menu, key="unique_main_menu")

    if choice == "Accueil":
        st.write("Bienvenue dans le système d'analyse et de recommandation de films !")

    elif choice == "Visualisation des données":
        df = prepare_data()
        visualisation_data(df)

    elif choice == "Recommandations":
        df = prepare_data()

        movie_title = st.text_input("Entrez un titre de film", key="unique_movie_title_input")
        if st.button("Recommander", key="unique_recommend_button"):
            recommend_movies(df, movie_title)

    elif choice == "Estimation de la popularité":
        df = prepare_data()

        X, y = prepare_model_data(df)
        model, X_test, y_test = train_popularity_model(X, y)

        st.write("### Entrez les caractéristiques du film :")
        genre = st.multiselect("Genre", options=[x for x in df['genre'].str.split(",").explode().unique() if x != "\\N"], key="unique_genre_input")
        actor = st.selectbox("Acteur/Actrice (facultatif)", options=[None] + list(df['primaryName'].unique()), key="unique_actor_input")
        category = st.selectbox("Catégorie (facultatif)", options=[None] + list(df['category'].unique()), key="unique_category_input")
        rating = st.slider("Note minimale (facultatif)", min_value=0.0, max_value=10.0, step=0.1, key="unique_rating_input")
        votes = st.slider("Nombre minimum de votes (facultatif)", min_value=0, max_value=int(df['votes'].max()), step=10, key="unique_votes_input")

        if st.button("Estimer la popularité", key="unique_predict_button"):
            filtered_data = df

            if genre:
                filtered_data = filtered_data[filtered_data['genre'].str.contains('|'.join(genre))]
            if actor:
                filtered_data = filtered_data[filtered_data['primaryName'] == actor]
            if category:
                filtered_data = filtered_data[filtered_data['category'] == category]
            if rating:
                filtered_data = filtered_data[filtered_data['rating'] >= rating]
            if votes:
                filtered_data = filtered_data[filtered_data['votes'] >= votes]

            if filtered_data.empty:
                st.warning("Aucun film trouvé avec les critères spécifiés.")
            else:
                avg_rating = filtered_data['rating'].mean()
                avg_votes = filtered_data['votes'].mean()
                st.success(f"Popularité estimée : Note moyenne = {avg_rating:.2f}, Votes moyens = {avg_votes:.0f}")

if __name__ == "__main__":
    main()
