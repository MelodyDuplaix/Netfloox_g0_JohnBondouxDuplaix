import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os

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
# Mise à jour avec les chemins corrects pour les fichiers uploadés
def load_data():
    basics = load_tsv(r"C:\Users\sbond\Desktop\Netfloox\1-Data\data\title.basics-10k.tsv")
    ratings = load_tsv(r"C:\Users\sbond\Desktop\Netfloox\1-Data\data\title.ratings-10k.tsv")
    crew = load_tsv(r"C:\Users\sbond\Desktop\Netfloox\1-Data\data\title.crew-10k.tsv")
    principals = load_tsv(r"C:\Users\sbond\Desktop\Netfloox\1-Data\data\title.principals-10k.tsv")
    return basics, ratings, crew, principals

# Étape 2 : Préparer et fusionner les données
def prepare_data():
    basics, ratings, crew, principals = load_data()

    if basics is None or ratings is None:
        st.stop()

    # Vérification des colonnes nécessaires
    if 'tconst' not in basics.columns:
        st.error("La colonne 'tconst' est absente dans le fichier title.basics-10k.tsv")
        st.stop()

    if 'tconst' not in ratings.columns:
        st.error("La colonne 'tconst' est absente dans le fichier title.ratings-10k.tsv")
        st.stop()

    # Fusion des tables basics et ratings
    df = pd.merge(basics, ratings, on='tconst', how='inner')

    # Exemple de fusion avec principals pour ajouter des acteurs
    if 'tconst' in principals.columns:
        df = pd.merge(df, principals, on='tconst', how='left')
    else:
        st.warning("La colonne 'tconst' est absente dans le fichier title.principals-10k.tsv. Fusion ignorée.")

    # Nettoyage et filtrage des colonnes inutiles
    required_columns = ['tconst', 'primaryTitle', 'genres', 'averageRating', 'numVotes']
    for col in required_columns:
        if col not in df.columns:
            st.error(f"La colonne requise '{col}' est absente après la fusion.")
            st.stop()

    df = df[required_columns]
    df = df.rename(columns={
        'primaryTitle': 'title',
        'averageRating': 'rating',
        'numVotes': 'votes',
        'genres': 'genre',
    })

    return df

# Étape 3 : Analyse des données et visualisation
def plot_genre_distribution(df):
    """Affiche la distribution des 10 genres les plus fréquents."""
    """Affiche la distribution des 10 genres les plus fréquents."""
    genre_counts = df['genre'].value_counts().head(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=genre_counts.index, y=genre_counts.values)
    plt.title("Top 10 des genres de films les plus fréquents")
    plt.title("Répartition des genres de films")
    plt.xlabel("Genre")
    plt.ylabel("Nombre")
    plt.xticks(rotation=45)
    st.pyplot(plt)

# Étape 4 : Fonctionnalité de recommandation
def recommend_movies(movie_title, df):
    """Recommande des films basés sur le genre, sans répétition dans les tconst et titres."""
    """Recommande des films basés sur le genre, sans répéter le film d'origine."""
    movie = df[df['title'].str.contains(movie_title, case=False, na=False)]
    if movie.empty:
        st.warning("Aucun film trouvé avec ce titre.")
        return pd.DataFrame()
    genre = movie['genre'].iloc[0] if not movie.empty else None
    recommendations = (
        df[(df['genre'] == genre) & (~df['tconst'].isin(movie['tconst'].tolist())) & (~df['title'].isin(movie['title'].tolist()))]
        .sort_values(by='rating', ascending=False)
        .drop_duplicates(subset=['tconst', 'title'])
        .head(5)
        if genre else pd.DataFrame()
    )
    return recommendations

# Étape 5 : Modèle de prédiction de popularité
from sklearn.metrics import mean_squared_error
import numpy as np

try:
    # Import de la fonction root_mean_squared_error si disponible
    from sklearn.metrics import root_mean_squared_error
    USE_NEW_RMSE = True
except ImportError:
    USE_NEW_RMSE = False

def train_popularity_model(df):
    """Entraîne un modèle de prédiction de popularité et calcule manuellement le RMSE."""
    X = df[['votes', 'rating']]  # Exemple de caractéristiques
    y = df['rating']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    from sklearn.metrics import mean_squared_error, mean_squared_error as root_mean_squared_error
    from sklearn.metrics import mean_squared_error as root_mean_squared_error
# Correction pour remplacer `squared=False` par une méthode explicite dans les versions futures
    import numpy as np
        # Calcul du RMSE avec la nouvelle fonction ou méthode alternative
    if USE_NEW_RMSE:
        rmse = root_mean_squared_error(y_test, y_pred)
    else:
        mse = np.mean((y_test - y_pred) ** 2)  # Mean Squared Error
        rmse = np.sqrt(mse)  # Root Mean Squared Error
    st.write(f"RMSE du modèle : {rmse}")
    return model

# Étape 6 : Interface Streamlit
def display_top_actors_and_directors(df):
    """Affiche les 10 acteurs/actrices et réalisateurs les plus présents."""
    # Acteurs et actrices
    top_actors = df[df['category'].str.contains('actor|actress', na=False)]
    top_actors_count = top_actors['nconst'].value_counts().head(10)

    st.write("### Top 10 Acteurs/Actrices")
    for actor_id in top_actors_count.index:
        actor_name = top_actors[top_actors['nconst'] == actor_id]['nconst'].iloc[0]
        st.write(f"- {actor_name} ({top_actors_count[actor_id]} films)")

    # Réalisateurs
    top_directors = df[df['category'] == 'director']
    top_directors_count = top_directors['nconst'].value_counts().head(10)

    st.write("### Top 10 Réalisateurs")
    for director_id in top_directors_count.index:
        director_name = top_directors[top_directors['nconst'] == director_id]['nconst'].iloc[0]
        st.write(f"- {director_name} ({top_directors_count[director_id]} films)")

def main():
    st.title("Système d'analyse et de recommandation de films")

    # Menu latéral
    menu = ["Accueil", "Analyse des données", "Recommandations", "Prédiction de popularité"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Accueil":
        st.write("Bienvenue dans le système d'analyse et de recommandation de films !")

    elif choice == "Analyse des données":
        st.subheader("Analyse des données et visualisation")
        df = prepare_data()

        st.write("### Aperçu du dataset")
        st.dataframe(df.head())

        st.write("### Répartition des genres")
        plot_genre_distribution(df)

        st.write("### Top Acteurs/Actrices et Réalisateurs")
        display_top_actors_and_directors(df)
    st.title("Système d'analyse et de recommandation de films")

    # Menu latéral
    menu = ["Accueil", "Analyse des données", "Recommandations", "Prédiction de popularité"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Accueil":
        st.write("Bienvenue dans le système d'analyse et de recommandation de films !")

    elif choice == "Analyse des données":
        st.subheader("Analyse des données et visualisation")
        df = prepare_data()

        st.write("### Aperçu du dataset")
        st.dataframe(df.head())

        st.write("### Répartition des genres")
        plot_genre_distribution(df)

    elif choice == "Recommandations":
        st.subheader("Recommandations de films")
        df = prepare_data()

        movie_title = st.text_input("Entrez un titre de film")
        if st.button("Recommander"):
            recommendations = recommend_movies(movie_title, df)
            if not recommendations.empty:
                st.write("### Films recommandés")
                st.dataframe(recommendations)

    elif choice == "Prédiction de popularité":
        st.subheader("Prédire la popularité d'un film")
        df = prepare_data()

        model = train_popularity_model(df)
        st.write("### Modèle de prédiction de popularité entraîné !")

if __name__ == "__main__":
    main()
