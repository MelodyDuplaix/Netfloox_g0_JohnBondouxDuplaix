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
    names = load_tsv(r"C:\Users\sbond\Desktop\Netfloox\1-Data\data\name.basics-10k.tsv")
    return basics, ratings, crew, principals, names

# Étape 2 : Préparer et fusionner les données
def prepare_data():
    """Préparer et fusionner les données nécessaires."""
    # Charger les fichiers
    basics, ratings, crew, principals, names = load_data()

    if basics is None or ratings is None or principals is None or names is None:
        st.stop()

    # Vérification des colonnes essentielles
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

    # Fusion basics et ratings
    df = pd.merge(basics, ratings, on='tconst', how='inner')

    # Fusion principals avec names pour obtenir primaryName
    principals = pd.merge(principals, names[['nconst', 'primaryName']], on='nconst', how='left')

    # Fusion principale avec primaryName et category
    df = pd.merge(df, principals[['tconst', 'primaryName', 'category']], on='tconst', how='left')

    # Nettoyage des colonnes et renommage
    df = df.rename(columns={
        'primaryTitle': 'title',
        'averageRating': 'rating',
        'numVotes': 'votes',
        'genres': 'genre',
    })

    final_columns = ['tconst', 'title', 'genre', 'rating', 'votes', 'primaryName', 'category']
    for col in final_columns:
        if col not in df.columns:
            st.error(f"La colonne requise '{col}' est absente après la fusion.")
            st.stop()

    return df[final_columns]

# Étape 3 : Analyse des données et visualisation
def plot_genre_distribution(df):
    """Affiche un graphique des 10 genres les plus fréquents."""
    genre_counts = df['genre'].value_counts().head(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=genre_counts.values, y=genre_counts.index, orient='h')
    plt.title("Top 10 des genres de films les plus fréquents")
    plt.xlabel("Nombre de films")
    plt.ylabel("Genres")
    st.pyplot(plt)

def display_top_actors_and_directors(df):
    """Affiche les graphiques séparés des 10 acteurs/actrices et réalisateurs les plus présents."""
    if 'primaryName' not in df.columns or 'category' not in df.columns:
        st.error("Les colonnes 'primaryName' ou 'category' sont absentes des données.")
        return

    # Acteurs et actrices
    top_actors = df[df['category'].str.contains('actor|actress', na=False)]
    top_actors_count = top_actors['primaryName'].value_counts().head(10)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_actors_count.values, y=top_actors_count.index, orient='h')
    plt.title("Top 10 Acteurs/Actrices")
    plt.xlabel("Nombre de films")
    plt.ylabel("Acteurs/Actrices")
    st.pyplot(plt)

    # Réalisateurs
    top_directors = df[df['category'] == 'director']
    top_directors_count = top_directors['primaryName'].value_counts().head(10)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_directors_count.values, y=top_directors_count.index, orient='h')
    plt.title("Top 10 Réalisateurs")
    plt.xlabel("Nombre de films")
    plt.ylabel("Réalisateurs")
    st.pyplot(plt)

def recommend_movies(movie_title, df):
    """Recommande des films basés sur le genre, sans répétition dans les tconst et titres."""
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

def prepare_model_data(df):
    """Prépare les données pour le modèle prédictif."""
    # Sélection des colonnes pertinentes pour la prédiction
    features = ['genre', 'primaryName', 'category']
    target = 'votes'

    # Remplir les valeurs manquantes
    df = df.dropna(subset=features + [target])

    # Créer les variables X (caractéristiques) et y (cible)
    X = df[features]
    y = df[target]

    return X, y

def train_popularity_model(X, y):
    """Entraîne un modèle pour prédire la popularité des films."""
    # Définir un préprocesseur pour encoder les variables catégoriques
    categorical_features = ['genre', 'primaryName', 'category']
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    # Créer un pipeline avec un modèle RandomForest
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entraîner le modèle
    model.fit(X_train, y_train)

    return model, X_test, y_test

def predict_popularity(model, genre, actor, category):
    """Prédit la popularité d'un film en fonction de ses caractéristiques."""
    # Vérifiez et assurez-vous que les valeurs sont sous forme de chaînes
    genre = str(genre)
    actor = str(actor)
    category = str(category)

    # Créer une entrée pour prédiction
    input_data = pd.DataFrame({
        'genre': [genre],
        'primaryName': [actor],
        'category': [category]
    })

    # Essayer de prédire et capturer les erreurs
    try:
        prediction = model.predict(input_data)[0]
    except ValueError as e:
        st.error(f"Erreur : Les caractéristiques fournies ne sont pas compatibles avec le modèle. {e}")
        return None

    return prediction

def main_popularity_estimation():
    st.subheader("Estimation de la popularité des films")
    df = prepare_data()

    # Préparer les données et entraîner le modèle
    X, y = prepare_model_data(df)
    model, X_test, y_test = train_popularity_model(X, y)

    # Interface utilisateur pour estimer la popularité
    st.write("### Entrez les caractéristiques du film :")
    genre = st.selectbox("Genre", options=df['genre'].unique(), key="unique_genre_input")
    actor = st.selectbox("Acteur/Actrice", options=df['primaryName'].unique(), key="unique_actor_input")
    category = st.selectbox("Catégorie", options=df['category'].unique(), key="unique_category_input")

    if st.button("Estimer la popularité", key="unique_predict_button"):
        if genre not in df['genre'].unique():
            st.error("Le genre saisi est inconnu. Veuillez sélectionner un genre valide.")
        elif actor not in df['primaryName'].unique():
            st.error("L'acteur ou actrice saisi(e) est inconnu(e). Veuillez sélectionner une valeur valide.")
        elif category not in df['category'].unique():
            st.error("La catégorie saisie est inconnue. Veuillez sélectionner une valeur valide.")
        else:
            popularity = predict_popularity(model, genre, actor, category)
            if popularity is not None:
                st.success(f"La popularité estimée est de : {popularity:.2f} votes")

def main():
    st.title("Système d'analyse et de recommandation de films")

    # Menu latéral
    menu = ["Accueil", "Analyse des données", "Recommandations", "Estimation de la popularité"]
    choice = st.sidebar.selectbox("Menu principal", menu, key="unique_main_menu")

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

    elif choice == "Recommandations":
        st.subheader("Recommandations de films")
        df = prepare_data()

        movie_title = st.text_input("Entrez un titre de film", key="unique_movie_title_input")
        if st.button("Recommander", key="unique_recommend_button"):
            recommendations = recommend_movies(movie_title, df)
            if not recommendations.empty:
                st.write("### Films recommandés")
                st.dataframe(recommendations)

    elif choice == "Estimation de la popularité":
        main_popularity_estimation()

if __name__ == "__main__":
    main()
