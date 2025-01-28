import pandas as pd  # Bibliothèque pour manipuler les données sous forme de DataFrame
import streamlit as st  # Bibliothèque pour créer des interfaces web interactives
import matplotlib.pyplot as plt  # Bibliothèque pour tracer des graphiques
import seaborn as sns  # Extension de Matplotlib pour des visualisations plus avancées
from sklearn.ensemble import RandomForestRegressor  # Modèle de régression basé sur les forêts aléatoires
from sklearn.model_selection import train_test_split  # Pour diviser les données en ensembles d'entraînement et de test
from sklearn.metrics import mean_squared_error  # Pour calculer les erreurs de prédiction
from sklearn.preprocessing import OneHotEncoder  # Pour encoder les variables catégoriques
from sklearn.compose import ColumnTransformer  # Pour appliquer des transformations sur des colonnes spécifiques
from sklearn.pipeline import Pipeline  # Pour créer des pipelines de traitement et de modélisation
from sklearn.base import BaseEstimator, TransformerMixin  # Base pour créer des classes de transformation personnalisées
from sklearn.pipeline import FunctionTransformer  # Pour créer des transformations simples
from sklearn.preprocessing import MultiLabelBinarizer  # Pour binariser les colonnes multi-étiquettes
from sklearn.feature_extraction.text import TfidfVectorizer  # Pour calculer les scores TF-IDF
from sklearn.metrics.pairwise import linear_kernel  # Pour calculer la similarité cosinus

# Étape 1 : Charger les fichiers TSV
def load_tsv(file_path):
    try:
        data = pd.read_csv(file_path, sep='\t', low_memory=False)  # Chargement du fichier TSV avec séparation par tabulation
        if 'streamlit' in globals(): st.write(f"Fichier chargé : {file_path}")  # Afficher le fichier chargé dans Streamlit
        if 'streamlit' in globals(): st.write(f"Colonnes disponibles : {list(data.columns)}")  # Afficher les colonnes disponibles
        return data  # Retourner le DataFrame chargé
    except FileNotFoundError:
        st.error(f"Fichier non trouvé : {file_path}")  # Gérer le cas où le fichier est introuvable
        return None
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier {file_path} : {e}")  # Gérer d'autres erreurs
        return None

# Charger les fichiers nécessaires
def load_data():
    basics = load_tsv(r"data/title.basics-10k.tsv")  # Charger les données de base des films
    ratings = load_tsv(r"data/title.ratings-10k.tsv")  # Charger les notes des films
    crew = load_tsv(r"data/title.crew-10k.tsv")  # Charger les informations sur l'équipe des films
    principals = load_tsv(r"data/title.principals-10k.tsv")  # Charger les informations principales des films
    names = load_tsv(r"data/name.basics-10k.tsv")  # Charger les noms des personnes associées aux films
    return basics, ratings, crew, principals, names  # Retourner tous les DataFrames chargés

# Étape 2 : Préparer et fusionner les données
def prepare_data():
    basics, ratings, crew, principals, names = load_data()  # Charger toutes les données

    if basics is None or ratings is None or principals is None or names is None:
        st.stop()  # Arrêter l'exécution si une des données est introuvable

    df = pd.merge(basics, ratings, on='tconst', how='inner')  # Fusion des données de base avec les notes
    principals = pd.merge(principals, names[['nconst', 'primaryName']], on='nconst', how='left')  # Ajouter les noms des personnes
    df = pd.merge(df, principals[['tconst', 'primaryName', 'category']], on='tconst', how='left')  # Ajouter les catégories principales

    df = df.rename(columns={
        'primaryTitle': 'title',  # Renommer la colonne pour le titre du film
        'averageRating': 'rating',  # Renommer la colonne pour la note moyenne
        'numVotes': 'votes',  # Renommer la colonne pour le nombre de votes
        'genres': 'genre'  # Renommer la colonne pour les genres
    })

    return df[['tconst', 'title', 'genre', 'rating', 'votes', 'primaryName', 'category']]  # Retourner les colonnes nécessaires

# Analyse des données
def analyze_data(df):
    st.subheader("Analyse des données")  # Titre de la section dans Streamlit

    st.write("### Répartition des genres")  # Titre pour le graphique des genres
    genre_counts = df["genre"].str.split(',').explode().value_counts().head(10)  # Calculer la fréquence des genres
    plt.figure(figsize=(10, 6))  # Définir la taille du graphique
    sns.barplot(x=genre_counts.values, y=genre_counts.index, orient='h')  # Créer un graphique à barres horizontal
    plt.title("Top 10 des genres les plus fréquents")  # Titre du graphique
    plt.xlabel("Nombre de films")  # Étiquette de l'axe X
    plt.ylabel("Genres")  # Étiquette de l'axe Y
    st.pyplot(plt)  # Afficher le graphique dans Streamlit

    st.write("### Répartition des notes moyennes")  # Titre pour le graphique des notes
    plt.figure(figsize=(10, 6))  # Définir la taille du graphique
    sns.histplot(df['rating'], bins=20, kde=True)  # Créer un histogramme des notes avec une courbe KDE
    plt.title("Répartition des notes moyennes")  # Titre du graphique
    plt.xlabel("Note moyenne")  # Étiquette de l'axe X
    plt.ylabel("Nombre de films")  # Étiquette de l'axe Y
    st.pyplot(plt)  # Afficher le graphique dans Streamlit

    st.write("### Top 10 des acteurs/actrices les plus actifs")  # Titre pour le graphique des acteurs
    actors = df[df['category'].str.contains('actor|actress', na=False)]['primaryName'].value_counts().head(10)  # Calculer les acteurs les plus fréquents
    plt.figure(figsize=(10, 6))  # Définir la taille du graphique
    sns.barplot(x=actors.values, y=actors.index, orient='h')  # Créer un graphique à barres horizontal
    plt.title("Top 10 des acteurs/actrices les plus actifs")  # Titre du graphique
    plt.xlabel("Nombre de films")  # Étiquette de l'axe X
    plt.ylabel("Acteurs/Actrices")  # Étiquette de l'axe Y
    st.pyplot(plt)  # Afficher le graphique dans Streamlit

    st.write("### Top 10 des réalisateurs les plus actifs")  # Titre pour le graphique des réalisateurs
    directors = df[df['category'] == 'director']['primaryName'].value_counts().head(10)  # Calculer les réalisateurs les plus fréquents
    plt.figure(figsize=(10, 6))  # Définir la taille du graphique
    sns.barplot(x=directors.values, y=directors.index, orient='h')  # Créer un graphique à barres horizontal
    plt.title("Top 10 des réalisateurs les plus actifs")  # Titre du graphique
    plt.xlabel("Nombre de films")  # Étiquette de l'axe X
    plt.ylabel("Réalisateurs")  # Étiquette de l'axe Y
    st.pyplot(plt)  # Afficher le graphique dans Streamlit

# Système de recommandation basé sur TF-IDF
def recommend_movies_tfidf(df, movie_title):
    st.subheader("Recommandations basées sur TF-IDF")  # Titre de la section

    # Préparer la matrice TF-IDF sur les genres et titres
    df['content'] = df['title'] + " " + df['genre'].fillna("")  # Créer une colonne combinant titre et genres
    tfidf = TfidfVectorizer(stop_words='english')  # Initialiser le vecteur TF-IDF en excluant les mots courants
    tfidf_matrix = tfidf.fit_transform(df['content'])  # Calculer la matrice TF-IDF

    # Trouver l'index du film
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()  # Associer les titres de films à leurs indices
    if movie_title not in indices:
        st.warning("Titre introuvable. Veuillez essayer un autre film.")  # Alerte si le titre est introuvable
        return None

    idx = indices[movie_title]  # Obtenir l'indice du film spécifié

    # Calculer les similarités cosines
    cosine_sim = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()  # Calculer la similarité cosinus

    # Obtenir les indices des films similaires
    similar_indices = cosine_sim.argsort()[:-11:-1]  # Obtenir les 10 films les plus similaires
    similar_movies = df.iloc[similar_indices]  # Sélectionner les films similaires

    # Afficher les recommandations
    st.write("### Films recommandés")  # Titre pour les recommandations
    st.dataframe(similar_movies[['title', 'genre', 'rating', 'votes']])  # Afficher les films recommandés

# Préparer les données pour le modèle prédictif
def prepare_model_data(df):
    features = ['genre', 'primaryName', 'category', 'rating', 'votes']  # Colonnes utilisées comme caractéristiques
    target = 'votes'  # Colonne cible

    df = df.dropna(subset=features)  # Supprimer les lignes avec des valeurs manquantes
    X = df[features]  # Séparer les caractéristiques
    y = df[target]  # Séparer la cible

    return X, y  # Retourner les données préparées

class MultiLabelBinarizerWrapper(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mlb = MultiLabelBinarizer()  # Initialiser le binariseur multi-étiquettes

    def fit(self, X, y=None):
        self.mlb.fit(X)  # Ajuster le binariseur sur les données
        return self

    def transform(self, X):
        return self.mlb.transform(X)  # Transformer les données avec le binariseur

def train_popularity_model(X, y):
    categorical_features = ['primaryName', 'category']  # Colonnes catégoriques
    genre_features = ['genre']  # Colonnes des genres

    preprocessor = ColumnTransformer(
        transformers=[
            ('genre_encoder', Pipeline([
                ('splitter', FunctionTransformer(lambda x: x["genre"].str.split(','))),  # Séparer les genres
                ('mlb', MultiLabelBinarizerWrapper())  # Appliquer le binariseur multi-étiquettes
            ]), genre_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)  # Encoder les colonnes catégoriques
        ]
    )

    model = Pipeline([
        ('preprocessor', preprocessor),  # Ajouter le préprocesseur au pipeline
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))  # Ajouter le modèle de régression
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Diviser les données

    model.fit(X_train, y_train)  # Entraîner le modèle

    return model, X_test, y_test  # Retourner le modèle et les ensembles de test

# Fonction principale
def main():
    st.title("Système d'analyse et de recommandation de films")  # Titre principal de l'application

    menu = ["Accueil", "Analyse des données", "Recommandations", "Estimation de la popularité"]  # Menu principal
    choice = st.sidebar.selectbox("Menu principal", menu)  # Barre latérale pour choisir une option

    if choice == "Accueil":
        st.write("Bienvenue dans le système d'analyse et de recommandation de films !")  # Message d'accueil

    elif choice == "Analyse des données":
        df = prepare_data()  # Préparer les données
        analyze_data(df)  # Analyser les données

    elif choice == "Recommandations":
        df = prepare_data()  # Préparer les données

        movie_title = st.text_input("Entrez un titre de film", key="unique_movie_title_input")  # Entrée utilisateur pour le titre
        if st.button("Recommander", key="unique_recommend_button"):
            recommend_movies_tfidf(df, movie_title)  # Recommander des films

    elif choice == "Estimation de la popularité":
        df = prepare_data()  # Préparer les données

        X, y = prepare_model_data(df)  # Préparer les données pour le modèle
        model, X_test, y_test = train_popularity_model(X, y)  # Entraîner le modèle

        st.write("### Estimation de la popularité des films")  # Titre de la section
        genre = st.multiselect("Genre", options=[x for x in df['genre'].str.split(",").explode().unique() if x != "\\N"], key="unique_genre_input")  # Sélection des genres
        actor = st.selectbox("Acteur/Actrice (facultatif)", options=[None] + list(df['primaryName'].unique()), key="unique_actor_input")  # Sélection d'un acteur
        category = st.selectbox("Catégorie (facultatif)", options=[None] + list(df['category'].unique()), key="unique_category_input")  # Sélection d'une catégorie
        rating = st.slider("Note minimale (facultatif)", min_value=0.0, max_value=10.0, step=0.1, key="unique_rating_input")  # Filtrage par note
        votes = st.slider("Nombre minimum de votes (facultatif)", min_value=0, max_value=int(df['votes'].max()), step=10, key="unique_votes_input")  # Filtrage par votes

        if st.button("Estimer la popularité", key="unique_predict_button"):
            filtered_data = df  # Filtrer les données

            if genre:
                filtered_data = filtered_data[filtered_data['genre'].str.contains('|'.join(genre))]  # Filtrer par genre
            if actor:
                filtered_data = filtered_data[filtered_data['primaryName'] == actor]  # Filtrer par acteur
            if category:
                filtered_data = filtered_data[filtered_data['category'] == category]  # Filtrer par catégorie
            if rating:
                filtered_data = filtered_data[filtered_data['rating'] >= rating]  # Filtrer par note
            if votes:
                filtered_data = filtered_data[filtered_data['votes'] >= votes]  # Filtrer par votes

            if filtered_data.empty:
                st.warning("Aucun film trouvé avec les critères spécifiés.")  # Alerte si aucun film n'est trouvé
            else:
                avg_rating = filtered_data['rating'].mean()  # Calculer la note moyenne
                avg_votes = filtered_data['votes'].mean()  # Calculer le nombre moyen de votes
                st.success(f"Popularité estimée : Note moyenne = {avg_rating:.2f}, Votes moyens = {avg_votes:.0f}")  # Afficher les résultats

if __name__ == "__main__":
    main()  # Exécuter l'application
