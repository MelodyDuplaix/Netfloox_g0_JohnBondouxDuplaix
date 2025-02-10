import os
import sys
sys.path.append('..')
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
from sqlalchemy import create_engine
from dotenv import load_dotenv
from nltk import PorterStemmer
import nltk
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Imports supplémentaires pour la recommandation
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Charger le fichier .env et vérifier DATABASE_URL
load_dotenv()
if os.getenv("DATABASE_URL") is None:
    st.error(
        "Erreur : La variable DATABASE_URL n'est pas définie dans le fichier .env.\n"
        "Veuillez ajouter une ligne comme :\n\nDATABASE_URL=postgresql://username:password@host:port/database_name"
    )
    st.stop()

# Téléchargement des ressources NLTK (à exécuter une seule fois)
nltk.download('wordnet')
nltk.download('omw-1.4')

# Nom du fichier de cache local (sera créé dans le même dossier que ce script)
CACHE_FILE = "data_cache.parquet"

# ---------------------------------------------------------------------
# PARTIE 1 : Extraction et Nettoyage des Données
# ---------------------------------------------------------------------

def get_extracted_features(line_number: int = 200) -> pd.DataFrame:
    """
    Extraction des données issues de plusieurs tables, fusion et agrégation.
    Si un cache local existe (fichier Parquet), les données sont chargées depuis ce fichier.
    Sinon, elles sont extraites depuis la base de données et sauvegardées en cache.
    """
    if os.path.exists(CACHE_FILE):
        st.write("Chargement des données depuis le cache local...")
        df_final = pd.read_parquet(CACHE_FILE)
        return df_final

    st.write("Extraction des données depuis la base de données...")
    database_url = os.getenv("DATABASE_URL")
    if database_url is None:
        raise ValueError("DATABASE_URL n'est pas défini dans le fichier .env.")
    engine = create_engine(database_url)

    # Extraction de la table title_basics
    query = f"""
    SELECT tconst, titletype, primarytitle, isadult, startyear, genres, averagerating, numvotes 
    FROM sebastien.title_basics 
    WHERE startyear IS NOT NULL 
    ORDER BY startyear DESC 
    LIMIT {line_number};
    """
    df = pd.read_sql_query(query, engine)
    st.write("Table 1/4 chargée.")

    # Extraction et fusion avec la table title_episode
    query = "SELECT * FROM sebastien.title_episode;"
    df_episode = pd.read_sql_query(query, engine)
    df_merge = df.merge(df_episode, on="tconst", how="left")
    df_merge_parent = df.merge(df_episode, left_on="tconst", right_on="parenttconst", how="left", suffixes=('', '_parent'))
    df = pd.concat([df_merge, df_merge_parent], ignore_index=True)
    df = df.groupby('tconst').agg({
        'titletype': 'first',
        'primarytitle': 'first',
        'isadult': 'first',
        'startyear': 'first',
        'genres': 'first',
        'averagerating': 'first',
        'numvotes': 'first',
        'seasonnumber': 'max',
        'episodenumber': 'max'
    }).reset_index()
    st.write("Table 2/4 fusionnée.")

    # Extraction de la table title_akas
    query_title_akas = f"""
    SELECT 
        ta.tconst, 
        COUNT(DISTINCT ta.region) AS regionnumber, 
        ARRAY_AGG(ta.region) AS regionlist
    FROM 
        sebastien.title_akas ta
    WHERE 
        ta.tconst IN (
            SELECT tconst FROM sebastien.title_basics 
            WHERE startyear IS NOT NULL 
            ORDER BY startyear DESC LIMIT {line_number})
    GROUP BY 
        ta.tconst;
    """
    df_akas = pd.read_sql_query(query_title_akas, engine)
    df = df.merge(df_akas, on="tconst", how="left")
    def replace_and_filter(region_list):
        if isinstance(region_list, (list, np.ndarray)):
            return [region for region in region_list if region != '\\N' and region != '']
        if pd.isnull(region_list):
            return []
        return region_list
    df_akas['regionlist'] = df_akas['regionlist'].apply(replace_and_filter)
    st.write("Table 3/4 chargée.")

    # Extraction de la table title_principals
    query_title_principals = f"""
    SELECT 
        tconst,
        category,
        primaryname
    FROM 
        sebastien.title_principals ta
    JOIN 
        sebastien.name_basics nb 
    ON 
        ta.nconst = nb.nconst
    WHERE 
        ta.tconst IN (
            SELECT tconst FROM sebastien.title_basics  
            WHERE startyear IS NOT NULL 
            ORDER BY startyear DESC LIMIT {line_number});
    """
    df_principals = pd.read_sql_query(query_title_principals, engine)
    df_principals = df_principals.groupby(['tconst', 'category'])['primaryname'].agg(list).unstack(fill_value=[]).reset_index()
    for col in ["actor", "self", "producer", "actress", "director"]:
        if col not in df_principals.columns:
            df_principals[col] = [[] for _ in range(len(df_principals))]
    df_principals = df_principals[["tconst", "actor", "self", "producer", "actress", "director"]]
    df_final = pd.merge(df, df_principals, on='tconst', how='left')
    st.write("Table 4/4 fusionnée.")

    df_final.to_parquet(CACHE_FILE)
    st.write("Données extraites et sauvegardées dans le cache local.")
    return df_final

def clean_list(list_str) -> list:
    """
    Convertit une chaîne représentant une liste en une vraie liste.
    """
    if isinstance(list_str, (list, np.ndarray)):
        return list(list_str)
    if list_str is None:
        return []
    s = str(list_str).strip("[]").replace("'", "").replace('"', "")
    if s == "":
        return []
    return [item.strip() for item in s.split(",") if item.strip()]

def clean_region_list(list_str) -> list:
    """
    Nettoie une liste de régions en supprimant les valeurs indésirables.
    """
    if isinstance(list_str, (list, np.ndarray)):
        return list(list_str)
    if list_str is None:
        return []
    s = str(list_str).strip("[]").replace("'", "").replace('"', "").replace("\\\\N", "")
    if s == "":
        return []
    return [item.strip() for item in s.split(",") if item.strip()]

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoyage des données et feature engineering.
    """
    # Conversion et gestion des valeurs manquantes
    df['numvotes'] = pd.to_numeric(df['numvotes'].replace("\\N", np.nan), errors='coerce')
    df['averagerating'] = pd.to_numeric(df['averagerating'].replace("\\N", np.nan), errors='coerce')
    df['numvotes'].fillna(0, inplace=True)
    df['averagerating'].fillna(df['averagerating'].mean(), inplace=True)
    
    C = df['averagerating'].mean()
    m = 1000
    def weighted_rating(x, m=m, C=C):
        v = x['numvotes']
        R = x['averagerating']
        return (v / (v + m) * R) + (m / (v + m) * C)
    df['weighted_score'] = df.apply(weighted_rating, axis=1)

    def stemming(text: str) -> str:
        ps = PorterStemmer()
        tokens = text.split()
        stemmed_tokens = [ps.stem(token) for token in tokens]
        return " ".join(stemmed_tokens)

    def clean_text(text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s\[\]]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    df['primarytitle'] = df['primarytitle'].apply(lambda x: stemming(clean_text(x)))
    df['genres'] = df['genres'].astype(str).apply(lambda x: x.split(','))

    for col in ["actor", "actress", "self", "producer", "director"]:
        df[col] = df[col].apply(clean_list)
        counts = df[col].explode().value_counts()
        top_names = counts.head(10).index.tolist()  # Conserver les 10 noms les plus fréquents
        df[col] = df[col].apply(lambda names: [name for name in names if name in top_names])
    df["regionlist"] = df["regionlist"].apply(clean_region_list)

    return df

# Pour la fonction de recommandation, nous définissons Featurescleaning comme alias de clean_data
def Featurescleaning(df: pd.DataFrame) -> pd.DataFrame:
    return clean_data(df)

# ---------------------------------------------------------------------
# PARTIE 2 : Visualisation
# ---------------------------------------------------------------------

def visualize_data(df: pd.DataFrame):
    st.subheader("Visualisation des données")
    st.write("Top Acteurs")
    if "actor" in df.columns:
        actor_counts = df["actor"].explode().value_counts().head(10)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=actor_counts.values, y=actor_counts.index)
        plt.title("Top 10 Acteurs")
        st.pyplot(plt)
        plt.clf()
    st.write("Top Actrices")
    if "actress" in df.columns:
        actress_counts = df["actress"].explode().value_counts().head(10)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=actress_counts.values, y=actress_counts.index)
        plt.title("Top 10 Actrices")
        st.pyplot(plt)
        plt.clf()
    st.write("Top Réalisateurs")
    if "director" in df.columns:
        director_counts = df["director"].explode().value_counts().head(10)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=director_counts.values, y=director_counts.index)
        plt.title("Top 10 Réalisateurs")
        st.pyplot(plt)
        plt.clf()
    st.write("Top Genres")
    if "genres" in df.columns:
        genre_series = pd.Series([genre for genres in df['genres'] for genre in genres])
        genre_counts = genre_series.value_counts().head(10)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=genre_counts.values, y=genre_counts.index)
        plt.title("Top 10 Genres")
        st.pyplot(plt)
        plt.clf()

# ---------------------------------------------------------------------
# PARTIE 3 : Recommandation
# ---------------------------------------------------------------------

def RecommandationSystem(df, film):
    """
    Recommande des films similaires au film donné en se basant sur la similarité cosinus.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données du film avec des features telles que
                           'startyear', 'seasonnumber', 'episodenumber', 'titletype', 'genres', 'actor', 'actress'
                           et 'primarytitle'.
        film (str): Le titre du film pour lequel la recommandation est demandée.
        
    Returns:
        pd.DataFrame: Un DataFrame contenant les 5 films les plus similaires.
    """
    # Nettoyage des données (on utilise ici notre fonction clean_data via Featurescleaning)
    df = Featurescleaning(df)

    # Sélection des features pertinentes pour la recommandation
    features = df.drop(columns=["averagerating", "numvotes", "weighted_score", "tconst", "primarytitle", "self", "director", "producer"])
    # Renommage éventuel pour éviter les conflits (ex. "self" devient "selfperson")
    features.rename(columns={"self": "selfperson"}, inplace=True)

    # Définition des pipelines pour chaque type de donnée
    yearPipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    seasonEpisodeNumberPipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value=1)),
        ("scaler", StandardScaler())
    ])
    titletypePipeline = Pipeline(steps=[
        ("encoder", OrdinalEncoder())
    ])
    genresPipeline = Pipeline(steps=[
        ('binarizer', CountVectorizer(analyzer=lambda x: set(x)))
    ])
    actorPipeline = Pipeline(steps=[
        ('binarizer', CountVectorizer(analyzer=lambda x: set(x)))
    ])
    actressPipeline = Pipeline(steps=[
        ('binarizer', CountVectorizer(analyzer=lambda x: set(x)))
    ])

    # Création du ColumnTransformer qui applique chaque pipeline sur les colonnes correspondantes
    preprocessing = ColumnTransformer(transformers=[
        ("year", yearPipeline, ["startyear"]),
        ("seasonEpisodeNumber", seasonEpisodeNumberPipeline, ["seasonnumber", "episodenumber"]),
        ("titletype", titletypePipeline, ["titletype"]),
        ("genres", genresPipeline, "genres"),
        ("actor", actorPipeline, "actor"),
        ("actress", actressPipeline, "actress")
    ])

    # Pipeline complet de préparation des données
    modelPipeReco = Pipeline(steps=[
        ("preparation", preprocessing)
    ])

    features_transformed = modelPipeReco.fit_transform(features)
    cosine_sim = cosine_similarity(features_transformed)

    if film not in df["primarytitle"].values:
        raise ValueError(f"Le film '{film}' n'est pas présent dans la colonne 'primarytitle'.")
    index = df[df["primarytitle"] == film].index[0]
    cosine_similarities = cosine_sim[index]
    similar_indices = cosine_similarities.argsort()[::-1][1:6]
    similar_movies = df.iloc[similar_indices]
    
    return similar_movies

# ---------------------------------------------------------------------
# MAIN (Interface Streamlit)
# ---------------------------------------------------------------------

def main():
    st.title("Application de Visualisation et de Recommandation de Films")
    df = get_extracted_features(line_number=200)
    # Nettoyage des données une seule fois et transmission aux différentes fonctionnalités
    df = clean_data(df)
    
    menu = [
        "Visualisation",
        "Recommandation"
    ]
    choice = st.sidebar.selectbox("Menu principal", menu)

    if choice == "Visualisation":
        st.write("Aperçu des données nettoyées :")
        st.dataframe(df.head())
        visualize_data(df)
    elif choice == "Recommandation":
        film = st.text_input("Entrez le titre du film pour lequel vous souhaitez une recommandation")
        if st.button("Obtenir des recommandations"):
            try:
                similar_movies = RecommandationSystem(df, film)
                st.write("Films recommandés :")
                st.dataframe(similar_movies[["primarytitle", "startyear", "genres"]])
            except ValueError as e:
                st.error(str(e))

if __name__ == "__main__":
    main()
