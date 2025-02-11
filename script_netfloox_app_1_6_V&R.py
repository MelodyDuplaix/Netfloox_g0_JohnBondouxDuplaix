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

# Imports pour le traitement et la modélisation
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

# Charger le fichier .env et définir DATABASE_URL (valeur par défaut utilisée si non définie)
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://psqladmin:GRETAP4!2025***@netfloox-psqlflexibleserver-0.postgres.database.azure.com:5432/postgres")

# Téléchargement des ressources NLTK (à exécuter une seule fois)
nltk.download('wordnet')
nltk.download('omw-1.4')

# Nom du fichier de cache local
CACHE_FILE = "data_cache.parquet"

# ---------------------------------------------------------------------
# PARTIE 1 : Extraction et Nettoyage des Données
# ---------------------------------------------------------------------

def get_extracted_features(line_number=10000) -> pd.DataFrame:
    if os.path.exists(CACHE_FILE):
        st.write("Chargement des données depuis le cache local...")
        df_final = pd.read_parquet(CACHE_FILE)
        return df_final

    st.write("Extraction des données depuis la base de données...")
    engine = create_engine(DATABASE_URL)

    # Extraction de la table title_basics
    query = f"""
    SELECT tconst, titletype, primarytitle, isadult, startyear, genres, averagerating, numvotes 
    FROM sebastien.title_basics 
    WHERE startyear IS NOT NULL 
    AND averagerating IS NOT NULL
    AND numvotes IS NOT NULL
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
            AND averagerating IS NOT NULL
            AND numvotes IS NOT NULL
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
            AND averagerating IS NOT NULL
            AND numvotes IS NOT NULL
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
    if isinstance(list_str, (list, np.ndarray)):
        return list(list_str)
    if list_str is None:
        return []
    s = str(list_str).strip("[]").replace("'", "").replace('"', "")
    if s == "":
        return []
    return [item.strip() for item in s.split(",") if item.strip()]

def clean_region_list(list_str) -> list:
    if isinstance(list_str, (list, np.ndarray)):
        return list(list_str)
    if list_str is None:
        return []
    s = str(list_str).strip("[]").replace("'", "").replace('"', "").replace("\\\\N", "")
    if s == "":
        return []
    return [item.strip() for item in s.split(",") if item.strip()]

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
  
    df["averagerating"] = df["averagerating"].astype(float)
    df["numvotes"] = df["numvotes"].astype(float)
    df['averagerating'] = pd.to_numeric(df['averagerating'].replace("\\N", np.nan), errors='coerce')    
    C = df['averagerating'].fillna(5).mean()
    m = 1000
    
    def weighted_rating(x, m=m, C=C):
        v = x['numvotes'] if pd.notna(x['numvotes']) else 1
        R = x['averagerating'] if pd.notna(x['averagerating']) else 0
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
        top_names = counts.head(10).index.tolist()
        df[col] = df[col].apply(lambda names: [name for name in names if name in top_names])
    df["regionlist"] = df["regionlist"].apply(clean_region_list)
    
    return df

# Définition locale de Featurescleaning (remplaçant l'import manquant)
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
# PARTIE 3 : Recommandation (strictement le code fourni)
# ---------------------------------------------------------------------

def RecommandationSystem(df, film):
    """
    Recommends movies similar to the given film based on the provided DataFrame.
    
    Args:
        df (pd.DataFrame): The DataFrame (not cleaned but with the features) containing movie data with features such as 'startyear', 'seasonnumber', 'episodenumber', 'titletype', 'genres', 'actor', 'actress', and 'primarytitle'.
        film (str): The title of the film for which recommendations are to be made.
        ValueError: If the specified film is not present in the 'primarytitle' column of the DataFrame.
        
    Returns:
        pd.DataFrame: A DataFrame containing the top 5 movies similar to the given film based on cosine similarity.
    """
    df = Featurescleaning(df)

    # Création des features et de la target
    features = df.drop(columns=["averagerating", "numvotes", "weighted_score", "tconst", "primarytitle", "self", "director", "producer"])
    features.rename(columns={"self": "selfperson"}, inplace=True)
    target = df[["weighted_score"]].fillna(0)
    features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Définir les pipelines
    yearPipeline = Pipeline(steps=[("inputing", SimpleImputer(strategy="median")), ("scaling", StandardScaler())])
    seasonEpisodeNumberPipeline = Pipeline(steps=[("inputing", SimpleImputer(strategy="constant", fill_value=1)), ("scaling", StandardScaler())])
    primarytitlePipeline = Pipeline(steps=[("tfidf", TfidfVectorizer(stop_words="english"))])
    titletypePipeline = Pipeline(steps=[("tfidf", OrdinalEncoder())])
    genresPipeline = Pipeline(steps=[
        ('binarizer', CountVectorizer(analyzer=lambda x: set(x)))
    ])
    actorPipeline = Pipeline(steps=[
            ('binarizer', CountVectorizer(analyzer=lambda x: set(x)))
    ])
    actressPipeline = Pipeline(steps=[
            ('binarizer', CountVectorizer(analyzer=lambda x: set(x)))
    ])

    # Création du ColumnTransformer
    preprocessing = ColumnTransformer(transformers=[
        ("year", yearPipeline, ["startyear"]),
        ("seasonEpisodeNumber", seasonEpisodeNumberPipeline, ["seasonnumber", "episodenumber"]),
        ("titletype", titletypePipeline, ["titletype"]),
        ("genres", genresPipeline, "genres"),
        ("actor", actorPipeline, "actor"),
        ("actress", actressPipeline, "actress")
    ])

    # Création du pipeline de recommandation
    modelPipeReco = Pipeline(steps=[
        ("prep données", preprocessing)
    ])

    # Entraînement du modèle
    features_transformed = modelPipeReco.fit_transform(features)
    cosine_sim = cosine_similarity(features_transformed)

    if film not in df["primarytitle"].values:
        raise ValueError(f"Le film '{film}' n'est pas présent dans la colonne 'primarytitle'.")
    index = df[df["primarytitle"] == film].index[0]
    cosine_similarities = cosine_sim[index]
    similar_indices = cosine_similarities.argsort()[::-1][0:6]
    similar_indices = [i for i in similar_indices if i != index]
    similar_movies = df.iloc[similar_indices]
    
    return similar_movies

# ---------------------------------------------------------------------
# PARTIE 4 : Estimation de la Popularité par Critères
# ---------------------------------------------------------------------

class PopularityPrediction():
    """
    A class used to predict the popularity of TV shows or movies based on various features.
    Methods
    -------
    
    fit(df)
        Fits the model to the provided DataFrame `df`.
        
    predict(features)
        Predicts the popularity score for the given features.
        
    evaluate()
        Evaluates the model on the test set and prints the Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R2) score.
        
    Example
    -------
    >>> import sys
    >>> sys.path.append('..')
    >>> from scripts.PopularitySystem import PopularityPrediction
    >>> from scripts.Cleaning import Featurescleaning
    >>> import pandas as pd
    >>> df = Featurescleaning(pd.read_csv(filepath_or_buffer='../data/all_data_for_10000_lines.csv'))
    >>> model = PopularityPrediction()
    >>> model.fit(df)
    >>> model.evaluate()
    MSE:  0.1234
    MAE:  0.5678
    R2:  0.9101
    >>> data = df.iloc[0:1]
    >>> predictions = model.predict(data)
    """
    
    def __init__(self) -> None:
        self.yearPipeline = Pipeline(steps=[("inputing", SimpleImputer(strategy="median")), ("scaling", StandardScaler())])
        self.seasonEpisodeNumberPipeline = Pipeline(steps=[("inputing", SimpleImputer(strategy="constant", fill_value=1)), ("scaling", StandardScaler())])
        self.primarytitlePipeline = Pipeline(steps=[("tfidf", TfidfVectorizer(stop_words="english"))])
        self.titletypePipeline = Pipeline(steps=[("encoder", OrdinalEncoder())])
        self.genresPipeline = Pipeline(steps=[
        ('binarizer', CountVectorizer(analyzer=lambda x: set(x)))
        ])
        self.actorPipeline = Pipeline(steps=[
            ('binarizer', CountVectorizer(analyzer=lambda x: set(x)))
        ])
        self.actressPipeline = Pipeline(steps=[
            ('binarizer', CountVectorizer(analyzer=lambda x: set(x)))
        ])
        self.preprocessing = ColumnTransformer(transformers=[
            ("year", self.yearPipeline, ["startyear"]),
            ("seasonEpisodeNumber", self.seasonEpisodeNumberPipeline, ["seasonnumber", "episodenumber"]),
            ("titletype", self.titletypePipeline, ["titletype"]),
            ("genres", self.genresPipeline, "genres"),
            ("actor", self.actorPipeline, "actor"),
            ("actress", self.actressPipeline, "actress"),
        ])
        self.modelPipe = Pipeline(steps=[
            ("prep données", self.preprocessing),
            ("model", KNeighborsRegressor())
        ])
    
    def fit(self, df):
        self.df = df
        self.features = df.drop(columns=["averagerating", "numvotes", "weighted_score", "tconst", "primarytitle", "self", "director", "producer"])
        self.features.rename(columns={"self": "selfperson"}, inplace=True)
        self.target = df[["weighted_score"]].fillna(0)
        self.features_train, self.features_test, self.target_train, self.target_test = train_test_split(self.features, self.target, test_size=0.2, random_state=42)
        self.modelPipe.fit( self.features_train, self.target_train)
        
    def predict(self, features):
        return self.modelPipe.predict(features)

    def evaluate(self):
        y_pred = self.modelPipe.predict(self.features_test)
        print("MSE: ", mean_squared_error(self.target_test, y_pred))
        print("MAE: ", mean_absolute_error(self.target_test, y_pred))
        print("R2: ", r2_score(self.target_test, y_pred))

# ---------------------------------------------------------------------
# INTERFACE STREAMLIT
# ---------------------------------------------------------------------

def main():
    st.title("Application de Visualisation, Recommandation et Estimation de la Popularité par Critères")
    
    # Bouton pour actualiser la BD (supprimer le cache)
    if st.sidebar.button("Actualiser la BD"):
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
            st.sidebar.success("Cache supprimé, la BD sera réextrait.")
    
    df = get_extracted_features(line_number=200)
    df = clean_data(df)
    model = PopularityPrediction()
    print(df.head())
    model.fit(df)
    print(model.predict(df.iloc[0:1]))
    print("predictiont")
    
    menu = [
        "Visualisation",
        "Recommandation",
        "Estimation de la Popularité par Critères"
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
    
    elif choice == "Estimation de la Popularité par Critères":
        st.subheader("Estimation de la Popularité par Critères")
        with st.form(key="criteria_form"):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                use_year = st.checkbox("Filtrer par année")
                if use_year:
                    year_val = st.number_input("Année", min_value=1900, max_value=2100, value=2000)
                else:
                    year_val = None
            with col2:
                actor_list = sorted(set([a for sublist in df["actor"] for a in sublist if a]))
                actor_options = ["Aucun"] + actor_list
                actor_choice = st.selectbox("Acteur/Actrice", actor_options)
                if actor_choice == "Aucun":
                    actor_choice = None
            with col3:
                director_list = sorted(set([d for sublist in df["director"] for d in sublist if d]))
                director_options = ["Aucun"] + director_list
                director_choice = st.selectbox("Réalisateur", director_options)
                if director_choice == "Aucun":
                    director_choice = None
            with col4:
                genre_list = sorted(set([g for genres in df["genres"] for g in genres if g]))
                genre_options = ["Aucun"] + genre_list
                genre_choice = st.selectbox("Genre", genre_options)
                if genre_choice == "Aucun":
                    genre_choice = None
            submit_button = st.form_submit_button(label="Estimer la popularité (par critères)")
        
        if submit_button:
            input_data = pd.DataFrame({
                "startyear": [year_val if year_val is not None else int(df["startyear"].median())],
                "seasonnumber": [1.0],
                "episodenumber": [1.0],
                "titletype": [df["titletype"].mode()[0]],
                "genres": [[genre_choice] if genre_choice is not None else []],
                "actor": [[actor_choice] if actor_choice is not None else []],
                "actress": [[]],
                "director": [[director_choice] if director_choice is not None else []]
            })
            
            model = PopularityPrediction()
            # with st.spinner("Entraînement du modèle..."):
            model.fit(df)
            # Affichage des types de données pour vérifier les conversions
            input_data["seasonnumber"] = input_data["seasonnumber"].astype(float)
            input_data["episodenumber"] = input_data["episodenumber"].astype(float)
            
            prediction = model.predict(input_data)
            st.write("Estimation de la popularité (par critères) :", prediction[0])
           
            
if __name__ == "__main__":
    main()
