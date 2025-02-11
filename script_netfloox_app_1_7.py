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
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression

# Charger le fichier .env et définir DATABASE_URL (valeur par défaut utilisée si non définie)
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://psqladmin:GRETAP4!2025***@netfloox-psqlflexibleserver-0.postgres.database.azure.com:5432/postgres")

# Téléchargement des ressources NLTK (à exécuter une seule fois)
nltk.download('wordnet')
nltk.download('omw-1.4')

# Nom du fichier de cache local
CACHE_FILE = "data_cache.parquet"

# ============================================================================
# PARTIE 1 : Extraction et Nettoyage des Données
# ============================================================================

def get_extracted_features(line_number=5000000) -> pd.DataFrame:
    if os.path.exists(CACHE_FILE):
        st.write("Chargement des données depuis le cache local...")
        df_final = pd.read_parquet(CACHE_FILE)
        return df_final

    st.write("Extraction des données depuis la base de données...")
    engine = create_engine(DATABASE_URL)

    # Extraction de la table title_basics avec uniquement les données postérieures à 2000
    query = f"""
    SELECT tconst, titletype, primarytitle, isadult, startyear, genres, averagerating, numvotes 
    FROM sebastien.title_basics 
    WHERE startyear IS NOT NULL 
      AND averagerating IS NOT NULL
      AND numvotes IS NOT NULL
      AND startyear > 2000
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
              AND startyear > 2000
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
              AND startyear > 2000
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

def Featurescleaning(df: pd.DataFrame) -> pd.DataFrame:
    return clean_data(df)

# ============================================================================
# PARTIE 2 : Visualisation
# ============================================================================

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

# ============================================================================
# PARTIE 3 : Recommandation
# ============================================================================

def RecommandationSystem(df, film):
    """
    Recommande des films similaires au film donné en se basant sur la similarité
    des features.
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

    # Entraînement du modèle et calcul de la similarité cosinus
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

# ============================================================================
# PARTIE 4 : Estimation de la Popularité par Critères
# ============================================================================

class PopularityPrediction():
    """
    Classe pour prédire la popularité (weighted_score) en fonction de diverses features.
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
        self.modelPipe.fit(self.features_train, self.target_train)
        
    def predict(self, features):
        return self.modelPipe.predict(features)

    def evaluate(self):
        y_pred = self.modelPipe.predict(self.features_test)
        print("MSE: ", mean_squared_error(self.target_test, y_pred))
        print("MAE: ", mean_absolute_error(self.target_test, y_pred))
        print("R2: ", r2_score(self.target_test, y_pred))

# ============================================================================
# Fonction d'affinage combiné du score (numérique + catégoriel)
# ============================================================================

def affiner_score_combined(df, input_numeric, input_categorical_tokens):
    """
    Pour la partie numérique, on utilise les colonnes 'startyear', 'averagerating' et 'numvotes'.  
    Pour la partie catégorielle, on construit une nouvelle feature 'cat_features' en concaténant 
    les listes d'acteurs, d'actrices, de réalisateurs et de genres, de sorte que chaque nom (même composé) reste intact.  
    Les deux prédictions (numérique et catégorielle) sont ensuite combinées (ici à poids égal) pour obtenir une estimation affinée.
    
    Parameters:
        df: DataFrame complet
        input_numeric: DataFrame contenant les colonnes numériques (pour la prédiction numérique)
        input_categorical_tokens: Liste de tokens (ex. ['Brad Pitt', 'Spielberg', 'Action']) correspondant aux choix utilisateur
        
    Returns:
        num_model, cat_model, vectorizer, combined_pred
    """
    y = df['weighted_score']
    
    # Modèle numérique
    X_num = df[['startyear', 'averagerating', 'numvotes']]
    X_train_num, X_test_num, y_train_num, y_test_num = train_test_split(X_num, y, test_size=0.2, random_state=42)
    num_model = LinearRegression()
    num_model.fit(X_train_num, y_train_num)
    num_pred = num_model.predict(input_numeric)  # prédiction numérique
    
    # Préparation de la feature catégorielle en combinant listes sans découpage supplémentaire
    df = df.copy()
    def join_cat(row):
        # Concatène les listes pour conserver les noms complets (par exemple, "Brad Pitt" reste intact)
        actors = row["actor"] if isinstance(row["actor"], list) else []
        actresses = row["actress"] if isinstance(row["actress"], list) else []
        directors = row["director"] if isinstance(row["director"], list) else []
        genres = row["genres"] if isinstance(row["genres"], list) else []
        return actors + actresses + directors + genres
    df["cat_features"] = df.apply(join_cat, axis=1)
    
    # Vectorisation : on utilise CountVectorizer avec un analyseur personnalisé qui renvoie directement la liste
    vectorizer = CountVectorizer(analyzer=lambda x: x, preprocessor=None, token_pattern=None)
    X_cat = vectorizer.fit_transform(df["cat_features"])
    X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(X_cat, y, test_size=0.2, random_state=42)
    cat_model = LinearRegression()
    cat_model.fit(X_train_cat, y_train_cat)
    
    # Transformation de l'input catégoriel : input_categorical_tokens est une liste de tokens
    input_cat_vec = vectorizer.transform([input_categorical_tokens])
    cat_pred = cat_model.predict(input_cat_vec)
    
    # Combinaison simple des deux prédictions (pondération égale)
    combined_pred = 0.5 * num_pred[0] + 0.5 * cat_pred[0]
   
    st.write("Prédiction numérique :", num_pred[0])
    st.write("Prédiction catégorielle :", cat_pred[0])
    st.write("Prédiction combinée :", combined_pred)
    
    return num_model, cat_model, vectorizer, combined_pred

# ============================================================================
# INTERFACE STREAMLIT
# ============================================================================

def main():
    st.title("Application de Visualisation, Recommandation et Estimation de la Popularité par Critères")
    
    # Bouton pour actualiser la BD (supprimer le cache)
    if st.sidebar.button("Actualiser la BD"):
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
            st.sidebar.success("Cache supprimé, la BD sera réextrait.")
    
    df = get_extracted_features(line_number=200)
    df = clean_data(df)
    model_pop = PopularityPrediction()
    st.write("Aperçu des données nettoyées :")
    st.dataframe(df.head())
    
    menu = [
        "Visualisation",
        "Recommandation",
        "Estimation de la Popularité par Critères"
    ]
    choice = st.sidebar.selectbox("Menu principal", menu)
    
    if choice == "Visualisation":
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
            # Pour la partie numérique : on utilise 'startyear', 'averagerating' et 'numvotes'
            startyear_input = year_val if year_val is not None else int(df["startyear"].median())
            median_rating = df["averagerating"].median()
            median_votes = df["numvotes"].median()
            
            input_features_numeric = pd.DataFrame({
                "startyear": [startyear_input],
                "averagerating": [median_rating],
                "numvotes": [median_votes]
            })
            
            # Pour la partie catégorielle : on récupère les choix (s'ils existent) sous forme de liste de tokens
            input_features_categorical_tokens = []
            if actor_choice is not None:
                input_features_categorical_tokens.append(actor_choice)
            if director_choice is not None:
                input_features_categorical_tokens.append(director_choice)
            if genre_choice is not None:
                input_features_categorical_tokens.append(genre_choice)
            
            # Appel de la fonction combinée
            num_model, cat_model, vectorizer, combined_pred = affiner_score_combined(df, input_features_numeric, input_features_categorical_tokens)
            st.write("Estimation affinée combinée de la popularité :", combined_pred)
            
            # Optionnel : utilisation du modèle de prédiction de popularité basé sur KNN
            model_pop.fit(df)
            # Pour KNN, on crée un DataFrame contenant toutes les colonnes requises par le pipeline
            input_features_knn = pd.DataFrame({
                "startyear": [startyear_input],
                "seasonnumber": [1.0],                   # valeur par défaut
                "episodenumber": [1.0],                  # valeur par défaut
                "titletype": [df["titletype"].mode()[0]],  # modalité la plus fréquente
                "genres": [[genre_choice] if genre_choice is not None else []],
                "actor": [[actor_choice] if actor_choice is not None else []],
                "actress": [[]]  # pas de sélection pour actrices
            })
            knn_prediction = model_pop.predict(input_features_knn)
            st.write("Estimation de la popularité par KNN :", knn_prediction[0])
    
if __name__ == "__main__":
    main()
