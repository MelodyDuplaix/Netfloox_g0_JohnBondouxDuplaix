# Importation des modules standard pour la gestion des fichiers, des données, de l'interface et des calculs
import os  # Pour les opérations système, notamment vérifier l'existence d'un fichier
import sys  # Pour modifier le chemin d'importation
sys.path.append('..')  # Ajoute le dossier parent au chemin d'importation
import pandas as pd  # Pour la manipulation de DataFrame
import streamlit as st  # Pour créer l'interface web avec Streamlit
import matplotlib.pyplot as plt  # Pour la création de graphiques
import seaborn as sns  # Pour la visualisation statistique (graphique amélioré)
import numpy as np  # Pour les calculs numériques
import re  # Pour le traitement et la manipulation de chaînes avec des expressions régulières

# Importation des modules pour se connecter à la base de données et charger les variables d'environnement
from sqlalchemy import create_engine  # Pour créer une connexion à la base de données
from dotenv import load_dotenv  # Pour charger les variables d'environnement depuis un fichier .env

# Importation de NLTK pour le stemming (réduction des mots à leur radical)
from nltk import PorterStemmer
import nltk  # Pour télécharger les ressources NLTK

# Importations pour la création de pipelines et transformations avec Scikit-Learn
from sklearn.pipeline import Pipeline  # Pour créer des pipelines de transformation
from sklearn.preprocessing import FunctionTransformer  # Pour créer des transformateurs personnalisés
from sklearn.model_selection import train_test_split  # Pour diviser le jeu de données en ensembles d'entraînement et de test
from sklearn.metrics.pairwise import cosine_similarity  # Pour calculer la similarité cosinus entre des vecteurs

# Importations supplémentaires pour la recommandation et la prédiction
from sklearn.compose import ColumnTransformer  # Pour appliquer des transformations sur des colonnes spécifiques
from sklearn.impute import SimpleImputer  # Pour gérer les valeurs manquantes
from sklearn.preprocessing import StandardScaler, OrdinalEncoder  # Pour normaliser et encoder les données
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer  # Pour transformer le texte en vecteurs numériques
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # Pour évaluer la performance du modèle
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors  # Pour utiliser des modèles basés sur les k plus proches voisins
from sklearn.metrics.pairwise import cosine_similarity  # Pour calculer la similarité cosinus (importé à nouveau, mais nécessaire)

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()
# Vérifier que DATABASE_URL est défini, sinon afficher une erreur et stopper l'application
if os.getenv("DATABASE_URL") is None:
    st.error(
        "Erreur : La variable DATABASE_URL n'est pas définie dans le fichier .env.\n"
        "Veuillez ajouter une ligne comme :\n\nDATABASE_URL=postgresql://username:password@host:port/database_name"
    )
    st.stop()

# Récupérer l'URL de la base de données depuis la variable d'environnement
DATABASE_URL = os.getenv("DATABASE_URL")
st.write("URL de la base de données utilisée :", DATABASE_URL)

# Télécharger les ressources nécessaires de NLTK pour le stemming
nltk.download('wordnet')
nltk.download('omw-1.4')

# Définir le nom du fichier qui servira de cache local pour stocker le DataFrame extrait
CACHE_FILE = "data_cache.parquet"

# ---------------------------------------------------------------------
# PARTIE 1 : Extraction et Nettoyage des Données
# ---------------------------------------------------------------------
def get_extracted_features(line_number: int = 200) -> pd.DataFrame:
    """
    Extrait les données depuis la base de données en exécutant plusieurs requêtes SQL,
    fusionne les résultats et enregistre le DataFrame final dans un fichier de cache local.
    Si le cache existe, il est utilisé pour accélérer le chargement.
    """
    if os.path.exists(CACHE_FILE):
        st.write("Chargement des données depuis le cache local...")
        df_final = pd.read_parquet(CACHE_FILE)
        return df_final

    st.write("Extraction des données depuis la base de données...")
    engine = create_engine(DATABASE_URL)  # Crée une connexion à la base de données

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
    # Fusion sur la colonne 'tconst'
    df_merge = df.merge(df_episode, on="tconst", how="left")
    # Fusion avec la table 'title_episode' sur la colonne 'parenttconst'
    df_merge_parent = df.merge(df_episode, left_on="tconst", right_on="parenttconst", how="left", suffixes=('', '_parent'))
    # Concaténation des deux fusions
    df = pd.concat([df_merge, df_merge_parent], ignore_index=True)
    # Agrégation par 'tconst' pour obtenir une seule ligne par film
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
    # Fusionner le résultat avec le DataFrame principal
    df = df.merge(df_akas, on="tconst", how="left")
    # Fonction pour nettoyer la colonne 'regionlist'
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
    # Transformation de la colonne 'primaryname' en liste par 'category'
    df_principals = df_principals.groupby(['tconst', 'category'])['primaryname'].agg(list).unstack(fill_value=[]).reset_index()
    # S'assurer que toutes les colonnes attendues existent
    for col in ["actor", "self", "producer", "actress", "director"]:
        if col not in df_principals.columns:
            df_principals[col] = [[] for _ in range(len(df_principals))]
    df_principals = df_principals[["tconst", "actor", "self", "producer", "actress", "director"]]
    # Fusionner avec le DataFrame principal
    df_final = pd.merge(df, df_principals, on='tconst', how='left')
    st.write("Table 4/4 fusionnée.")

    # Sauvegarde du DataFrame final dans un fichier Parquet pour le cache
    df_final.to_parquet(CACHE_FILE)
    st.write("Données extraites et sauvegardées dans le cache local.")
    return df_final

def clean_list(list_str) -> list:
    """
    Convertit une chaîne représentant une liste en une liste Python.
    Exemple : "['A', 'B', 'C']" -> ['A', 'B', 'C']
    """
    if isinstance(list_str, (list, np.ndarray)):
        return list(list_str)
    if list_str is None:
        return []
    # Enlève les crochets et les guillemets
    s = str(list_str).strip("[]").replace("'", "").replace('"', "")
    if s == "":
        return []
    # Sépare la chaîne en éléments et enlève les espaces superflus
    return [item.strip() for item in s.split(",") if item.strip()]

def clean_region_list(list_str) -> list:
    """
    Nettoie une chaîne représentant une liste de régions en supprimant les valeurs indésirables.
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
    Applique un nettoyage complet au DataFrame :
      - Convertit 'numvotes' et 'averagerating' en numérique et gère les valeurs manquantes.
      - Calcule le score pondéré (weighted_score) en fonction de 'numvotes' et 'averagerating'.
      - Applique des opérations de nettoyage sur les colonnes textuelles et les transforme en listes.
    """
    # Conversion des colonnes en numérique et gestion des valeurs manquantes
    df['numvotes'] = pd.to_numeric(df['numvotes'].replace("\\N", np.nan), errors='coerce')
    df['averagerating'] = pd.to_numeric(df['averagerating'].replace("\\N", np.nan), errors='coerce')
    df['numvotes'].fillna(0, inplace=True)
    df['averagerating'].fillna(df['averagerating'].mean(), inplace=True)
    
    # Calcul du score pondéré : C est la moyenne globale de 'averagerating', m est un seuil fixé (ici 1000)
    C = df['averagerating'].mean()
    m = 1000
    def weighted_rating(x, m=m, C=C):
        v = x['numvotes']
        R = x['averagerating']
        # Calcule le score pondéré en combinant R et la moyenne globale C
        return (v / (v + m) * R) + (m / (v + m) * C)
    df['weighted_score'] = df.apply(weighted_rating, axis=1)
    
    # Fonction de stemming : réduit chaque mot à son radical
    def stemming(text: str) -> str:
        ps = PorterStemmer()
        tokens = text.split()
        stemmed_tokens = [ps.stem(token) for token in tokens]
        return " ".join(stemmed_tokens)
    
    # Fonction pour nettoyer le texte : met en minuscule, retire les caractères spéciaux et espaces superflus
    def clean_text(text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s\[\]]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    # Nettoyage du titre principal avec nettoyage du texte et stemming
    df['primarytitle'] = df['primarytitle'].apply(lambda x: stemming(clean_text(x)))
    # Transformation de la colonne 'genres' en liste
    df['genres'] = df['genres'].astype(str).apply(lambda x: x.split(','))
    
    # Pour chaque colonne de type liste, on applique le nettoyage et on ne garde que les 10 valeurs les plus fréquentes
    for col in ["actor", "actress", "self", "producer", "director"]:
        df[col] = df[col].apply(clean_list)
        counts = df[col].explode().value_counts()
        top_names = counts.head(10).index.tolist()  # On garde les 10 plus fréquents
        df[col] = df[col].apply(lambda names: [name for name in names if name in top_names])
    df["regionlist"] = df["regionlist"].apply(clean_region_list)
    
    return df

# On définit Featurescleaning comme alias de clean_data pour la cohérence avec d'autres modules
def Featurescleaning(df: pd.DataFrame) -> pd.DataFrame:
    return clean_data(df)

# ---------------------------------------------------------------------
# PARTIE 2 : Visualisation
# ---------------------------------------------------------------------
def visualize_data(df: pd.DataFrame):
    """
    Affiche des graphiques pour visualiser les 10 éléments les plus fréquents
    dans les colonnes 'actor', 'actress', 'director' et 'genres'.
    """
    st.subheader("Visualisation des données")
    
    # Visualisation des acteurs
    st.write("Top Acteurs")
    if "actor" in df.columns:
        actor_counts = df["actor"].explode().value_counts().head(10)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=actor_counts.values, y=actor_counts.index)
        plt.title("Top 10 Acteurs")
        st.pyplot(plt)
        plt.clf()
    
    # Visualisation des actrices
    st.write("Top Actrices")
    if "actress" in df.columns:
        actress_counts = df["actress"].explode().value_counts().head(10)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=actress_counts.values, y=actress_counts.index)
        plt.title("Top 10 Actrices")
        st.pyplot(plt)
        plt.clf()
    
    # Visualisation des réalisateurs
    st.write("Top Réalisateurs")
    if "director" in df.columns:
        director_counts = df["director"].explode().value_counts().head(10)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=director_counts.values, y=director_counts.index)
        plt.title("Top 10 Réalisateurs")
        st.pyplot(plt)
        plt.clf()
    
    # Visualisation des genres
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
    Recommande des films similaires au film donné en se basant sur la similarité cosinus.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données du film (non nettoyé initialement) avec des colonnes telles que
                           'startyear', 'seasonnumber', 'episodenumber', 'titletype', 'genres', 'actor', 'actress' et 'primarytitle'.
        film (str): Le titre du film pour lequel on souhaite obtenir des recommandations.
        
    Returns:
        pd.DataFrame: Un DataFrame contenant les films recommandés.
    """
    # Nettoyage des données via Featurescleaning
    df = Featurescleaning(df)

    # Sélection des features pertinentes pour la recommandation
    features = df.drop(columns=["averagerating", "numvotes", "weighted_score", "tconst", "primarytitle", "self", "director", "producer"])
    # Renommage de la colonne 'self' en 'selfperson' pour éviter les conflits
    features.rename(columns={"self": "selfperson"}, inplace=True)
    # Cible : weighted_score (bien que non utilisée dans ce cas)
    target = df[["weighted_score"]].fillna(0)
    # Division du DataFrame (même si cette division n'est pas utilisée dans ce cas)
    features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Définition des pipelines pour chaque colonne
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

    # Création d'un ColumnTransformer pour appliquer les transformations aux colonnes correspondantes
    preprocessing = ColumnTransformer(transformers=[
        ("year", yearPipeline, ["startyear"]),
        ("seasonEpisodeNumber", seasonEpisodeNumberPipeline, ["seasonnumber", "episodenumber"]),
        ("titletype", titletypePipeline, ["titletype"]),
        ("genres", genresPipeline, "genres"),
        ("actor", actorPipeline, "actor"),
        ("actress", actressPipeline, "actress")
    ])

    # Pipeline complet de préparation des données
    modelPipeReco = Pipeline(steps=[("preparation", preprocessing)])
    
    # Transformation des features et calcul de la similarité cosinus
    features_transformed = modelPipeReco.fit_transform(features)
    cosine_sim = cosine_similarity(features_transformed)

    # Vérifier que le film est présent dans la colonne 'primarytitle'
    if film not in df["primarytitle"].values:
        raise ValueError(f"Le film '{film}' n'est pas présent dans la colonne 'primarytitle'.")
    # Trouver l'indice du film dans le DataFrame
    index = df[df["primarytitle"] == film].index[0]
    cosine_similarities = cosine_sim[index]
    # Sélectionner les 5 films les plus similaires (excluant le film de référence)
    similar_indices = cosine_similarities.argsort()[::-1][0:6]
    similar_indices = [i for i in similar_indices if i != index]
    similar_movies = df.iloc[similar_indices]
    
    return similar_movies

# ---------------------------------------------------------------------
# PARTIE 4 : Estimation de la Popularité par Critères
# ---------------------------------------------------------------------
class PopularityPrediction():
    """
    Classe pour prédire la popularité d'un film (score pondéré) en fonction de critères saisis.
    """
    
    def __init__(self) -> None:
        # Pipeline pour la colonne 'startyear'
        self.yearPipeline = Pipeline(steps=[
            ("to_float", FunctionTransformer(lambda X: X.astype(np.float64))),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        # Pipeline pour 'seasonnumber' et 'episodenumber'
        self.seasonEpisodeNumberPipeline = Pipeline(steps=[
            ("to_float", FunctionTransformer(lambda X: X.astype(np.float64))),
            ("imputer", SimpleImputer(strategy="constant", fill_value=1)),
            ("scaler", StandardScaler())
        ])
        # Pipeline pour transformer 'primarytitle' en vecteurs via TF-IDF
        self.primarytitlePipeline = Pipeline(steps=[("tfidf", TfidfVectorizer(stop_words="english"))])
        # Pipeline pour encoder 'titletype'
        self.titletypePipeline = Pipeline(steps=[("encoder", OrdinalEncoder())])
        # Pipeline pour transformer 'genres' en vecteurs
        self.genresPipeline = Pipeline(steps=[
            ('binarizer', CountVectorizer(analyzer=lambda x: set(x)))
        ])
        # Pipeline pour 'actor'
        self.actorPipeline = Pipeline(steps=[
            ('binarizer', CountVectorizer(analyzer=lambda x: set(x)))
        ])
        # Pipeline pour 'actress'
        self.actressPipeline = Pipeline(steps=[
            ('binarizer', CountVectorizer(analyzer=lambda x: set(x)))
        ])
        # Pipeline pour 'director'
        self.directorPipeline = Pipeline(steps=[
            ('binarizer', CountVectorizer(analyzer=lambda x: set(x)))
        ])
        # Création d'un ColumnTransformer qui applique chaque pipeline aux colonnes correspondantes
        self.preprocessing = ColumnTransformer(transformers=[
            ("year", self.yearPipeline, ["startyear"]),
            ("seasonEpisodeNumber", self.seasonEpisodeNumberPipeline, ["seasonnumber", "episodenumber"]),
            ("titletype", self.titletypePipeline, ["titletype"]),
            ("genres", self.genresPipeline, "genres"),
            ("actor", self.actorPipeline, "actor"),
            ("actress", self.actressPipeline, "actress"),
            ("director", self.directorPipeline, "director")
        ])
        # Pipeline complet qui combine le prétraitement et le modèle de régression (KNeighborsRegressor)
        self.modelPipe = Pipeline(steps=[
            ("prep données", self.preprocessing),
            ("model", KNeighborsRegressor())
        ])
    
    def fit(self, df):
        """
        Entraîne le modèle sur le DataFrame fourni.
        On retire certaines colonnes non pertinentes, renomme 'self' en 'selfperson', et définit la cible (weighted_score).
        """
        self.df = df
        self.features = df.drop(columns=["averagerating", "numvotes", "weighted_score", "tconst", "primarytitle", "self", "producer"])
        # Renommer la colonne 'self' en 'selfperson' pour éviter des conflits dans le modèle
        self.features.rename(columns={"self": "selfperson"}, inplace=True)
        self.target = df[["weighted_score"]].fillna(0)
        # Diviser le DataFrame en ensembles d'entraînement et de test
        self.features_train, self.features_test, self.target_train, self.target_test = train_test_split(
            self.features, self.target, test_size=0.2, random_state=42)
        # Entraîner le modèle
        self.modelPipe.fit(self.features_train, self.target_train)
        
    def predict(self, features):
        """
        Retourne la prédiction du modèle pour les features fournies.
        """
        return self.modelPipe.predict(features)
    
    def evaluate(self):
        """
        Évalue le modèle sur l'ensemble de test et renvoie les métriques MSE, MAE et R².
        """
        y_pred = self.modelPipe.predict(self.features_test)
        mse = mean_squared_error(self.target_test, y_pred)
        mae = mean_absolute_error(self.target_test, y_pred)
        r2 = r2_score(self.target_test, y_pred)
        return mse, mae, r2

# ---------------------------------------------------------------------
# INTERFACE STREAMLIT
# ---------------------------------------------------------------------
def main():
    st.title("Application de Visualisation, Recommandation et Estimation de la Popularité par Critères")
    
    # Bouton pour actualiser la base de données (supprime le cache local)
    if st.sidebar.button("Actualiser la BD"):
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
            st.sidebar.success("Cache supprimé, la BD sera réextrait.")
    
    # Extraction et nettoyage des données
    df = get_extracted_features(line_number=200)
    df = clean_data(df)
    
    # Menu principal de l'application
    menu = [
        "Visualisation",
        "Recommandation",
        "Prédiction de la Popularité (par film)",
        "Estimation de la Popularité par Critères"
    ]
    choice = st.sidebar.selectbox("Menu principal", menu)
    
    # Option : Visualisation
    if choice == "Visualisation":
        st.write("Aperçu des données nettoyées :")
        st.dataframe(df.head())
        visualize_data(df)
    
    # Option : Recommandation
    elif choice == "Recommandation":
        film = st.text_input("Entrez le titre du film pour lequel vous souhaitez une recommandation")
        if st.button("Obtenir des recommandations"):
            try:
                similar_movies = RecommandationSystem(df, film)
                st.write("Films recommandés :")
                st.dataframe(similar_movies[["primarytitle", "startyear", "genres"]])
            except ValueError as e:
                st.error(str(e))
    
    # Option : Prédiction de la Popularité (par film)
    elif choice == "Prédiction de la Popularité (par film)":
        st.subheader("Prédiction de la Popularité (par film)")
        model = PopularityPrediction()
        with st.spinner("Entraînement du modèle..."):
            model.fit(df)
        mse, mae, r2 = model.evaluate()
        st.write(f"**MSE** : {mse:.4f}")
        st.write(f"**MAE** : {mae:.4f}")
        st.write(f"**R2**  : {r2:.4f}")
        
        film_choice = st.selectbox("Sélectionnez un film pour prédire sa popularité", df["primarytitle"].tolist())
        if st.button("Prédire la popularité (par film)"):
            selected_row = df[df["primarytitle"] == film_choice]
            prediction = model.predict(selected_row)
            st.write("Prédiction de popularité :", prediction[0])
    
    # Option : Estimation de la Popularité par Critères
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
            # Construction de la ligne d'entrée en fonction des critères sélectionnés
            input_data = pd.DataFrame({
                "startyear": [year_val if year_val is not None else int(df["startyear"].median())],
                "seasonnumber": [1.0],
                "episodenumber": [1.0],
                "titletype": [df["titletype"].mode()[0]],  # Utilise la valeur la plus fréquente
                "genres": [[genre_choice] if genre_choice is not None else []],
                "actor": [[actor_choice] if actor_choice is not None else []],
                "actress": [[]],  # Laisse vide pour cet exemple
                "director": [[director_choice] if director_choice is not None else []]
            })
            
            model = PopularityPrediction()
            with st.spinner("Entraînement du modèle..."):
                model.fit(df)
            # Conversion explicite pour s'assurer que les colonnes numériques sont bien en float
            input_data["seasonnumber"] = input_data["seasonnumber"].astype(float)
            input_data["episodenumber"] = input_data["episodenumber"].astype(float)
            # Préparation d'une copie pour diagnostic
            tab = input_data.copy()
            tab["actor"] = tab["actor"].astype(str)
            tab["director"] = tab["director"].astype(str)
            tab["actress"] = tab["actress"].astype(str)
            st.write("Données d'entrée pour la prédiction :")
            st.dataframe(tab)
            # Prédiction
            prediction = model.predict(input_data)
            st.write("Estimation de la popularité (par critères) :", prediction[0])
            st.write("Types des colonnes d'entrée :", input_data.dtypes)
            print(prediction)

if __name__ == "__main__":
    main()
