# Importation des modules nécessaires pour le fonctionnement de l'application
import os                         # Module pour interagir avec le système d'exploitation
import sys                        # Module pour interagir avec des variables et fonctions du système Python
sys.path.append('..')             # Ajoute le répertoire parent au chemin de recherche des modules
import pandas as pd               # Bibliothèque pour la manipulation de données sous forme de DataFrame
import streamlit as st            # Bibliothèque pour créer des applications web interactives
import matplotlib.pyplot as plt   # Bibliothèque pour créer des graphiques
import seaborn as sns             # Bibliothèque pour des visualisations statistiques plus avancées
import numpy as np                # Bibliothèque pour les calculs numériques et manipulations de tableaux
import re                         # Module pour les expressions régulières
from sqlalchemy import create_engine  # Pour créer une connexion à une base de données SQL
from dotenv import load_dotenv         # Pour charger les variables d'environnement depuis un fichier .env
from nltk import PorterStemmer         # Pour utiliser l'algorithme de racinisation (stemming)
import nltk                            # Bibliothèque de traitement du langage naturel
from sklearn.pipeline import Pipeline  # Pour créer des chaînes de transformations sur les données
from sklearn.preprocessing import FunctionTransformer  # Pour créer des transformateurs personnalisés
from sklearn.ensemble import RandomForestRegressor      # Algorithme de régression (ici non utilisé directement)
from sklearn.model_selection import train_test_split    # Pour diviser les données en ensembles d'entraînement et de test

# Importations supplémentaires pour le système de recommandation
from sklearn.compose import ColumnTransformer      # Pour appliquer différentes transformations sur des colonnes spécifiques
from sklearn.impute import SimpleImputer             # Pour gérer les valeurs manquantes
from sklearn.preprocessing import StandardScaler, OrdinalEncoder  # Pour normaliser les données et encoder des variables catégorielles
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer  # Pour transformer du texte en vecteurs
from sklearn.metrics.pairwise import cosine_similarity  # Pour calculer la similarité cosinus entre des vecteurs

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()
# Vérifie que la variable DATABASE_URL est définie, sinon affiche une erreur et arrête l'application Streamlit
if os.getenv("DATABASE_URL") is None:
    st.error(
        "Erreur : La variable DATABASE_URL n'est pas définie dans le fichier .env.\n"
        "Veuillez ajouter une ligne comme :\n\nDATABASE_URL=postgresql://username:password@host:port/database_name"
    )
    st.stop()

# Téléchargement des ressources NLTK nécessaires (à exécuter une seule fois)
nltk.download('wordnet')
nltk.download('omw-1.4')

# Définition du nom du fichier de cache local pour stocker les données extraites (format Parquet)
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
    # Si le fichier de cache existe, on charge directement les données pour accélérer l'exécution
    if os.path.exists(CACHE_FILE):
        st.write("Chargement des données depuis le cache local...")
        df_final = pd.read_parquet(CACHE_FILE)
        return df_final

    st.write("Extraction des données depuis la base de données...")
    # Récupération de l'URL de la base de données à partir des variables d'environnement
    database_url = os.getenv("DATABASE_URL")
    if database_url is None:
        raise ValueError("DATABASE_URL n'est pas défini dans le fichier .env.")
    # Création du moteur de connexion à la base de données via SQLAlchemy
    engine = create_engine(database_url)

    # Requête SQL pour extraire les données de la table 'title_basics'
    query = f"""
    SELECT tconst, titletype, primarytitle, isadult, startyear, genres, averagerating, numvotes 
    FROM sebastien.title_basics 
    WHERE startyear IS NOT NULL 
    ORDER BY startyear DESC 
    LIMIT {line_number};
    """
    df = pd.read_sql_query(query, engine)  # Exécute la requête et stocke les résultats dans un DataFrame
    st.write("Table 1/4 chargée.")

    # Requête pour extraire la table 'title_episode' et fusionner les données avec 'title_basics'
    query = "SELECT * FROM sebastien.title_episode;"
    df_episode = pd.read_sql_query(query, engine)  # Charge les informations d'épisodes
    df_merge = df.merge(df_episode, on="tconst", how="left")  # Fusionne les épisodes liés directement au film
    df_merge_parent = df.merge(df_episode, left_on="tconst", right_on="parenttconst", how="left", suffixes=('', '_parent'))
    # Concatène les deux DataFrames pour avoir une vue complète des épisodes
    df = pd.concat([df_merge, df_merge_parent], ignore_index=True)
    # Agrège les données par 'tconst' en prenant la première valeur pour certaines colonnes et la valeur max pour d'autres
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

    # Requête pour extraire les informations de la table 'title_akas' (concernant les régions)
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
    df_akas = pd.read_sql_query(query_title_akas, engine)  # Exécute la requête pour obtenir les régions
    # Fusionne les données de régions avec le DataFrame principal
    df = df.merge(df_akas, on="tconst", how="left")
    # Fonction locale pour nettoyer la liste des régions en retirant des valeurs indésirables
    def replace_and_filter(region_list):
        if isinstance(region_list, (list, np.ndarray)):
            return [region for region in region_list if region != '\\N' and region != '']
        if pd.isnull(region_list):
            return []
        return region_list
    # Applique la fonction de nettoyage sur la colonne 'regionlist' du DataFrame df_akas
    df_akas['regionlist'] = df_akas['regionlist'].apply(replace_and_filter)
    st.write("Table 3/4 chargée.")

    # Requête pour extraire les informations de la table 'title_principals' (acteurs, réalisateurs, etc.)
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
    df_principals = pd.read_sql_query(query_title_principals, engine)  # Charge les données des personnes liées aux films
    # Agrège les noms par film et par catégorie en créant une liste de noms pour chaque combinaison
    df_principals = df_principals.groupby(['tconst', 'category'])['primaryname'].agg(list).unstack(fill_value=[]).reset_index()
    # S'assure que les colonnes importantes existent même si elles ne sont pas présentes dans le résultat
    for col in ["actor", "self", "producer", "actress", "director"]:
        if col not in df_principals.columns:
            df_principals[col] = [[] for _ in range(len(df_principals))]
    # Réorganise le DataFrame pour ne garder que les colonnes souhaitées
    df_principals = df_principals[["tconst", "actor", "self", "producer", "actress", "director"]]
    # Fusionne les informations de la table 'title_principals' avec le DataFrame principal
    df_final = pd.merge(df, df_principals, on='tconst', how='left')
    st.write("Table 4/4 fusionnée.")

    # Sauvegarde du DataFrame final dans un fichier Parquet pour un accès plus rapide lors des exécutions suivantes
    df_final.to_parquet(CACHE_FILE)
    st.write("Données extraites et sauvegardées dans le cache local.")
    return df_final  # Retourne le DataFrame final contenant toutes les données fusionnées

def clean_list(list_str) -> list:
    """
    Convertit une chaîne représentant une liste en une vraie liste.
    """
    # Si la variable est déjà une liste ou un tableau numpy, la convertir en liste Python
    if isinstance(list_str, (list, np.ndarray)):
        return list(list_str)
    # Si la valeur est None, retourne une liste vide
    if list_str is None:
        return []
    # Nettoie la chaîne en retirant les crochets et les guillemets
    s = str(list_str).strip("[]").replace("'", "").replace('"', "")
    if s == "":
        return []
    # Sépare la chaîne par la virgule et retourne la liste des éléments nettoyés
    return [item.strip() for item in s.split(",") if item.strip()]

def clean_region_list(list_str) -> list:
    """
    Nettoie une liste de régions en supprimant les valeurs indésirables.
    """
    # Même logique que clean_list, en supprimant également les occurrences de '\\N'
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
    # Conversion des colonnes 'numvotes' et 'averagerating' en valeurs numériques en remplaçant les valeurs indésirables
    df['numvotes'] = pd.to_numeric(df['numvotes'].replace("\\N", np.nan), errors='coerce')
    df['averagerating'] = pd.to_numeric(df['averagerating'].replace("\\N", np.nan), errors='coerce')
    # Remplit les valeurs manquantes : pour 'numvotes' par 0 et pour 'averagerating' par la moyenne
    df['numvotes'].fillna(0, inplace=True)
    df['averagerating'].fillna(df['averagerating'].mean(), inplace=True)
    
    # Calcul d'une moyenne globale de 'averagerating' pour le calcul du score pondéré
    C = df['averagerating'].mean()
    m = 1000  # Seuil minimum de votes pour influencer le score
    # Fonction pour calculer le score pondéré en tenant compte du nombre de votes et de la note moyenne
    def weighted_rating(x, m=m, C=C):
        v = x['numvotes']
        R = x['averagerating']
        return (v / (v + m) * R) + (m / (v + m) * C)
    # Applique la fonction à chaque ligne pour créer une nouvelle colonne 'weighted_score'
    df['weighted_score'] = df.apply(weighted_rating, axis=1)

    # Fonction de racinisation (stemming) pour réduire les mots à leur forme racine
    def stemming(text: str) -> str:
        ps = PorterStemmer()
        tokens = text.split()  # Découpe le texte en mots
        stemmed_tokens = [ps.stem(token) for token in tokens]  # Applique le stemming à chaque mot
        return " ".join(stemmed_tokens)  # Recompose la phrase avec les mots racinisés

    # Fonction pour nettoyer le texte en le mettant en minuscule et en supprimant les caractères non désirés
    def clean_text(text: str) -> str:
        text = text.lower()  # Convertit le texte en minuscules
        text = re.sub(r'[^a-z0-9\s\[\]]', '', text)  # Supprime les caractères spéciaux, ne garde que lettres, chiffres, espaces et crochets
        text = re.sub(r'\s+', ' ', text)  # Remplace les espaces multiples par un seul espace
        return text.strip()  # Retire les espaces en début et fin de chaîne

    # Applique le nettoyage et le stemming à la colonne 'primarytitle'
    df['primarytitle'] = df['primarytitle'].apply(lambda x: stemming(clean_text(x)))
    # Transforme la colonne 'genres' en une liste de genres en se basant sur la séparation par la virgule
    df['genres'] = df['genres'].astype(str).apply(lambda x: x.split(','))

    # Pour les colonnes liées aux personnes (acteurs, réalisateurs, etc.), nettoie la liste et conserve les noms les plus fréquents
    for col in ["actor", "actress", "self", "producer", "director"]:
        df[col] = df[col].apply(clean_list)  # Nettoie la chaîne pour obtenir une liste
        counts = df[col].explode().value_counts()  # Compte le nombre d'apparitions de chaque nom
        top_names = counts.head(10).index.tolist()  # Garde uniquement les 10 noms les plus fréquents
        # Filtre la liste pour ne garder que les noms présents dans top_names
        df[col] = df[col].apply(lambda names: [name for name in names if name in top_names])
    # Nettoie la colonne 'regionlist' en appliquant la fonction dédiée
    df["regionlist"] = df["regionlist"].apply(clean_region_list)

    return df  # Retourne le DataFrame nettoyé

# Pour la fonction de recommandation, nous utilisons un alias pour clean_data
def Featurescleaning(df: pd.DataFrame) -> pd.DataFrame:
    return clean_data(df)

# ---------------------------------------------------------------------
# PARTIE 2 : Visualisation
# ---------------------------------------------------------------------

def visualize_data(df: pd.DataFrame):
    # Titre de la section dans l'application Streamlit
    st.subheader("Visualisation des données")
    st.write("Top Acteurs")
    # Visualisation des acteurs si la colonne 'actor' existe
    if "actor" in df.columns:
        # Compte les occurrences de chaque acteur et sélectionne les 10 plus fréquents
        actor_counts = df["actor"].explode().value_counts().head(10)
        plt.figure(figsize=(10, 6))  # Définit la taille de la figure
        sns.barplot(x=actor_counts.values, y=actor_counts.index)  # Crée un graphique à barres horizontal
        plt.title("Top 10 Acteurs")  # Définit le titre du graphique
        st.pyplot(plt)  # Affiche le graphique dans l'interface Streamlit
        plt.clf()  # Efface la figure pour préparer le prochain graphique
    st.write("Top Actrices")
    # Visualisation des actrices si la colonne 'actress' existe
    if "actress" in df.columns:
        actress_counts = df["actress"].explode().value_counts().head(10)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=actress_counts.values, y=actress_counts.index)
        plt.title("Top 10 Actrices")
        st.pyplot(plt)
        plt.clf()
    st.write("Top Réalisateurs")
    # Visualisation des réalisateurs si la colonne 'director' existe
    if "director" in df.columns:
        director_counts = df["director"].explode().value_counts().head(10)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=director_counts.values, y=director_counts.index)
        plt.title("Top 10 Réalisateurs")
        st.pyplot(plt)
        plt.clf()
    st.write("Top Genres")
    # Visualisation des genres en comptant leur occurrence dans la colonne 'genres'
    if "genres" in df.columns:
        # Crée une série de tous les genres présents dans le DataFrame
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
    # Nettoie les données en appliquant la fonction Featurescleaning (alias de clean_data)
    df = Featurescleaning(df)

    # Sélection des colonnes pertinentes pour la recommandation en supprimant celles non utiles
    features = df.drop(columns=["averagerating", "numvotes", "weighted_score", "tconst", "primarytitle", "self", "director", "producer"])
    # Renomme la colonne "self" en "selfperson" pour éviter les conflits de nommage
    features.rename(columns={"self": "selfperson"}, inplace=True)

    # Pipeline pour transformer la colonne 'startyear' : imputation par la médiane et standardisation
    yearPipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    # Pipeline pour transformer les colonnes 'seasonnumber' et 'episodenumber' : imputation constante et standardisation
    seasonEpisodeNumberPipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value=1)),
        ("scaler", StandardScaler())
    ])
    # Pipeline pour encoder la colonne 'titletype' en valeurs ordinales
    titletypePipeline = Pipeline(steps=[
        ("encoder", OrdinalEncoder())
    ])
    # Pipeline pour vectoriser la liste des genres en utilisant CountVectorizer avec un analyseur personnalisé
    genresPipeline = Pipeline(steps=[
        ('binarizer', CountVectorizer(analyzer=lambda x: set(x)))
    ])
    # Pipeline pour vectoriser la liste des acteurs
    actorPipeline = Pipeline(steps=[
        ('binarizer', CountVectorizer(analyzer=lambda x: set(x)))
    ])
    # Pipeline pour vectoriser la liste des actrices
    actressPipeline = Pipeline(steps=[
        ('binarizer', CountVectorizer(analyzer=lambda x: set(x)))
    ])

    # Création d'un ColumnTransformer pour appliquer les différents pipelines sur leurs colonnes respectives
    preprocessing = ColumnTransformer(transformers=[
        ("year", yearPipeline, ["startyear"]),
        ("seasonEpisodeNumber", seasonEpisodeNumberPipeline, ["seasonnumber", "episodenumber"]),
        ("titletype", titletypePipeline, ["titletype"]),
        ("genres", genresPipeline, "genres"),
        ("actor", actorPipeline, "actor"),
        ("actress", actressPipeline, "actress")
    ])

    # Création d'un pipeline complet qui prépare les données pour le calcul de similarité
    modelPipeReco = Pipeline(steps=[
        ("preparation", preprocessing)
    ])

    # Transformation des features en appliquant le pipeline
    features_transformed = modelPipeReco.fit_transform(features)
    # Calcul de la similarité cosinus entre tous les films
    cosine_sim = cosine_similarity(features_transformed)

    # Vérifie que le film fourni est présent dans la colonne 'primarytitle'
    if film not in df["primarytitle"].values:
        raise ValueError(f"Le film '{film}' n'est pas présent dans la colonne 'primarytitle'.")
    # Récupère l'index du film dans le DataFrame
    index = df[df["primarytitle"] == film].index[0]
    # Récupère les similarités cosinus pour le film donné
    cosine_similarities = cosine_sim[index]
    # Trie les indices par ordre décroissant de similarité et sélectionne les 5 films les plus similaires (en excluant le film lui-même)
    similar_indices = cosine_similarities.argsort()[::-1][1:6]
    similar_movies = df.iloc[similar_indices]
    
    return similar_movies  # Retourne un DataFrame des films recommandés

# ---------------------------------------------------------------------
# MAIN (Interface Streamlit)
# ---------------------------------------------------------------------

def main():
    # Titre principal de l'application
    st.title("Application de Visualisation et de Recommandation de Films")
    # Extraction des données via la fonction définie (depuis la base ou le cache)
    df = get_extracted_features(line_number=200)
    # Nettoie les données pour s'assurer qu'elles sont prêtes pour la visualisation et la recommandation
    df = clean_data(df)
    
    # Création d'un menu dans la barre latérale pour choisir entre "Visualisation" et "Recommandation"
    menu = [
        "Visualisation",
        "Recommandation"
    ]
    choice = st.sidebar.selectbox("Menu principal", menu)

    # Si l'utilisateur choisit l'option "Visualisation"
    if choice == "Visualisation":
        st.write("Aperçu des données nettoyées :")
        st.dataframe(df.head())  # Affiche les premières lignes du DataFrame
        visualize_data(df)         # Affiche les graphiques de visualisation
    # Si l'utilisateur choisit l'option "Recommandation"
    elif choice == "Recommandation":
        # Demande à l'utilisateur d'entrer le titre du film pour lequel il souhaite une recommandation
        film = st.text_input("Entrez le titre du film pour lequel vous souhaitez une recommandation")
        if st.button("Obtenir des recommandations"):
            try:
                # Appelle le système de recommandation pour trouver les films similaires
                similar_movies = RecommandationSystem(df, film)
                st.write("Films recommandés :")
                # Affiche un tableau avec le titre, l'année de début et les genres des films recommandés
                st.dataframe(similar_movies[["primarytitle", "startyear", "genres"]])
            except ValueError as e:
                # Affiche une erreur dans l'interface Streamlit si le film n'est pas trouvé ou en cas d'autre problème
                st.error(str(e))

# Point d'entrée principal du script
if __name__ == "__main__":
    main()
