import pandas as pd
import sys
sys.path.append('..')
from scripts.Cleaning import Featurescleaning
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.metrics.pairwise import cosine_similarity



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
    similar_indices = cosine_similarities.argsort()[::-1][1:6]
    similar_movies = df.iloc[similar_indices]
    
    return similar_movies