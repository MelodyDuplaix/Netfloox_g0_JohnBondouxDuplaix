import pandas as pd
import nltk
from nltk import PorterStemmer
import re
nltk.download('wordnet')
nltk.download('wordnet')


def stemming(liste):
  stemming = []
  for element in liste:
    elementStemme = PorterStemmer().stem(element)
    stemming.append(elementStemme)
  return "".join(stemming)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s\[\]]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def clean_list(list_str):
    if list_str == "[]":
        return []
    if type(list_str) == float:
        return []
    list_str = list_str.strip("[]").replace("'", "").replace('"', "")
    list = list_str.split(", ")
    return [individu for individu in list if individu]

def clean_region_list(list_str):
    if list_str == "[]":
        return []
    if type(list_str) == float:
        return []
    list_str = list_str.strip("[]").replace("'", "").replace('"', "").replace("\\\\N", "")
    list = list_str.split(", ")
    return [individu for individu in list if individu]

def Featurescleaning(df):
    
    # Paramètres pour le calcul du score pondéré
    C = df['averagerating'].mean()  # Score moyen de tous les films
    m = 1000  # Nombre minimum de votes requis pour être pris en compte

    # Calcul du score pondéré
    def weighted_rating(x, m=m, C=C):
        v = x['numvotes'] if pd.notnull(x['numvotes']) else 1
        R = x['averagerating'] if pd.notnull(x['averagerating']) else 0
        return (v / (v + m) * R) + (m / (v + m) * C)

    df['weighted_score'] = df.apply(weighted_rating, axis=1)

    
    #Retirer caractères spéciaux, espaces et retourner en minuscule
    # df['primarytitle'] = df['primarytitle'].apply(clean_text)
    #Stemming
    # df['primarytitle'] = df['primarytitle'].apply(stemming)
    
    
    df['genres'] = df['genres'].astype(str)
    df['genres'].apply(lambda x: x.split(','))
    
    
    df["actor"] = df["actor"].apply(clean_list)
    actor_counts = df["actor"].explode().value_counts()
    top_actors = actor_counts.head(500).index.tolist()
    df["actor"] = df['actor'].apply(lambda actor_list: [actor for actor in actor_list if actor in top_actors])
    
    df["actress"] = df["actress"].apply(clean_list)
    actress_counts = df["actress"].explode().value_counts()
    top_actress = actress_counts.head(500).index.tolist()
    df["actress"] = df['actress'].apply(lambda actress_list: [actress for actress in actress_list if actress in top_actress])
    
    df["self"] = df["self"].apply(clean_list)
    self_counts = df["self"].explode().value_counts()
    top_self = self_counts.head(500).index.tolist()
    df["self"] = df['self'].apply(lambda self_list: [self for self in self_list if self in top_self])
    
    df["producer"] = df["producer"].apply(clean_list)
    producer_counts = df["producer"].explode().value_counts()
    top_producer = producer_counts.head(500).index.tolist()
    df["producer"] = df['producer'].apply(lambda producer_list: [producer for producer in producer_list if producer in top_producer])
    
    df["director"] = df["director"].apply(clean_list)
    director_counts = df["director"].explode().value_counts()
    top_director = director_counts.head(500).index.tolist()
    df["director"] = df['director'].apply(lambda director_list: [director for director in director_list if director in top_director])
    
    df["regionlist"] = df["regionlist"].apply(clean_region_list)
    
    return df