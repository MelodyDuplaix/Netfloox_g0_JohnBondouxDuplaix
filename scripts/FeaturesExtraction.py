from sqlalchemy import create_engine
import pandas as pd
from dotenv import load_dotenv
import os

def get_extracted_features(line_number):
    """
    Récupère un certain nombre de ligne de la base de données et les traite pour en extraire les features
    
    Args:
        line_number (int): nombre de ligne à récupérer
        
    Returns:
        DataFrame: DataFrame contenant les features extrait
    """
    # Connexion à la base de données
    load_dotenv()
    database_url = os.getenv("DATABASE_URL") # url de la base de données stocké dans un fichier .env sous la variable DATABASE_URL
    engine = create_engine(database_url)  # type: ignore
    
    # Récupération des données de la table de base
    query = f"""
    SELECT tconst, titletype, primarytitle, isadult, startyear, genres, averagerating, numvotes 
    FROM sebastien.title_basics 
    WHERE startyear IS NOT NULL 
    ORDER BY startyear DESC 
    LIMIT {line_number};
    """
    with engine.connect() as conn, conn.begin():
        df = pd.read_sql_query(query, engine)
    print("table 1 / 4")
        
    # Table episode
    query = "SELECT * FROM sebastien.title_episode;"
    with engine.connect() as conn, conn.begin():
        df_episode = pd.read_sql_query(query, engine)
    df_merge = df.merge(df_episode, on="tconst", how="left")
    df_merge_parent = df.merge(df_episode, left_on="tconst", right_on="parenttconst", how="left", suffixes=('', '_parent'))
    df = pd.concat([df_merge, df_merge_parent], ignore_index=True)
    df = df.groupby('tconst').agg({
        'titletype': 'first', # on garde  le titletype de la première ligne
        'primarytitle': 'first',
        'isadult': 'first',
        'startyear': 'first',
        'genres': 'first',
        'averagerating': 'first',
        'numvotes': 'first',
        'seasonnumber': 'max', # on garde le max de seasonnumber, soit le non nul
        'episodenumber': 'max'
    }).reset_index()
    print("table 2 / 4")
    
    # table akas
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

    with engine.connect() as conn, conn.begin():
        df_akas = pd.read_sql_query(query_title_akas, conn)
    df = df.merge(df_akas, on="tconst", how="left")
    
    def replace_and_filter(region_list):
        if isinstance(region_list, list):
            return [region for region in region_list if region != '\\N' and region != '']
        return region_list
    
    df_akas['regionlist'] = df_akas['regionlist'].apply(replace_and_filter)
    print("table 3 / 4")
    
    # table principals
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
            ORDER BY startyear DESC LIMIT  {line_number});
    """

    with engine.connect() as conn, conn.begin():
        df_principals= pd.read_sql_query(query_title_principals, conn)
    df_principals = df_principals.groupby(['tconst', 'category'])['primaryname'].agg(list).unstack(fill_value=[]).reset_index()
    df_principals = df_principals[["tconst", "actor", "self", "producer", "actress", "director"]]
    df_final = pd.merge(df, df_principals, on='tconst', how='left')
    df_final.head()
    print("table 4 / 4")
    
    return df_final

    