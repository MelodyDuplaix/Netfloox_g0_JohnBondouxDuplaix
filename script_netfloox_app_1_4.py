import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

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
def load_data():
    basics = load_tsv(r"C:\Users\sbond\Desktop\Netfloox\1-Data\data\title.basics-10k.tsv")
    ratings = load_tsv(r"C:\Users\sbond\Desktop\Netfloox\1-Data\data\title.ratings-10k.tsv")
    principals = load_tsv(r"C:\Users\sbond\Desktop\Netfloox\1-Data\data\title.principals-10k.tsv")
    names = load_tsv(r"C:\Users\sbond\Desktop\Netfloox\1-Data\data\name.basics-10k.tsv")
    return basics, ratings, principals, names

# Étape 2 : Préparer et fusionner les données
def prepare_data():
    """Préparer et fusionner les données nécessaires."""
    basics, ratings, principals, names = load_data()

    if basics is None or ratings is None or principals is None or names is None:
        st.stop()

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

    df = pd.merge(basics, ratings, on='tconst', how='inner')
    principals = pd.merge(principals, names[['nconst', 'primaryName']], on='nconst', how='left')
    df = pd.merge(df, principals[['tconst', 'primaryName', 'category']], on='tconst', how='left')

    df = df.rename(columns={
        'primaryTitle': 'title',
        'averageRating': 'rating',
        'numVotes': 'votes',
        'genres': 'genre'
    })

    # Ne garder que le premier mot avant la virgule pour les genres
    df['genre'] = df['genre'].str.split(',').str[0]

    final_columns = ['tconst', 'title', 'genre', 'rating', 'votes', 'primaryName', 'category']
    for col in final_columns:
        if col not in df.columns:
            st.error(f"La colonne requise '{col}' est absente après la fusion.")
            st.stop()

    return df[final_columns]

# Analyse des données
def analyze_data(df):
    st.subheader("Analyse des données")

    # Répartition des genres
    st.write("### Répartition des genres")
    genre_counts = df['genre'].value_counts().head(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=genre_counts.values, y=genre_counts.index, orient='h')
    plt.title("Top 10 des genres les plus fréquents")
    plt.xlabel("Nombre de films")
    plt.ylabel("Genres")
    st.pyplot(plt)

    # Répartition des notes moyennes
    st.write("### Répartition des notes moyennes")
    plt.figure(figsize=(10, 6))
    sns.histplot(df['rating'], bins=20, kde=True)
    plt.title("Répartition des notes moyennes")
    plt.xlabel("Note moyenne")
    plt.ylabel("Nombre de films")
    st.pyplot(plt)

    # Top 10 des acteurs/actrices
    st.write("### Top 10 des acteurs/actrices les plus actifs")
    actors = df[df['category'].str.contains('actor|actress', na=False)]['primaryName'].value_counts().head(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=actors.values, y=actors.index, orient='h')
    plt.title("Top 10 des acteurs/actrices les plus actifs")
    plt.xlabel("Nombre de films")
    plt.ylabel("Acteurs/Actrices")
    st.pyplot(plt)

    # Top 10 des réalisateurs
    st.write("### Top 10 des réalisateurs les plus actifs")
    directors = df[df['category'] == 'director']['primaryName'].value_counts().head(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=directors.values, y=directors.index, orient='h')
    plt.title("Top 10 des réalisateurs les plus actifs")
    plt.xlabel("Nombre de films")
    plt.ylabel("Réalisateurs")
    st.pyplot(plt)

# Recommandations de films
def recommend_movies(df, movie_title):
    st.subheader("Recommandations de films")

    movie = df[df['title'].str.contains(movie_title, case=False, na=False)]
    if movie.empty:
        st.warning("Aucun film trouvé avec ce titre.")
        return None

    genre = movie.iloc[0]['genre']
    recommendations = df[(df['genre'] == genre) & (~df['title'].str.contains(movie_title, case=False, na=False))]
    recommendations = recommendations.sort_values(by='rating', ascending=False).drop_duplicates(subset=['title']).head(5)

    if recommendations.empty:
        st.warning("Aucune recommandation trouvée.")
    else:
        st.write("### Films recommandés")
        st.dataframe(recommendations[['title', 'genre', 'rating', 'votes']])

# Nouvelle fonction pour calculer la popularité basée sur des critères augmentés
def calculate_popularity(df, genre=None, rating_threshold=None, vote_threshold=None, actor=None, director=None):
    filtered_data = df

    # Appliquer les filtres séquentiellement
    if genre:
        filtered_data = filtered_data[filtered_data['genre'] == genre]
    if rating_threshold:
        filtered_data = filtered_data[filtered_data['rating'] >= rating_threshold]
    if vote_threshold:
        filtered_data = filtered_data[filtered_data['votes'] >= vote_threshold]
    if actor:
        filtered_data = filtered_data[(filtered_data['category'].str.contains('actor|actress', na=False)) &
                                       (filtered_data['primaryName'] == actor)]
    if director:
        filtered_data = filtered_data[(filtered_data['category'] == 'director') &
                                       (filtered_data['primaryName'] == director)]

    # Vérifier si les données filtrées sont vides et afficher les diagnostics
    if filtered_data.empty:
        st.warning("Aucun film trouvé avec ces critères.")
        st.write("Données originales disponibles :", df.head())
        st.write("Données après filtrage :", filtered_data)
        return None

    # Calculer la popularité
    average_rating = filtered_data['rating'].mean()
    popularity_score = max(min(average_rating, 10), 0)  # Normaliser entre 0 et 10

    return popularity_score

# Interface principale pour la popularité
def main_popularity_estimation():
    st.subheader("Estimation de la popularité des films")
    df = prepare_data()

    st.write("### Entrez les caractéristiques du film :")
    genre = st.selectbox("Genre (facultatif)", options=[None] + list(df['genre'].unique()), key="unique_genre_input")
    rating_threshold = st.slider("Note minimale (facultatif)", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
    vote_threshold = st.slider("Nombre minimum de votes (facultatif)", min_value=0, max_value=int(df['votes'].max()), value=100)
    actor = st.selectbox("Acteur/Actrice (facultatif)", options=[None] + list(df[df['category'].str.contains('actor|actress', na=False)]['primaryName'].unique()), key="unique_actor_input")
    director = st.selectbox("Réalisateur (facultatif)", options=[None] + list(df[df['category'] == 'director']['primaryName'].unique()), key="unique_director_input")

    if st.button("Estimer la popularité", key="unique_predict_button"):
        popularity = calculate_popularity(df, genre=genre, rating_threshold=rating_threshold,
                                          vote_threshold=vote_threshold, actor=actor, director=director)
        if popularity is not None:
            st.success(f"La popularité estimée pour les critères sélectionnés est de : {popularity:.2f} / 10")
        else:
            st.error("Impossible d'estimer la popularité. Vérifiez les critères sélectionnés et les données disponibles.")

# Interface principale pour les recommandations
def main_recommendations():
    st.subheader("Recommandations de films")
    df = prepare_data()

    movie_title = st.text_input("Entrez un titre de film", key="unique_movie_title_input")
    if st.button("Recommander", key="unique_recommend_button"):
        recommend_movies(df, movie_title)

# Fonction principale
def main():
    st.title("Système d'analyse et de recommandation de films")

    menu = ["Accueil", "Analyse des données", "Recommandations", "Estimation de la popularité"]
    choice = st.sidebar.selectbox("Menu principal", menu, key="unique_main_menu")

    if choice == "Accueil":
        st.write("Bienvenue dans le système d'analyse et de recommandation de films !")

    elif choice == "Analyse des données":
        df = prepare_data()
        analyze_data(df)

    elif choice == "Recommandations":
        main_recommendations()

    elif choice == "Estimation de la popularité":
        main_popularity_estimation()

if __name__ == "__main__":
    main()
