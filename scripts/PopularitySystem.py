import sys
sys.path.append('..')
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsRegressor



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
        self.titletypePipeline = Pipeline(steps=[("tfidf", OrdinalEncoder())])
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
