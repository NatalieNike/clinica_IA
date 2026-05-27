import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

#carregar dataset
df = pd.read_csv("dataset.csv")

x = df["sintomas"]
y = df["categoria"]

model = Pipeline([
    ("vectorizer", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])

model.fit(x, y)

joblib.dump(model, "model.pkl")

print("Modelo treinado com sucesso!")