import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils import resample

# === Load data ===
df = pd.read_csv("databersih1 (1) (1).csv")

# === Balance data (oversampling label minoritas) ===
df_0 = df[df['sentiment'] == 0]
df_1 = df[df['sentiment'] == 1]

df_1_upsampled = resample(df_1, replace=True, n_samples=len(df_0), random_state=42)
df_balanced = pd.concat([df_0, df_1_upsampled]).sample(frac=1, random_state=42)

# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(
    df_balanced['clean_text'], df_balanced['sentiment'], test_size=0.2, random_state=42, stratify=df_balanced['sentiment']
)

# === Pipeline training ===
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,2), stop_words='english', max_df=0.9, min_df=5)),
    ('clf', LogisticRegression(class_weight='balanced', max_iter=1000))
])

pipeline.fit(X_train, y_train)

# === Simpan model ===
os.makedirs("model", exist_ok=True)
joblib.dump(pipeline, "model/naive_bayes_model.pkl")

print("✅ Model berhasil dilatih dan disimpan di 'model/naive_bayes_model.pkl'")
