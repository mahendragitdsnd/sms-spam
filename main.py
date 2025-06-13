import numpy as np
import pandas as pd
import nltk
import string
import pickle

from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load and clean data
df = pd.read_csv('/content/drive/MyDrive/spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]   # Assuming 'v1' is label and 'v2' is email text
df.columns = ['Label', 'EmailText']

encoder = LabelEncoder()
df['Label'] = encoder.fit_transform(df['Label'])

df = df.drop_duplicates(keep='first')

# Text Preprocessing
ps = PorterStemmer()

def get_importantFeatures(sent):
    sent = sent.lower()
    tokens = nltk.word_tokenize(sent)
    return [word for word in tokens if word.isalnum()]

def removing_stopWords(tokens):
    stopwords = set(nltk.corpus.stopwords.words('english'))
    return [word for word in tokens if word not in stopwords and word not in string.punctuation]

def porter_stem(tokens):
    return " ".join([ps.stem(word) for word in tokens])

# Apply preprocessing
df['imp_feature'] = df['EmailText'].apply(get_importantFeatures)
df['imp_feature'] = df['imp_feature'].apply(removing_stopWords)
df['imp_feature'] = df['imp_feature'].apply(porter_stem)

# Splitting and vectorizing
X = df['imp_feature']
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

tfidf = TfidfVectorizer()
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

# SVM with GridSearchCV
tuned_parameters = {'kernel': ['linear', 'rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]}
model = GridSearchCV(svm.SVC(), tuned_parameters, cv=5)
model.fit(X_train_vec, y_train)

# Evaluate
accuracy = model.score(X_test_vec, y_test)
print("Accuracy:", accuracy)

# Save model and vectorizer
with open('finalized_model.sav', 'wb') as f:
    pickle.dump(model, f)

with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
