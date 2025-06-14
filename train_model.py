import pandas as pd
import nltk
import string
import pickle

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

# Download resources
nltk.download('punkt')
nltk.download('stopwords')

# Load data
df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['Label', 'EmailText']

# Encode labels
encoder = LabelEncoder()
df['Label'] = encoder.fit_transform(df['Label'])

# Remove duplicates
df = df.drop_duplicates(keep='first')

# Text preprocessing functions
ps = PorterStemmer()

def preprocess_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    clean_tokens = [ps.stem(word) for word in tokens if word.isalnum() and word not in stopwords.words('english')]
    return " ".join(clean_tokens)

df['processed'] = df['EmailText'].apply(preprocess_text)

# Train/test split
X = df['processed']
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# TF-IDF
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# SVM with GridSearch
params = {'kernel': ['linear', 'rbf'], 'C': [1, 10], 'gamma': [0.001, 0.0001]}
model = GridSearchCV(SVC(), params, cv=5)
model.fit(X_train_vec, y_train)

# Save model and vectorizer
pickle.dump(model, open('finalized_model.sav', 'wb'))
pickle.dump(vectorizer, open('tfidf_vectorizer.pkl', 'wb'))

# Optional: Evaluate
print("Best Parameters:", model.best_params_)
print("Accuracy:", model.score(X_test_vec, y_test))
