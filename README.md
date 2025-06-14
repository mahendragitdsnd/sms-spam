# SMS Spam Detection Web App
This project demonstrates a complete machine learning pipeline for detecting SMS spam messages using Support Vector Machine (SVM). The trained model is deployed using a Flask web application.
sms_spam_full_project/
├── train_model.py              # Trains the SVM model
├── app.py                      # Flask app for predictions
├── requirements.txt            # Dependencies
├── finalized_model.sav         # (Generated) Trained model file
├── tfidf_vectorizer.pkl        # (Generated) TF-IDF vectorizer
├── templates/
│   └── index.html              # HTML frontend
└── static/
    └── style.css               # CSS for UI
*How to Use
1. Install Dependencies
  pip install -r requirements.txt
2. Train the Model
 Place the spam.csv dataset in the same directory, then run:
  python train_model.py
 This creates finalized_model.sav and tfidf_vectorizer.pkl.

3. Launch the Web App
  python app.py
 Then go to http://localhost:5000 in your browser.

*Features
 Text preprocessing (tokenization, stopword removal, stemming)
 TF-IDF feature extraction
 GridSearchCV for hyperparameter tuning
 Spam/Ham classification
 Simple Flask-based UI

