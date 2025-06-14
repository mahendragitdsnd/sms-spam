
from flask import Flask, request, render_template
import pickle

# Load model and vectorizer
model = pickle.load(open('finalized_model.sav', 'rb'))
vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    data = vectorizer.transform([message])
    prediction = model.predict(data)[0]
    result = "Spam" if prediction == 1 else "Ham"
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
