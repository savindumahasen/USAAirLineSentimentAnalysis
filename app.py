from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load the trained model and TfidfVectorizer
tf_vec = joblib.load('tfidf_vectorizer.pkl')
model = joblib.load('sentimentanalysis_classifier.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    message = data['message']
    transformed_message = tf_vec.transform([message]).toarray()
    prediction = model.predict(transformed_message)[0]

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
