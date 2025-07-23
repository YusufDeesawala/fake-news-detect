from flask import Flask, request, render_template
import pickle

# Load the trained model and the fitted TFIDF vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    title = request.form['title']
    text = request.form['text']
    subject = request.form['subject']

    # Combine the input features as done during training
    combined_text = f"{title} {subject} {text}"

    # Vectorize using the loaded vectorizer
    vectorized_input = vectorizer.transform([combined_text])

    # Predict using the loaded model
    prediction = model.predict(vectorized_input)[0]

    # Map prediction to display text
    result = "True News ✅" if prediction == 1 else "Fake News ❌"
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
