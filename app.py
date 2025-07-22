from flask import Flask, request, render_template
import pickle

# Load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    title = request.form['title']
    text = request.form['text']
    subject = request.form['subject']

    # Concatenate features like in training
    combined_text = f"{title} {subject} {text}"
    prediction = model.predict([combined_text])[0]


    result = "True News ✅" if prediction == 1 else "Fake News ❌"
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
