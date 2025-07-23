from flask import Flask, render_template, request
import pickle

# Load the saved vectorizer and model
with open('model/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('model/model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    user_input = ""

    if request.method == 'POST':
        user_input = request.form['news']
        if user_input:
            vect = vectorizer.transform([user_input])
            pred = model.predict(vect)[0]
            prediction = "Fake News" if pred == 1 else "Real News"

    return render_template('index.html', prediction=prediction, user_input=user_input)

if __name__ == '__main__':
    app.run(debug=True)
