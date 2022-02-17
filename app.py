import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model_new = pickle.load(open('model_new.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]     
    prediction = model_new.predict(final_features)
    
    output = round(prediction[0],2)
    
    return render_template('index.html', prediction_text = 'Home Mortgage is {}'.format(output))

if __name__ == 'main':
    app.run(debug=True)