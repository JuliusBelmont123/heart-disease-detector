from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
# Initialize Flask app
app = Flask(__name__)
# Load the saved logistic regression model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
        # Extract the form data
        form_data = request.form
        
        # Collect data from the form
        data = {
            'age': int(form_data['age']),
            'sex': 1 if form_data['sex'] == 'male' else 0,
            'chestPain': int({
                'typicalAngina': 0,
                'atypicalAngina': 1,
                'nonAnginalPain': 2,
                'asymptomatic': 3
            }[form_data['chestPain']]),
            'restingBP': int(form_data['restingBP'].split('-')[0]),  # Take the lower range value
            'serumChol': int(form_data['serumChol']),
            'fastingBloodSugar': 1 if form_data['fastingBloodSugar'] == 'yes' else 0,
            'restingECG': int({
                'normal': 0,
                'abnormal': 1,
                'hypertrophy': 2
            }[form_data['restingECG']]),
            'maxHeartRate': int(form_data['maxHeartRate']),
            'exerciseAngina': 1 if form_data['exerciseAngina'] == 'yes' else 0,
            'oldpeak': float(form_data['oldpeak']),
            'slope': int({
                'upsloping': 0,
                'flat': 1,
                'downsloping': 2
            }[form_data['slope']]),
            'numVessels': int(form_data['numVessels']),
            'thal': int({
                'normal': 0,
                'fixedDefect': 1,
                'reversableDefect': 2
            }[form_data['thal']])
        }
        t=()
        for i in data:
                t+=(data[i],)
        input_data_as_numpy_array=np.asarray(t)
        input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
        prediction=model.predict(input_data_reshaped)
        if (prediction[0]==0):
                return render_template('pass.html',n="Congratulations! You do not have a heart disease.")
        else:
                return render_template('pass.html',n="You could probably have a heart disease . Go see a doctor.")

@app.route('/')
def back():
       return render_template('index.html')        

if __name__ == '__main__':
    app.run(debug=True)
