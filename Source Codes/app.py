from flask import Flask, render_template, request
import numpy
import pickle

app= Flask(__name__)

# Loading the trained model from the pickle file
with open('model.pkl', 'rb') as file:
    knn= pickle.load(file)

# Defining the home route
@app.route('/')
def home():
    # Rendering the main template
    return render_template('main.html')

# Defining the prediction route
@app.route('/predict', methods=['GET','POST'])

def predict():
    if request.method=='POST':
        # Retrieving form data from the request object
        age= int(request.form['age'])
        sex= int(request.form['sex'])
        cp= int(request.form['cp'])
        trestbps= int(request.form['trestbps'])
        chol= int(request.form['chol'])
        fbs= int(request.form['fbs'])
        restecg= int(request.form['restecg'])
        thalach= int(request.form['thalach'])
        exang= int(request.form['exang'])
        oldpeak= float(request.form['oldpeak'])
        slope= int(request.form['slope'])
        ca= int(request.form['ca'])
        thal= int(request.form['thal'])

        # Creating a numpy array from the form data
        input_data = numpy.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        
        # Making a prediction using the trained model
        prediction = knn.predict(input_data)

        # Rendering the result.html template with the prediction result
        return render_template('result.html', prediction=prediction)

# Running the Flask app in debug mode
if __name__ == '__main__':
    app.run(debug=True)