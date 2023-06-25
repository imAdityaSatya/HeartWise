import pickle

# Using trained model for prediction
with open('model.pkl', 'rb') as file:
    knn = pickle.load(file)

def predict_heart_disease(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    input_data = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
    
    prediction = knn.predict(input_data)

    if prediction[0] == 0:
        result = "Not likely to have heart disease"
    else:
        result = "Likely to have heart disease"
   
    return result

# Example usage
age = 35
sex = 0
cp = 1
trestbps = 120
chol = 263
fbs = 1
restecg = 173
thalach = 0
exang = 0
oldpeak = 2
slope = 0
ca = 3
thal = 1

prediction_result1 = predict_heart_disease(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
print("Prediction 1 :",prediction_result1)

age2 = 35
sex2 = 0
cp2 = 0
trestbps2 = 120
chol2 = 200
fbs2 = 0
restecg2 = 0
thalach2 = 180
exang2 = 0
oldpeak2 = 0.5
slope2 = 1
ca2 = 0
thal2 = 2

prediction_result2 = predict_heart_disease(age2, sex2, cp2, trestbps2, chol2, fbs2, restecg2, thalach2, exang2, oldpeak2, slope2, ca2, thal2)
print("Prediction 2:", prediction_result2)
