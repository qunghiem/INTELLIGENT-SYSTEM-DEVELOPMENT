import pickle
from flask import Flask, request, json, jsonify
import numpy as np
app = Flask(__name__)
#---the filename of the saved model---
filename = 'medical_cost_prediction_model.sav'
#---load the saved model---
loaded_model = pickle.load(open(filename, 'rb'))
@app.route('/medical/v1/predict', methods=['POST'])
def predict():
 #---get the features to predict---
 features = request.json
 #---create the features list for prediction---
 features_list = [
    float(features["Age"]),
    float(features["BMI"]),
    float(features["Num_of_Diseases"]),
    float(features["Annual_Income"]),
    float(features["Days_in_Hospital"])
]
 #---get the prediction class---
 prediction = loaded_model.predict([features_list])
 response = {}
 response['prediction'] = float(prediction[0])
 return jsonify(response)
if __name__ == '__main__':
 app.run(host='0.0.0.0', port=5000)