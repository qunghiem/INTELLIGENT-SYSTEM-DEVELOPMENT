import pickle
from flask import Flask, request, json, jsonify
import numpy as np
app = Flask(__name__)
#---the filename of the saved model---
filename = 'diemthi_model.sav'
#---load the saved model---
loaded_model = pickle.load(open(filename, 'rb'))
@app.route('/diemthi/v1/predict', methods=['POST'])
def predict():
 #---get the features to predict---
 features = request.json
 #---create the features list for prediction---
 features_list = [float(features["diem10%.1"]), float(features["diem10%.2"]), float(features["diem20%"])]
 #---get the prediction class---
 prediction = loaded_model.predict([features_list])
 #---formulate the response to return to client---
 response = {}
 response['prediction'] = float(prediction[0])
 return jsonify(response)
if __name__ == '__main__':
 app.run(host='0.0.0.0', port=5000)