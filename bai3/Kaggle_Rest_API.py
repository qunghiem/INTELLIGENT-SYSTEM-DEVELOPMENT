import pickle
from flask import Flask, request, json, jsonify
import numpy as np
app = Flask(__name__)
#---the filename of the saved model---
filename = 'exam_result_prediction_model.sav'
#---load the saved model---
loaded_model = pickle.load(open(filename, 'rb'))
@app.route('/exam/v1/predict', methods=['POST'])
def predict():
 #---get the features to predict---
 features = request.json
 #---create the features list for prediction---
 features_list = [float(features["study_hours"]), float(features["prev_exam_score"])]
 #---get the prediction class---
 prediction = loaded_model.predict([features_list])
 #---get the prediction probabilities---
 confidence = loaded_model.predict_proba([features_list])
 #---formulate the response to return to client---
 response = {}
 response['prediction'] = int(prediction[0])
 response['confidence'] = str(round(np.amax(confidence[0]) * 100 ,2))
 return jsonify(response)
if __name__ == '__main__':
 app.run(host='0.0.0.0', port=5000)