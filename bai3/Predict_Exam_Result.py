import json
import requests
def predict_diabetes(study_hours, prev_exam_score):
 url = 'http://127.0.0.1:5000/exam/v1/predict'
 data = {"study_hours":study_hours, "prev_exam_score":prev_exam_score}
 data_json = json.dumps(data)
 headers = {'Content-type':'application/json'}
 response = requests.post(url, data=data_json, headers=headers)
 result = json.loads(response.text)
 return result
if __name__ == "__main__":
 study_hours = input('Hours spend for studying: ')
 prev_exam_score = input('Previous exam score: ')
 predictions = predict_diabetes(study_hours, prev_exam_score)
 print("This student will pass the exam!" if predictions["prediction"] == 1 else "This student will not pass the exam!")