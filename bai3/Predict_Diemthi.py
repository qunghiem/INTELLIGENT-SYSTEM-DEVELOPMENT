import json
import requests
def predict_diabetes(diem1, diem2, diem3):
 url = 'http://127.0.0.1:5000/diemthi/v1/predict'
 data = {"diem10%.1":diem1, "diem10%.2":diem2, "diem20%":diem3}
 data_json = json.dumps(data)
 headers = {'Content-type':'application/json'}
 response = requests.post(url, data=data_json, headers=headers)
 result = json.loads(response.text)
 return result
if __name__ == "__main__":
 diem1 = input('Diem 10% thu nhat: ')
 diem2 = input('Diem 10% thu hai: ')
 diem3 = input('Diem 20%: ')
 predictions = predict_diabetes(diem1, diem2, diem3)
 print("Diem du doan: ", predictions["prediction"])