import json
import requests
def predict_medical_cost(age, bmi, num_of_diseases, annual_income, days_in_hospital):
 url = 'http://127.0.0.1:5000/medical/v1/predict'
 data = {
        "Age": age,
        "BMI": bmi,
        "Num_of_Diseases": num_of_diseases,
        "Annual_Income": annual_income,
        "Days_in_Hospital": days_in_hospital
    }
 data_json = json.dumps(data)
 headers = {'Content-type':'application/json'}
 response = requests.post(url, data=data_json, headers=headers)
 result = json.loads(response.text)
 return result
if __name__ == "__main__":
    # Nhập dữ liệu từ người dùng
    age = float(input('Enter Age: '))
    bmi = float(input('Enter BMI: '))
    num_of_diseases = int(input('Enter Number of Diseases: '))
    annual_income = float(input('Enter Annual Income: '))
    days_in_hospital = int(input('Enter Days in Hospital: '))

    # Dự đoán chi phí y tế
    predictions = predict_medical_cost(age, bmi, num_of_diseases, annual_income, days_in_hospital)
    
    # Xử lý và in kết quả dự đoán
    print(f"Predicted Medical Cost: {predictions['prediction']}")