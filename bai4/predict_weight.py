import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import tkinter as tk
from tkinter import messagebox

# Tạo dữ liệu mẫu
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 'Grace', 'Hannah', 'Ivy', 'Jack',
             'Karen', 'Leo', 'Mona', 'Nick', 'Olivia', 'Paul', 'Quincy', 'Rachel', 'Steve', 'Tina'],
    'Height': [1.60, 1.70, 1.75, 1.80, 1.65, 1.85, 1.70, 1.78, 1.68, 1.72,
               1.55, 1.90, 1.60, 1.82, 1.77, 1.68, 1.74, 1.79, 1.76, 1.62],
    'Weight': [50, 70, 75, 80, 65, 90, 72, 78, 68, 74,
               55, 85, 56, 82, 78, 67, 73, 77, 72, 60],
    'Job': ['Engineer', 'Doctor', 'Artist', 'Teacher', 'Chef', 'Nurse', 'Driver', 'Scientist',
            'Pilot', 'Architect', 'Lawyer', 'Musician', 'Police', 'Firefighter', 'Dancer',
            'Model', 'Actor', 'Writer', 'Farmer', 'Journalist']
}

# Chuyển dữ liệu thành DataFrame
df = pd.DataFrame(data)

# Tính toán BMI và phân loại
df['BMI'] = df['Weight'] / (df['Height'] ** 2)

def classify_bmi(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif bmi > 25:
        return 'Overweight'
    else:
        return 'Normal weight'

df['Classification'] = df['BMI'].apply(classify_bmi)

# Tạo biến độc lập (X) và biến phụ thuộc (y)
X = df[['Height', 'Job']]
y = df['Weight']

# Mã hóa biến Job thành dạng OneHot
column_transformer = ColumnTransformer(transformers=[
    ('encoder', OneHotEncoder(), ['Job'])
], remainder='passthrough')

# Tạo pipeline kết hợp mã hóa và mô hình hồi quy tuyến tính
model = Pipeline(steps=[
    ('preprocessor', column_transformer),
    ('regressor', LinearRegression())
])

# Huấn luyện mô hình
model.fit(X, y)

# Hàm để dự đoán cân nặng và BMI
def predict_weight_bmi(height, job):
    weight_pred = model.predict([[height, job]])
    bmi_pred = weight_pred[0] / (height ** 2)
    classification = classify_bmi(bmi_pred)
    return weight_pred[0], bmi_pred, classification

# Hàm khi người dùng nhấn nút phân loại
def classify():
    try:
        # Lấy giá trị chiều cao từ input
        height = float(entry_height.get().strip())
        
        # Lấy nghề nghiệp được chọn
        selected_job = job_var.get()

        # Dự đoán cân nặng và BMI
        weight_pred, bmi_pred, classification = predict_weight_bmi(height, selected_job)
        
        # Hiển thị kết quả dự đoán
        messagebox.showinfo("Prediction Result", 
                            f"Height: {height} m\n"
                            f"Job: {selected_job}\n"
                            f"Predicted Weight: {weight_pred:.2f} kg\n"
                            f"Predicted BMI: {bmi_pred:.2f}\n"
                            f"Classification: {classification}")
    except ValueError:
        messagebox.showerror("Error", "Please enter a valid height!")

# Tạo cửa sổ giao diện
window = tk.Tk()
window.title("Weight and BMI Prediction")

# Tạo các thành phần giao diện
tk.Label(window, text="Enter height (m):").pack()
entry_height = tk.Entry(window)
entry_height.pack()

tk.Label(window, text="Select job:").pack()

# Tạo danh sách nghề nghiệp
job_var = tk.StringVar()
job_var.set(df['Job'].unique()[0])  # Nghề nghiệp mặc định

job_menu = tk.OptionMenu(window, job_var, *df['Job'].unique())
job_menu.pack()

button_predict = tk.Button(window, text="Predict", command=classify)
button_predict.pack()

# Chạy vòng lặp giao diện
window.mainloop()
