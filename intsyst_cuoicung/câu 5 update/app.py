from flask import Flask, render_template, request, jsonify
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf

# Tải mô hình và tokenizer
save_path = "./saved_model"
try:
    # Thử tải mô hình từ thư mục đã lưu
    model = TFBertForSequenceClassification.from_pretrained(save_path)
    tokenizer = BertTokenizer.from_pretrained(save_path)
except:
    # Nếu không tìm thấy, tải mô hình mặc định và lưu lại
    model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

# Khởi tạo Flask app
app = Flask(__name__)

# Hàm dự đoán
def predict_sentiment(text):
    # Tokenize văn bản đầu vào
    encoding = tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors="tf")
    predictions = model.predict(encoding)
    logits = predictions.logits
    probabilities = tf.nn.softmax(logits, axis=-1)
    predicted_label = tf.argmax(probabilities, axis=-1).numpy()[0]
    return "Hài lòng" if predicted_label == 1 else "Không hài lòng"

# Route chính (trang giao diện)
@app.route("/")
def index():
    return render_template("index.html")

# API xử lý dự đoán
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    user_input = data.get("text", "")
    if not user_input.strip():
        return jsonify({"error": "Vui lòng nhập lời nhận xét!"}), 400
    result = predict_sentiment(user_input)
    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(debug=True, port=5001)
