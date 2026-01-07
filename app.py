import time
from flask import Flask, request, jsonify
import numpy as np
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
with open("mental_health_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Home route (for testing)
@app.route("/")
def home():
    return "Mental Health API Running"

# Prediction route
@app.route("/predict_v1", methods=["POST"])
def predict_v1():
    try:
        start_time = time.time()

        # Get input JSON
        input_data = request.get_json()

        screen_time = input_data.get("Screen_Time")
        anxiety_level = input_data.get("Anxiety_Level")
        sleep_hours = input_data.get("Sleep_Hours")

        # Validate input
        if screen_time is None or anxiety_level is None or sleep_hours is None:
            return jsonify({"error": "Missing input data"}), 400

        # Prepare features
        features = np.array([[screen_time, anxiety_level, sleep_hours]])

        # Predict
        prediction = model.predict(features)

        # Map prediction
        if prediction[0] == 0:
            result_label = "Normal"
        elif prediction[0] == 1:
            result_label = "Mild Stress"
        elif prediction[0] == 2:
            result_label = "Anxiety"
        else:
            result_label = "Depression Risk"

        execution_time = time.time() - start_time

        return jsonify({
            "prediction": result_label,
            "execution_time": execution_time
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# DO NOT use waitress or gunicorn here
# Gunicorn will run this app automatically on Render
if __name__ == "__main__":
    app.run()

