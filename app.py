import time
from flask import Flask, request, jsonify
import numpy as np
import pickle
from waitress import serve  # Using Waitress to serve the app

# Initialize Flask app
app = Flask(__name__)

# Load the trained model (ensure this path is correct)
with open('mental_health_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Define the prediction route
@app.route("/predict_v1", methods=["POST"])
def predict_v1():
    try:
        start_time = time.time()  # Start time for prediction

        # Get input features from the user
        input_data = request.get_json()

        # Extract features from the input
        screen_time = input_data.get('Screen_Time')
        anxiety_level = input_data.get('Anxiety_Level')
        sleep_hours = input_data.get('Sleep_Hours')

        # Validate input
        if screen_time is None or anxiety_level is None or sleep_hours is None:
            return jsonify({'error': 'Missing input data'}), 400

        # Process the input data
        features = np.array([[screen_time, anxiety_level, sleep_hours]])

        # Make the prediction using the trained model
        prediction = model.predict(features)

        # Map the prediction to a human-readable label
        if prediction[0] == 0:
            result_label = 'Normal'
        elif prediction[0] == 1:
            result_label = 'Mild Stress'
        elif prediction[0] == 2:
            result_label = 'Anxiety'
        else:
            result_label = 'Depression Risk'

        # End time for prediction
        end_time = time.time()
        execution_time = end_time - start_time  # Calculate time taken

        # Return the result as JSON along with the execution time
        result = {'prediction': result_label, 'execution_time': execution_time}
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the app with Waitress instead of Gunicorn
if __name__ == "__main__":
    serve(app, host='0.0.0.0', port=5000)
