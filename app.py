from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load model and scaler
model = joblib.load("model.pkl")
try:
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError:
    print("Scaler file (scaler.pkl) not found. Proceeding without scaling.")
    scaler = None
except Exception as e:
    print(f"Error loading scaler: {e}. Proceeding without scaling.")
    scaler = None


@app.route('/')
def home():
    """Renders the home page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction request from the form."""
    try:
        # Collect input values from form
        features = [
            float(request.form['subscribers']),
            float(request.form['video_count']),
            float(request.form['account_age']),
            float(request.form['post_frequency_per_year']),
            float(request.form['like_count']),
            float(request.form['comment_count'])
        ]

        # Convert to numpy array and reshape for a single prediction
        final_features = np.array(features).reshape(1, -1)

        # Apply scaler if it was successfully loaded
        if scaler:
            final_features = scaler.transform(final_features)

        # Make prediction
        prediction = model.predict(final_features)[0]

        # Format the prediction for display
        prediction_output = f'Estimated View Count: {int(prediction):,}'

    except Exception as e:
        # Handle errors (e.g., non-numeric input)
        prediction_output = f'Error: {str(e)}. Please check your inputs.'

    # Render the home page again, this time with the prediction text
    return render_template('index.html', prediction_text=prediction_output)


if __name__ == "__main__":
    # Run the app in debug mode
    # host='0.0.0.0' makes it accessible on your network
    app.run(debug=True, host='0.0.0.0', port=5000)