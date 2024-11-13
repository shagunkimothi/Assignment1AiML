from flask import Flask, request, render_template_string
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and scaler (ensure the paths are correct)
model = joblib.load('best_rf_model (1).pkl')  # Update the path if necessary
scaler = joblib.load('scaler.pkl')  # Load the scaler (if it's saved)

# Define the HTML form inside the Python code for simplicity
form_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prediction Form</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
            background-color: #f4f6f8;
        }
        h2 {
            color: #333;
            text-align: center;
        }
        form {
            background-color: #fff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            max-width: 400px;
            width: 100%;
        }
        label {
            display: block;
            margin-top: 10px;
            font-weight: bold;
            color: #555;
        }
        input[type="number"] {
            width: 100%;
            padding: 8px;
            margin-top: 5px;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-sizing: border-box;
        }
        input[type="submit"] {
            width: 100%;
            padding: 10px;
            background-color: #28a745;
            color: #fff;
            border: none;
            border-radius: 4px;
            font-size: 16px;
            cursor: pointer;
            margin-top: 15px;
        }
        input[type="submit"]:hover {
            background-color: #218838;
        }
        h3 {
            color: #333;
            text-align: center;
            margin-top: 15px;
        }
    </style>
</head>
<body>
    <div>
        <h2>Enter Feature Values for Prediction</h2>
        <form method="POST" action="/predict">
            <label for="feature1">Feature 1:</label>
            <input type="number" step="any" name="feature1" required>

            <label for="feature2">Feature 2:</label>
            <input type="number" step="any" name="feature2" required>

            <label for="feature3">Feature 3:</label>
            <input type="number" step="any" name="feature3" required>

            <label for="feature4">Feature 4:</label>
            <input type="number" step="any" name="feature4" required>

            <label for="feature5">Feature 5:</label>
            <input type="number" step="any" name="feature5" required>

            <label for="feature6">Feature 6:</label>
            <input type="number" step="any" name="feature6" required>

            <label for="feature7">Feature 7:</label>
            <input type="number" step="any" name="feature7" required>

            <input type="submit" value="Predict">
        </form>

        {% if prediction_text %}
            <h3>{{ prediction_text }}</h3>
        {% endif %}
    </div>
</body>
</html>
"""



@app.route('/')
def index():
    return render_template_string(form_html)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input features from the form
        feature1 = float(request.form['feature1'])
        feature2 = float(request.form['feature2'])
        feature3 = float(request.form['feature3'])
        feature4 = float(request.form['feature4'])
        feature5 = float(request.form['feature5'])
        feature6 = float(request.form['feature6'])
        feature7 = float(request.form['feature7'])

        # Print the received input values for debugging
        print(f"Received input: {feature1}, {feature2}, {feature3}, {feature4}, {feature5}, {feature6}, {feature7}")

        # Combine the features into an array for the model
        features = np.array([feature1, feature2, feature3, feature4, feature5, feature6, feature7]).reshape(1, -1)

        # Scale the features using the loaded scaler
        features_scaled = scaler.transform(features)

        # Print the scaled features to check the scaling process
        print(f"Scaled features: {features_scaled}")

        # Make the prediction
        prediction = model.predict(features_scaled)[0]

        # Translate prediction to readable result
        result = "High" if prediction == 1 else "Low"

        return render_template_string(form_html, prediction_text=f"Predicted Food Waste Category: {result}")

    except Exception as e:
        # Print the detailed error message for debugging
        print(f"Error: {str(e)}")
        return render_template_string(form_html, prediction_text="Error occurred. Please check your inputs.")

if __name__ == '__main__':
    app.run(debug=True)