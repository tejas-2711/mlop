from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)
model_path = "src/models/best_model.pkl"

if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    raise FileNotFoundError(
        f"The model file {model_path} does not exist. "
        "Please run the training script to generate the model."
    )


@app.route("/")
def home():
    return render_template("form.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        print("Received data:", data)  # Debugging statement

        features = np.array(data["features"]).reshape(1, -1)
        print("Features:", features)  # Debugging statement

        prediction = model.predict(features)
        return jsonify({"prediction": int(prediction[0])})
    except Exception as e:
        print("Error during prediction:", e)  # Debugging statement
        return "An error occurred during prediction.", 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)