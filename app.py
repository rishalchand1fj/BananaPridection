from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model(
    "banana_classifier.keras"
)

# IMPORTANT:
# Make sure this order matches your Colab class_names
class_names = [
    "banana",
    "not_banana"
]

IMG_SIZE = (224, 224)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({
            "error": "No image uploaded"
        }), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({
            "error": "No image selected"
        }), 400

    try:

        # Open image
        image = Image.open(file).convert("RGB")

        # Resize to model input size
        image = image.resize(IMG_SIZE)

        # Convert to NumPy
        image_array = np.array(image)

        # Add batch dimension
        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        # Predict
        predictions = model.predict(
            image_array,
            verbose=0
        )[0]

        # Highest probability
        predicted_index = np.argmax(
            predictions
        )

        predicted_class = class_names[
            predicted_index
        ]

        confidence = (
            predictions[predicted_index] * 100
        )

        return jsonify({
            "prediction": predicted_class,
            "confidence": round(
                float(confidence),
                2
            )
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port
    )