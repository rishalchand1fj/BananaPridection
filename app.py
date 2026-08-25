from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import traceback

app = Flask(__name__)

print("Loading model...")

model = tf.keras.models.load_model(
    "banana_classifier.keras"
)

print("Model loaded successfully!")

# IMPORTANT:
# This order MUST match your Colab class_names
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

    print("Prediction request received")

    try:

        # Check file
        if "image" not in request.files:
            print("ERROR: No image in request")

            return jsonify({
                "error": "No image uploaded"
            }), 400

        file = request.files["image"]

        print("Received file:", file.filename)

        if file.filename == "":
            return jsonify({
                "error": "No image selected"
            }), 400

        # Open image
        image = Image.open(file).convert("RGB")

        print(
            "Original image size:",
            image.size
        )

        # Resize
        image = image.resize(IMG_SIZE)

        print(
            "Resized image:",
            image.size
        )

        # Convert to NumPy
        image_array = np.array(image)

        # Add batch dimension
        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        print(
            "Input shape:",
            image_array.shape
        )

        # Prediction
        predictions = model.predict(
            image_array,
            verbose=0
        )[0]

        print(
            "Predictions:",
            predictions
        )

        # Get highest probability
        predicted_index = np.argmax(
            predictions
        )

        predicted_class = class_names[
            predicted_index
        ]

        confidence = (
            predictions[predicted_index]
            * 100
        )

        print(
            "Prediction:",
            predicted_class
        )

        print(
            "Confidence:",
            confidence
        )

        return jsonify({
            "prediction": predicted_class,
            "confidence": round(
                float(confidence),
                2
            )
        })

    except Exception as e:

        print("PREDICTION ERROR:")
        print(str(e))

        traceback.print_exc()

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
