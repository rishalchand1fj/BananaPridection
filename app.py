from flask import Flask, render_template, request, jsonify

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import numpy as np
from PIL import Image
import traceback

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

app = Flask(__name__)

print("================================")
print("Starting Banana Classifier")
print("================================")

# -----------------------------
# Load model
# -----------------------------

print("Loading model...")

try:
    model = tf.keras.models.load_model(
        "banana_classifier.keras"
    )

    print("MODEL LOADED SUCCESSFULLY")

except Exception as e:
    print("MODEL LOADING FAILED")
    print(e)
    model = None


# -----------------------------
# Model settings
# -----------------------------

IMG_SIZE = (224, 224)

# Change these if your training
# class names are different
class_names = [
    "banana",
    "not_banana"
]


# -----------------------------
# Home page
# -----------------------------

@app.route("/")
def home():
    return render_template("index.html")


# -----------------------------
# Prediction
# -----------------------------

@app.route("/predict", methods=["POST"])
def predict():

    print("\n==============================")
    print("PREDICTION REQUEST RECEIVED")
    print("==============================")

    try:

        # Check model
        if model is None:

            return jsonify({
                "success": False,
                "error": "Model was not loaded."
            }), 500


        # Check uploaded file
        print(
            "Received files:",
            list(request.files.keys())
        )


        if "image" not in request.files:

            return jsonify({
                "success": False,
                "error": "No image was uploaded."
            }), 400


        file = request.files["image"]


        print(
            "Filename:",
            file.filename
        )


        if file.filename == "":

            return jsonify({
                "success": False,
                "error": "No image was selected."
            }), 400


        # Open image
        print("Opening image...")

        image = Image.open(file)

        print(
            "Original image:",
            image.size,
            image.mode
        )


        # Convert to RGB
        image = image.convert("RGB")


        # Resize
        image = image.resize(IMG_SIZE)

        print(
            "Resized image:",
            image.size
        )


        # Convert to numpy
        image_array = np.array(image)

        print(
            "Image array shape:",
            image_array.shape
        )


        # Normalize
        #
        # IMPORTANT:
        # If your Colab training used rescaling
        # such as /255, keep this.
        #
        image_array = image_array / 255.0


        # Add batch dimension
        image_array = np.expand_dims(
            image_array,
            axis=0
        )


        print(
            "Model input shape:",
            image_array.shape
        )


        # -----------------------------
        # Prediction
        # -----------------------------

        print("Running prediction...")


        predictions = model.predict(
            image_array,
            verbose=0
        )


        print(
            "Raw model output:",
            predictions
        )


        # Remove batch dimension
        predictions = predictions[0]


        print(
            "Processed output:",
            predictions
        )


        # -----------------------------
        # Handle binary model
        # -----------------------------

        if np.size(predictions) == 1:

            probability = float(
                np.asarray(predictions).reshape(-1)[0]
            )


            # Assumption:
            # 0 = not_banana
            # 1 = banana

            if probability >= 0.5:

                predicted_class = "banana"

                confidence = probability * 100

            else:

                predicted_class = "not_banana"

                confidence = (
                    1 - probability
                ) * 100


        # -----------------------------
        # Handle 2-class model
        # -----------------------------

        else:

            predicted_index = int(
                np.argmax(predictions)
            )


            predicted_class = class_names[
                predicted_index
            ]


            confidence = float(
                predictions[
                    predicted_index
                ] * 100
            )


        print(
            "FINAL PREDICTION:",
            predicted_class
        )


        print(
            "CONFIDENCE:",
            confidence
        )


        # -----------------------------
        # Return JSON
        # -----------------------------

        response = {

            "success": True,

            "prediction":
                predicted_class,

            "confidence":
                round(
                    confidence,
                    2
                )
        }


        print(
            "Returning:",
            response
        )


        return jsonify(response)


    except Exception as e:

        print("\n!!!!!!!!!!!!!!!!!!!!!!!!")
        print("PREDICTION ERROR")
        print("!!!!!!!!!!!!!!!!!!!!!!!!")

        print(
            str(e)
        )

        traceback.print_exc()


        return jsonify({

            "success": False,

            "error":
                str(e)

        }), 500


# -----------------------------
# Start Flask
# -----------------------------

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
