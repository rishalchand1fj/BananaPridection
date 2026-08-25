from flask import Flask, render_template, request, jsonify
import os

# Limit TensorFlow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import numpy as np
from PIL import Image
import traceback


# Limit TensorFlow CPU threads
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)


app = Flask(__name__)


# ==========================================
# LOAD MODEL
# ==========================================

print("================================")
print("Starting Banana Classifier")
print("================================")

print("Loading model...")

try:

    model = tf.keras.models.load_model(
        "banana_classifier.keras"
    )

    print("MODEL LOADED SUCCESSFULLY")

    print(
        "Model input shape:",
        model.input_shape
    )

    print(
        "Model output shape:",
        model.output_shape
    )

except Exception as e:

    print("MODEL LOADING FAILED")

    print(e)

    model = None


# ==========================================
# SETTINGS
# ==========================================

IMG_SIZE = (224, 224)

class_names = [
    "banana",
    "not_banana"
]


# ==========================================
# HOME
# ==========================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ==========================================
# PREDICT
# ==========================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    print("")
    print("==============================")
    print("PREDICTION REQUEST RECEIVED")
    print("==============================")


    try:

        # ------------------------------
        # Check model
        # ------------------------------

        if model is None:

            return jsonify({
                "success": False,
                "error":
                    "Model was not loaded."
            }), 500


        # ------------------------------
        # Check image
        # ------------------------------

        print(
            "Files:",
            list(request.files.keys())
        )


        if "image" not in request.files:

            return jsonify({
                "success": False,
                "error":
                    "No image received."
            }), 400


        file = request.files["image"]


        print(
            "Filename:",
            file.filename
        )


        # ------------------------------
        # Open image
        # ------------------------------

        print(
            "Opening image..."
        )


        image = Image.open(
            file
        ).convert("RGB")


        print(
            "Original size:",
            image.size
        )


        # ------------------------------
        # Resize
        # ------------------------------

        image = image.resize(
            IMG_SIZE
        )


        print(
            "Resized:",
            image.size
        )


        # ------------------------------
        # NumPy
        # ------------------------------

        image_array = np.array(
            image,
            dtype=np.float32
        )


        # ------------------------------
        # Normalize
        # ------------------------------

        image_array /= 255.0


        # ------------------------------
        # Add batch
        # ------------------------------

        image_array = np.expand_dims(
            image_array,
            axis=0
        )


        print(
            "Input shape:",
            image_array.shape
        )


        print(
            "Starting TensorFlow prediction..."
        )


        # =================================
        # PREDICTION
        # =================================

        predictions = model.predict(
            image_array,
            batch_size=1,
            verbose=0
        )


        print(
            "Prediction completed!"
        )


        print(
            "Raw output:",
            predictions
        )


        predictions = np.asarray(
            predictions
        )


        # =================================
        # HANDLE OUTPUT
        # =================================

        # Remove batch dimension

        if predictions.ndim > 1:

            predictions = predictions[0]


        print(
            "Processed output:",
            predictions
        )


        # =================================
        # BINARY MODEL
        # =================================

        if predictions.size == 1:

            probability = float(
                predictions.flatten()[0]
            )


            if probability >= 0.5:

                predicted_class = "banana"

                confidence = (
                    probability * 100
                )

            else:

                predicted_class = "not_banana"

                confidence = (
                    (1 - probability)
                    * 100
                )


        # =================================
        # TWO CLASS MODEL
        # =================================

        else:

            predicted_index = int(
                np.argmax(
                    predictions
                )
            )


            predicted_class = (
                class_names[
                    predicted_index
                ]
            )


            confidence = float(
                predictions[
                    predicted_index
                ] * 100
            )


        print(
            "Prediction:",
            predicted_class
        )


        print(
            "Confidence:",
            confidence
        )


        # =================================
        # RETURN
        # =================================

        return jsonify({

            "success": True,

            "prediction":
                predicted_class,

            "confidence":
                round(
                    confidence,
                    2
                )

        })


    except Exception as e:

        print("")
        print("==============================")
        print("PREDICTION ERROR")
        print("==============================")


        print(
            str(e)
        )


        traceback.print_exc()


        return jsonify({

            "success": False,

            "error":
                str(e)

        }), 500


# ==========================================
# START
# ==========================================

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
