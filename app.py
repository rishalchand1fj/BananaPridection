from flask import Flask, render_template, request, jsonify
import os
import traceback

# Reduce TensorFlow messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import numpy as np
from PIL import Image


# =========================================================
# TENSORFLOW SETTINGS
# =========================================================

# Limit CPU usage on Render
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)


# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)


# =========================================================
# MODEL SETTINGS
# =========================================================

MODEL_PATH = "banana_classifier.keras"

IMG_SIZE = (224, 224)

# Change these if your training classes were different
CLASS_NAMES = [
    "Banana",
    "Not Banana"
]


# =========================================================
# LOAD MODEL
# =========================================================

print("========================================")
print("       BANANA CLASSIFIER")
print("========================================")

print("Loading model...")

try:

    model = tf.keras.models.load_model(
        MODEL_PATH
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

    print(str(e))

    traceback.print_exc()

    model = None


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================================
# PREDICT
# =========================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    print("")
    print("========================================")
    print("PREDICTION REQUEST RECEIVED")
    print("========================================")


    try:

        # -------------------------------------------------
        # CHECK MODEL
        # -------------------------------------------------

        if model is None:

            return jsonify({
                "success": False,
                "error": "Model was not loaded."
            }), 500


        # -------------------------------------------------
        # CHECK IMAGE
        # -------------------------------------------------

        if "image" not in request.files:

            return jsonify({
                "success": False,
                "error": "No image was uploaded."
            }), 400


        file = request.files["image"]


        if file.filename == "":

            return jsonify({
                "success": False,
                "error": "No image selected."
            }), 400


        print(
            "Image received:",
            file.filename
        )


        # -------------------------------------------------
        # OPEN IMAGE
        # -------------------------------------------------

        print("Opening image...")


        image = Image.open(
            file
        ).convert("RGB")


        print(
            "Original image size:",
            image.size
        )


        # -------------------------------------------------
        # RESIZE IMAGE
        # -------------------------------------------------

        image = image.resize(
            IMG_SIZE
        )


        print(
            "Resized image:",
            image.size
        )


        # -------------------------------------------------
        # CONVERT TO NUMPY
        # -------------------------------------------------

        image_array = np.array(
            image,
            dtype=np.float32
        )


        # -------------------------------------------------
        # NORMALIZE
        # -------------------------------------------------

        image_array = (
            image_array / 255.0
        )


        # -------------------------------------------------
        # ADD BATCH DIMENSION
        # -------------------------------------------------

        image_array = np.expand_dims(
            image_array,
            axis=0
        )


        print(
            "Input shape:",
            image_array.shape
        )


        print(
            "Expected model input:",
            model.input_shape
        )


        # -------------------------------------------------
        # PREDICT
        # -------------------------------------------------

        print("")
        print("STARTING MODEL PREDICTION...")


        prediction = model.predict(
            image_array,
            batch_size=1,
            verbose=0
        )


        print(
            "PREDICTION FINISHED!"
        )


        print(
            "Raw prediction:",
            prediction
        )


        # -------------------------------------------------
        # CONVERT OUTPUT
        # -------------------------------------------------

        prediction = np.asarray(
            prediction
        )


        # Remove batch dimension

        if prediction.ndim > 1:

            prediction = prediction[0]


        print(
            "Processed prediction:",
            prediction
        )


        # =================================================
        # BINARY CLASSIFICATION
        # =================================================

        if prediction.size == 1:

            probability = float(
                prediction.flatten()[0]
            )


            print(
                "Probability:",
                probability
            )


            # ---------------------------------------------
            # IMPORTANT
            # ---------------------------------------------
            #
            # If your model was trained with:
            #
            # banana = 1
            # not banana = 0
            #
            # this is correct.
            #
            # ---------------------------------------------

            if probability >= 0.5:

                result = "Banana"

                confidence = (
                    probability * 100
                )

            else:

                result = "Not Banana"

                confidence = (
                    (1 - probability) * 100
                )


        # =================================================
        # TWO-CLASS OUTPUT
        # =================================================

        else:

            predicted_index = int(
                np.argmax(
                    prediction
                )
            )


            print(
                "Predicted class index:",
                predicted_index
            )


            # Make sure index exists

            if (
                predicted_index
                < len(CLASS_NAMES)
            ):

                result = (
                    CLASS_NAMES[
                        predicted_index
                    ]
                )

            else:

                result = (
                    "Class "
                    + str(predicted_index)
                )


            confidence = float(
                prediction[
                    predicted_index
                ] * 100
            )


        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        print("")
        print("========================================")
        print("RESULT:", result)
        print(
            "CONFIDENCE:",
            round(confidence, 2),
            "%"
        )
        print("========================================")


        # -------------------------------------------------
        # RETURN JSON
        # -------------------------------------------------

        return jsonify({

            "success": True,

            "prediction": result,

            "confidence": round(
                confidence,
                2
            )

        })


    # =====================================================
    # ERROR
    # =====================================================

    except Exception as e:

        print("")
        print("========================================")
        print("PREDICTION ERROR")
        print("========================================")

        print(
            str(e)
        )

        traceback.print_exc()


        return jsonify({

            "success": False,

            "error": str(e)

        }), 500


# =========================================================
# HEALTH CHECK
# =========================================================

@app.route("/health")
def health():

    if model is not None:

        return jsonify({
            "status": "ok",
            "model": "loaded"
        })

    else:

        return jsonify({
            "status": "error",
            "model": "not loaded"
        }), 500


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )


    print(
        "Starting Flask server on port",
        port
    )


    app.run(
        host="0.0.0.0",
        port=port
    )
