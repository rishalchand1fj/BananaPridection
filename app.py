from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import traceback

app = Flask(__name__)

print("=================================")
print("Starting Banana Classifier")
print("=================================")

# Load model
print("Loading model...")

try:
    model = tf.keras.models.load_model(
        "banana_classifier.keras"
    )

    print("MODEL LOADED SUCCESSFULLY")

except Exception as e:

    print("MODEL LOADING FAILED")
    print(str(e))

    model = None


# IMPORTANT:
# Change this if your Colab class order is different
class_names = [
    "banana",
    "not_banana"
]

IMG_SIZE = (224, 224)


@app.route("/")
def home():

    return render_template(
        "index.html"
    )


@app.route("/predict", methods=["POST"])
def predict():

    print("")
    print("==============================")
    print("PREDICTION REQUEST RECEIVED")
    print("==============================")


    try:

        # -------------------------
        # Check model
        # -------------------------

        if model is None:

            return jsonify({
                "error":
                "Model failed to load on server."
            }), 500


        # -------------------------
        # Check uploaded file
        # -------------------------

        print(
            "Files received:",
            list(request.files.keys())
        )


        if "image" not in request.files:

            print(
                "ERROR: image field missing"
            )

            return jsonify({
                "error":
                "No image was received."
            }), 400


        file =
            request.files["image"]


        print(
            "Filename:",
            file.filename
        )


        if file.filename == "":

            return jsonify({
                "error":
                "No image selected."
            }), 400


        # -------------------------
        # Open image
        # -------------------------

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


        # -------------------------
        # Resize image
        # -------------------------

        image = image.resize(
            IMG_SIZE
        )


        print(
            "Resized size:",
            image.size
        )


        # -------------------------
        # Convert to NumPy
        # -------------------------

        image_array = np.array(
            image
        )


        print(
            "Array shape:",
            image_array.shape
        )


        # -------------------------
        # Add batch dimension
        # -------------------------

        image_array = np.expand_dims(
            image_array,
            axis=0
        )


        print(
            "Final input shape:",
            image_array.shape
        )


        # -------------------------
        # Make prediction
        # -------------------------

        print(
            "Running model prediction..."
        )


        predictions = model.predict(
            image_array,
            verbose=0
        )


        print(
            "Raw prediction:",
            predictions
        )


        predictions = predictions[0]


        # -------------------------
        # Find predicted class
        # -------------------------

        predicted_index = int(
            np.argmax(predictions)
        )


        print(
            "Predicted index:",
            predicted_index
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
            "Predicted class:",
            predicted_class
        )


        print(
            "Confidence:",
            confidence
        )


        # -------------------------
        # Return JSON
        # -------------------------

        result = {

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
            result
        )


        return jsonify(
            result
        )


    except Exception as e:


        print("")
        print(
            "!!!!!!!!!!!! ERROR !!!!!!!!!!!!"
        )


        print(
            str(e)
        )


        traceback.print_exc()


        print(
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        )


        return jsonify({

            "error":
                str(e)

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
