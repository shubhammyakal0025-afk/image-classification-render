from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

model = tf.keras.models.load_model("image_model.h5")
class_names = ["cat", "dog"]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        file = request.files["image"]
        image = Image.open(file).resize((224, 224))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)

        pred = model.predict(image)
        prediction = class_names[np.argmax(pred)]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run()
