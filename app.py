from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import io

app = Flask(__name__)
model = tf.keras.models.load_model("sign_language_cnn.h5")

def num_to_char(num):
    return chr(num + 65) if num < 9 else chr(num + 66)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]
    image = Image.open(io.BytesIO(file.read())).convert("L")
    image = image.resize((28, 28))

    img = np.array(image) / 255.0
    img = img.reshape(1, 28, 28, 1)

    pred = model.predict(img)
    cls = int(np.argmax(pred))
    conf = float(np.max(pred))

    return jsonify({
        "prediction": num_to_char(cls),
        "confidence": conf
    })

if __name__ == "__main__":
    app.run(debug=True)
