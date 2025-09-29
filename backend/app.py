# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import tensorflow as tf
import json
import os

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "ml", "model.h5")
model = tf.keras.models.load_model(MODEL_PATH)

with open(os.path.join(os.path.dirname(__file__),"..","ml","class_indices.json")) as f:
    class_indices = json.load(f)
inv_idx = {v:k for k,v in class_indices.items()}

def preprocess(img):
    img = img.resize((224,224)).convert('RGB')
    arr = np.array(img).astype('float32')/255.0
    arr = np.expand_dims(arr, 0)
    return arr

PICKUPS_FILE = os.path.join(os.path.dirname(__file__), "pickups.json")
if not os.path.exists(PICKUPS_FILE):
    with open(PICKUPS_FILE, "w") as f:
        json.dump([], f)

@app.route('/infer', methods=['POST'])
def infer():
    if 'image' not in request.files:
        return jsonify({"error":"no image"}), 400
    f = request.files['image']
    img = Image.open(f.stream)
    x = preprocess(img)
    preds = model.predict(x)[0]
    idx = int(np.argmax(preds))
    result = {"class": inv_idx[idx], "confidence": float(preds[idx])}
    return jsonify(result)

@app.route('/pickups', methods=['GET','POST'])
def pickups():
    with open(PICKUPS_FILE) as f:
        data = json.load(f)
    if request.method == 'GET':
        return jsonify(data)
    new = request.get_json()
    data.append(new)
    with open(PICKUPS_FILE, "w") as f:
        json.dump(data, f, indent=2)
    return jsonify({"status":"created", "pickup": new})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
