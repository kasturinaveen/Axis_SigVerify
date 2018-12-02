from flask import Flask, request, flash, redirect, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
import glob
import sys
import _pickle as pickle
import os
import cv2
from keras.layers import Input, Lambda
from keras.models import load_model, Model
from preprocess_image import preprocess_image
from keras.preprocessing import image as prep_image
from SignatureVerification import build_model
import tensorflow as tf
app = Flask(__name__)
app.secret_key = 'XXX'
CORS(app)
# model = load_model("D:\\Work\\axis bank\\to_be_named\\model_prep_acc.h5")
# model = load_weights("D:\\Work\\axis bank\\to_be_named\\model_prep_acc.h5")
global graph


@app.route("/upload", methods=['POST'])
def newcus_upload():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    filename = secure_filename(file.filename)
    cus_info = request.form
    cus_id = cus_info['id']

    filename = cus_id + ".png"
    file.save("D:\\Work\\axis bank\\to_be_named\\data\\" + filename)
    return jsonify(filename)


@app.route("/retrieve", methods=['GET'])
def retrieve_id():
    files = glob.glob("D:\\Work\\axis bank\\to_be_named\\data\\*.*")
    for i in range(len(files)):
        files[i] = files[i].split('\\')[-1].split('.')[0]
    return jsonify(files)


@app.route("/verify", methods=['POST'])
def verify_customer():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    cus_info = request.form
    cus_id = cus_info['id']
    filename = cus_id+"_verf.png"
    file.save("D:\\Work\\axis bank\\to_be_named\\dataVerify\\" + filename)
    org_img = "D:/Work/axis bank/to_be_named/data/" + cus_id + ".png"
    prep_org_img = preprocess_image(org_img)
    img1 = prep_image.img_to_array(prep_org_img)
    verf_img = "D:/Work/axis bank/to_be_named/dataVerify/" + cus_id+"_verf.png"
    prep_verf_img = preprocess_image(verf_img)
    img2 = prep_image.img_to_array(prep_verf_img)
    with graph.as_default():
        distance = model.predict([[img1], [img2]])
    result_json = jsonify({"distance": distance.tolist()})
    return result_json


if __name__ == '__main__':
    model = build_model()
    graph = tf.get_default_graph()
    app.run(debug=True)
