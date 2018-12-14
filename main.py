from flask import Flask, request, flash, redirect, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
import glob
import numpy as np
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
    file.save("D:\\nkasturi122817\\axisbank\\Axis_SigVerify-master\\data\\" + filename)
    return jsonify(filename)


@app.route("/retrieve", methods=['GET'])
def retrieve_id():
    files = glob.glob("D:\\nkasturi122817\\axisbank\\Axis_SigVerify-master\\data\\*.*")
    for i in range(len(files)):
        files[i] = files[i].split('\\')[-1].split('.')[0]
    return jsonify(files)


@app.route("/verify", methods=['POST'])
def verify_customer():
    with graph.as_default():
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        cus_info = request.form
        cus_id = cus_info['id']
        filename = cus_id+"_verf.png"
        file.save("D:\\nkasturi122817\\axisbank\\Axis_SigVerify-master\\dataVerify\\" + filename)
        verf_img = "D:\\nkasturi122817\\axisbank\\Axis_SigVerify-master/dataVerify/" + cus_id+"_verf.png"
        prep_verf_img = preprocess_image(verf_img)
        img2 = prep_image.img_to_array(prep_verf_img)
        org_img_paths = glob.glob("D:\\nkasturi122817\\axisbank\\Axis_SigVerify-master/data/" + cus_id + "*.png")
        distance = 0
        num_smpls = len(org_img_paths)
        input_shape = (155, 220, 1)
        X1 = np.empty((num_smpls, *input_shape))
        X2 = np.empty((num_smpls, *input_shape))
        for i, org_img in zip(range(num_smpls), org_img_paths):
            prep_org_img = preprocess_image(org_img)
            img1 = prep_image.img_to_array(prep_org_img)
            X1[i,] = img1
            X2[i,] = img2
        distance  = model.predict([X1, X2])
        print(distance)
        avg_distance = (distance.sum()/len(org_img_paths))
        result_json = jsonify({"distance": avg_distance.tolist()})
    return result_json

@app.route("/finetune", methods=['POST'])
def finetune_model():
    with graph.as_default():
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        cus_info = request.form
        cus_id = cus_info['id']
        filename = cus_id+"_verf.png"
        file.save("D:\\nkasturi122817\\axisbank\\Axis_SigVerify-master\\dataVerify\\" + filename)
        verf_img = "D:\\nkasturi122817\\axisbank\\Axis_SigVerify-master/dataVerify/" + cus_id+"_verf.png"
        prep_verf_img = preprocess_image(verf_img)
        img2 = prep_image.img_to_array(prep_verf_img)
        org_img_paths = glob.glob("D:\\nkasturi122817\\axisbank\\Axis_SigVerify-master/data/" + cus_id + "*.png")
        distance = 0
        num_smpls = len(org_img_paths)
        input_shape = (155, 220, 1)
        X1 = np.empty((num_smpls, *input_shape))
        X2 = np.empty((num_smpls, *input_shape))
        for i, org_img in zip(range(num_smpls), org_img_paths):
            prep_org_img = preprocess_image(org_img)
            img1 = prep_image.img_to_array(prep_org_img)
            X1[i,] = img1
            X2[i,] = img2
        distance  = model.fit([X1, X2],[0])
        print(distance)
        avg_distance = (distance.sum()/len(org_img_paths))
        result_json = jsonify({"distance": avg_distance.tolist()})
    return result_json


if __name__ == '__main__':
    model = build_model()
    graph = tf.get_default_graph()
    app.run(debug=True)
