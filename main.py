from flask import Flask, request, flash, redirect, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
import sys
import _pickle as pickle
import os
import cv2
from keras.models import load_model
from preprocess_image import preprocess_image
from keras.preprocessing import image as prep_image
app = Flask(__name__)
app.secret_key = 'XXX'
CORS(app)
model = load_model("model_prep_acc.h5")

@app.route("/newcus", methods=['POST'])
def newcus_upload():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    cus_info = request.get_json()
    cus_id = cus_info['cus_id']

    filename = cus_id +"."+"png"
    file.save(filename)
    return True

@app.route("/verify", methods=['POST'])
def newcus_upload():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    cus_info = request.get_json()
    cus_id = cus_info['cus_id']
    filename = cus_id+"_verf.png"
    file.save(filename)
    org_img = cv2.imread(cus_id +".png")
    prep_org_img = preprocess_image(org_img)
    img1 = prep_image.img_to_array(prep_org_img)
    verf_img = cv2.imread(cus_id+"_verf.png")
    prep_verf_img = preprocess_image(verf_img)
    img2 = prep_image.img_to_array(prep_verf_img)
    distance = model.predict([img1, img2])
    result_json = jsonify({"distance":distance, "file": org_img})
    return result_json



if __name__ == '__main__':
    app.run()
