import requests
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask,flash, request, render_template, redirect, url_for
import os
from werkzeug.utils import secure_filename
from tensorflow.python.keras.backend import set_session

app = Flask(__name__,template_folder='../templates',static_folder='../static')

basepath=os.path.dirname(__file__)
upload_path=os.path.join(os.path.dirname(basepath),'static','uploads')

veg_model = load_model(os.path.join(basepath,"Vegetable.h5"))
fruit_model = load_model(os.path.join(basepath,"Fruit.h5"))


@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html')

@app.route("/login")
def login():
    if request.method == 'POST':
        return render_template('login.html')
    else:
        return render_template('login.html')

@app.route("/register")
def register():
    if request.method == 'POST':
        return render_template('register.html')
    else:
        return render_template('register.html')

@app.route("/predict")
def predict():
    return render_template("predict.html")

@app.route("/prediction",methods=['Get','POST'])
def prediction():
    if request.method == 'POST':
        f=request.files['image']
        file_path=os.path.join(upload_path,secure_filename(f.filename))
        f.save(file_path)
        img=image.load_img(file_path,target_size=(128,128))
        x=image.img_to_array(img)
        x=np.expand_dims(x,axis=0)
        plant=request.form['plant']
        if(plant=="vegetable"):
            pred=veg_model.predict(x)
            df=pd.read_excel(os.path.join(basepath,'Precautions-veg.xlsx'))
            prec=df.iloc[np.argmax(pred),1]
            return render_template("result.html",msg=prec)
        else:
            pred=fruit_model.predict(x)
            df=pd.read_excel(os.path.join(basepath,'Precautions-fruit.xlsx'))
            prec=df.iloc[np.argmax(pred),1]
            return render_template("result.html",msg=prec)

if __name__ == "__main__":
    app.run(debug=True)