import requests
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for
import os
from werkzeug.utils import secure_filename
from tensorflow.python.keras.backend import set_session
from cloudant.client import Cloudant

client = Cloudant.iam('1245c35d-e88f-4b11-82d6-ccee61b098ff-bluemix',
                      '0PYL_N3AdMnURBW1SvQPic81f0-4XdUQdkf0L1eiNAl9', connect=True)
my_database = client.create_database('my_database')


app = Flask(__name__, template_folder='../templates',
            static_folder='../static')

basepath = os.path.dirname(__file__)
upload_path = os.path.join(os.path.dirname(basepath), 'static', 'uploads')

veg_model = load_model(os.path.join(basepath, "Vegetable.h5"))
fruit_model = load_model(os.path.join(basepath, "Fruit.h5"))


@app.route("/home")
def home():
    return render_template('home.html')


@app.route("/")
@app.route("/login")
def login():
    return render_template('login.html')


@app.route("/afterLogin", methods=['POST'])
def afterLogin():
    userid = request.form['_id']
    password = request.form['password']

    query = {'_id': {'$eq': userid}}

    docs = my_database.get_query_result(query)

    if(len(docs.all()) == 0):
        return render_template('login.html',msg="User name not found")
    else:
        if((userid == docs[0][0]['_id'] and password == docs[0][0]['password'])):
            return redirect(url_for('home'))
        else:
            return render_template('login.html',msg="Invalid User")


@app.route("/register")
def register():
    return render_template('register.html')


@app.route("/afterRegister", methods=['POST'])
def afterRegister():
    id = request.form['_id']
    name = request.form['name']
    email = request.form['email']
    password = request.form['password']
    confirmPassword = request.form['confirmPassword']
    data = {
        '_id': id,
        'name': name,
        'email': email,
        'password': password,
        'confirmPassword': confirmPassword,
    }

    query = {'_id': {'$eq': data['_id']}}
    docs = my_database.get_query_result(query)

    if(len(docs.all()) == 0):
        if(password == confirmPassword):
            url = my_database.create_document(data)
            return render_template('login.html')
        else:
            return render_template('register.html',msg="Password not matched")

    else:
        return render_template('login.html',msg="You are already a user. please Login !!!")


@app.route("/predict")
def predict():
    return render_template("predict.html")


@app.route("/prediction", methods=['Get', 'POST'])
def prediction():
    if request.method == 'POST':
        f = request.files['image']
        file_path = os.path.join(upload_path, secure_filename(f.filename))
        f.save(file_path)
        img = image.load_img(file_path, target_size=(128, 128))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        plant = request.form['plant']
        if(plant == "vegetable"):
            pred = veg_model.predict(x)
            print(pred)
            print("Prediction is ",np.argmax(pred))
            df = pd.read_excel(os.path.join(basepath, 'precautions-veg.xlsx'))
            prec = df.iloc[np.argmax(pred),0]
            return render_template("result.html", msg=prec)
        else:
            pred = fruit_model.predict(x)
            print(pred)
            print("Prediction is ",np.argmax(pred))
            df = pd.read_excel(os.path.join(basepath, 'precautions-fruit.xlsx'))
            prec = df.iloc[np.argmax(pred),0]
            return render_template("result.html", msg=prec)


if __name__ == "__main__":
    app.run(debug=False)
