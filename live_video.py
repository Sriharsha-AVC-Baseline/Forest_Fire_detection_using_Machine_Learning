from flask import Flask,render_template,request,Response
import cv2
import matplotlib.pyplot as plt
import seaborn 
import tensorflow as tf
from tensorflow import keras
import pathlib
import glob
import numpy
import pandas
import os
from flask_wtf import FlaskForm
from wtforms import FileField,SubmitField
from werkzeug.utils import secure_filename
from sklearn.linear_model import LogisticRegression
import pickle
import requests
import imutils
from playsound import playsound
from twilio.rest import Client
import keys
from datetime import datetime

loaded_model = pickle.load(open('Predict.sav', 'rb'))



app=Flask(__name__)

app = Flask(__name__)
app.config['SECRET_KEY']='secret'
app.config['UPLOAD_FOLDER'] = 'static\\files'

class UploadFileForm(FlaskForm):
    file=FileField("File")
    submit=SubmitField("Upload File")


font=cv2.FONT_HERSHEY_SIMPLEX
fontScale=1
color=(255,255,255)
thickness=2

model = new_model = tf.keras.models.load_model("AlexNet 8000imgs RGB_Binary.h5")
model.summary()
value='none'

@app.route('/1')
def welcome1():
    return render_template("predict.html")

@app.route('/Submit',methods=['POST','GET'])
def submit():
    if request.method =='POST':
        Humidity=int(request.form['Humidity'])
        Temperature=int(request.form['Temprature'])
        Oxygen=int(request.form['Oxygen'])
        result=loaded_model.predict([[Humidity,Temperature,Oxygen]])
        if result == 1:
            res ="Danger!!! There maybe Fire"
        else:
            res ="Forest is Safe"
    return render_template('predict.html',result=res)
url = "http://192.168.43.1:8080/shot.jpg"
def gen_frm(source,b):
    i=0
    font=cv2.FONT_HERSHEY_SIMPLEX
    if b==0:
        cap=cv2.VideoCapture(source)
        width= int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ht=height+50
        org=(50,ht)
    else :
        height=480
        width=640
        ht=height+100
        org=(100,400)
    while True:
        if b==0:
            success,frame=cap.read()
        else:
            img_resp = requests.get(url)
            img_arr = numpy.array(bytearray(img_resp.content), dtype=numpy.uint8)
            frame = cv2.imdecode(img_arr, -1)
            frame = imutils.resize(frame, width=640, height=480)
            success=True
        
        if not success:
            break

        else:
            cv2.imwrite('Frame.jpg', frame)
            test_array=[]
            img = cv2.imread('Frame.jpg')
            blankimg=numpy.zeros([width,height,3],numpy.uint8)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (227, 227)) #4000 model size has 180 180 BGR2YCrCb
                test_array.append(img)
            test_array = numpy.array(test_array)
            test_array = test_array/ 255
            pred = model.predict(test_array)
            print(pred.any())
            preds = pred.round(decimals=0).flatten()
            results=[]
            if pred<0.1 : #pred[0][0]>pred[0][1]
                #results.append(0)
                results=0
            elif pred>0.1:
                results=1
            font1 = {'family':'serif','color':'blue','size':20}
            
            txt='none'
            if results == 1:
                txt="Non-Fire"
                value = [255,0,0]        
            elif results == 0:
                txt="Fire"
                value = [0,0,250]
                playsound('o.wav')
                if i == 0 and source == 0:
                    i=1
                    now = datetime.now()
                    current_time = now.strftime("%I:%M:%S %p" )
                    client = Client(keys.account_sid,keys.auth_token)
                    message = client.messages.create(
                        body = "Fire !!!!\nFire spotted on : " + current_time,
                        from_=keys.twilio_number,
                        to="+917019056494")
                    print(message.body)
            vidres(txt)
            frame = cv2.copyMakeBorder(frame, 0, 100, 0, 0, cv2.BORDER_CONSTANT, None, value)
            txt=txt+" "+str(pred*100)
            frame = cv2.putText(frame,txt,org,font,fontScale,color,thickness,cv2.LINE_AA)
            ret,buffer=cv2.imencode('.jpg',frame)
            print(txt)
            frame=buffer.tobytes()
            
            yield(b'--frame\r\n'b'Content-Type:image/jpeg\r\n\r\n'+frame+b'\r\n')
     




@app.route('/lv')
def default():
    return render_template('live.html')

@app.route('/lvr')
def li_rm():
    return render_template('lv_rm.html')

@app.route('/liv')
def de():
    return render_template('pg2.html')

@app.route('/')
def mn():
    return render_template('index.html')

@app.route('/video')
def video(a=0,b=0):
    return Response(gen_frm(a,b),mimetype='multipart/x-mixed-replace;boundary=frame')

@app.route('/vidres')
def vidres(a=0,b=1):
    return Response(gen_frm(a,b),mimetype='multipart/x-mixed-replace;boundary=frame')


@app.route('/upload',methods=['GET','POST'])
def upld():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data
        print(file.filename)
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename)))
        path=("D:\\bunchofcodes-main\\Major project\\Flask\\static\\files\\" + file.filename)
        return video(path)
    return render_template('local.html',form=form)


@app.route('/res')
def res():
    return Response(txt)

if __name__=="__main__":
    app.run(debug=True)