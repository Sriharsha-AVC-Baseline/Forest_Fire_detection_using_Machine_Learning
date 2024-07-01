from flask import Flask,render_template,url_for,redirect,request
from sklearn.linear_model import LogisticRegression
import pickle

loaded_model = pickle.load(open('Predict.sav', 'rb'))
app = Flask(__name__)

@app.route('/')
def welcome():
    return render_template("index.html")
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

if __name__=='__main__':
    app.run(debug=True)