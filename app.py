import sys
from src.exception import CustomException
# from src.pipeline.prediction_pipeline import SinglePrediction
from flask import Flask, render_template,request

import traceback

app = Flask(__name__)

@app.route('/',methods= ['GET'])
def land():
    return render_template("index.html")

@app.route('/index',methods= ['GET','POST'])
def index():
    try:
        if request.method == 'POST':
            english = request.form.get('english')
            german = request.form.get('german')
            chinese = request.form.get('chinese')
            italian = request.form.get('italian')
            japanese = request.form.get('japanese')
            french = request.form.get('french')
            korean = request.form.get('korean')
            russian = request.form.get('russian')
            language_list = {
                'english':english,
                'german':german,
                'chinese':chinese,
                "italian":italian,
                'japanese':japanese,
                'french':french,
                'korean':korean,
                'russian':russian}
            data = dict()
            for key, value in language_list.items():
                if value == 'on':
                    data[key] = key
        return render_template("reviews.html",data=data)
    except Exception as e:
        raise CustomException(e,sys)

@app.route('/result',methods= ['GET','POST'])
def result():
    try:
        if request.method == 'POST':
            input_text = request.form.getlist('language')
            # obj = SinglePrediction()
            # result = obj.predict(input_text)
            # print(result)
            # return render_template("result.html", results = result)
            return render_template("result.html")
    except Exception as e:
        traceback.print_exc() 
        raise CustomException(e,sys)
    

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000, debug=True)