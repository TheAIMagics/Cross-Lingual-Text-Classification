import sys
import traceback
from src.exception import CustomException
from src.pipeline.prediction_pipeline import SinglePrediction
from flask import Flask, render_template,request

app = Flask(__name__)

@app.route('/',methods= ['GET'])
    
def land():
    """
    This function defines a route for the root URL and returns the rendered template for the index.html
    file.
    :return: The function `land()` returns the rendered template of the `index.html` file.
    """
    return render_template("index.html")

@app.route('/index',methods= ['GET','POST'])
def index():
    """
    This function receives a POST request with language options, creates a dictionary of selected
    languages, and renders a template with the selected languages.
    :return: a rendered template "reviews.html" with a dictionary "data" as a parameter. The "data"
    dictionary is created by iterating through a "language_list" dictionary and adding the keys that
    have a value of 'on' to the "data" dictionary.
    """
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
    """
    This function takes input from a form, uses it to make a prediction, and returns the result to be
    displayed on a webpage.
    :return: a rendered HTML template with the results of a prediction made by an object of the class
    SinglePrediction. The results are passed as a parameter to the template with the name "results".
    """
    try:
        if request.method == 'POST':
            input_text = request.form.getlist('language')
            obj = SinglePrediction()
            result = obj.predict(input_text)
            print(result)
            return render_template("result.html", results = result)
    except Exception as e:
        traceback.print_exc() 
        raise CustomException(e,sys)
    

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000, debug=True)