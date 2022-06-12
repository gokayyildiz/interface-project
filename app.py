from flask import Flask, Response, request, render_template
from roberta_model_utils import *
from bert_model_utils import *
from distilbert_model_utils import *


import requests

app = Flask(__name__)


@app.route('/', methods=['GET','POST'])
def denemeSayfası():
    msg =''
    str1 =''
    str2 =''
    str =''
    bugr = ""
    questionr = ""
    enhancementr = ""
    if request.method == 'POST' and 'issueTitle' in request.form and 'issueBody' in request.form:
        # Create variables for easy access
        issueTitle = request.form['issueTitle']
        issueBody = request.form['issueBody']

        concanatated = issueTitle + " " + issueBody

        bugr, questionr, enhancementr = run_roberta_model(concanatated)
        result2 =""# run_bert_model(concanatated)
        result3 ="" # run_distilbert_model(concanatated)


        bugr= u"\u2713" if bugr==1 else ""
        questionr= u"\u2713" if questionr==1 else ""
        enhancementr= u"\u2713" if enhancementr==1 else ""
        str1= u"\u2713" if result2==1 else ""
        str2= u"\u2713" if result3==1 else ""



    return render_template('home.html',  bugr= bugr, questionr = questionr, enhancementr = enhancementr, bugb= str1,bugd= str2)


@app.route('/getaBugFrontend', methods=['GET','POST'])
def getRobertaBugFrontend():
    if request.get_json() is None:
        return Response("Body is empty!",status=400)

    data = request.get_json()
    if request.method == 'POST' and 'issueText' in data:
        result = run_roberta_model(data['issueText'])

        if result == 1:
            return "Bug"

        return "Non Bug"

    return 'Error'


@app.route('/getBertBug', methods=['GET'])
def getRobertaBug():
    
    issue_title = "the last run of the code version"
    issue_desc = "the last run trial of the code does not work in the local environment"

    concanatated = issue_title + " " + issue_desc
    
    result = run_bert_model(concanatated)

    if result == 1:
        return "Bug"

    return "Non Bug"





if __name__=='__main__':
    app.run(debug=True)