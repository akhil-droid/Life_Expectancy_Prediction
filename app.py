import flask
from collections.abc import Mapping
import pickle
import os

from model import main, inputs_for_pickle

app = flask.Flask(__name__)
country = ''
yr = 0


@app.route('/')
def hello_world():
    return flask.render_template("index.html")


@app.route('/predict', methods=['POST', 'GET'])
def login():
    global country, yr
    country = flask.request.form['Country']
    yr = flask.request.form['myRange']
    default_values = main(country, yr)
    return flask.render_template('index2.html', default_values=default_values)


@app.route('/predict2', methods=['POST', 'GET'])
def my():
    l = list()
    global country, yr
    l.append(flask.request.form['am'])
    l.append(flask.request.form['bmi'])
    l.append(flask.request.form['dp'])
    l.append(flask.request.form['haa'])
    l.append(flask.request.form['tn'])
    l.append(flask.request.form['icr'])
    l.append(flask.request.form['sc'])
    result = inputs_for_pickle(country, yr, l)
    return flask.render_template('login.html', result=result)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))

    if port == 5000:
        app.debug = True

    app.run(host='0.0.0.0', port=port)
