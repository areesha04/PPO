from tempfile import template
from flask import Flask, render_template, request
from flask_restful import Api
from flask_cors import CORS
import requests

app = Flask(__name__, template_folder='./template', static_folder='./static')

CORS(app)
api = Api(app)


@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/analysis')
def analysis():
    return render_template('analysis.html')


@app.route('/optimization')
def optimize():

    output=requests.get('http://127.0.0.1:5000/machine_info',allow_redirects=True)
    open('machine_info.json', 'wb').write(output.content)
    response = ''
    return render_template('optimization.html', result=response)


@app.route('/optimize', methods=['POST'])
def optimization():
    #response={}
    data = {'material_type': request.form.get('material_type'), 'quantity': float(request.form.get('quantity')),
            'deadline_days': float(request.form.get('deadline_days'))}
    resp=requests.post("http://127.0.0.1:5000/optimization",json=data)
    return render_template('optimization.html',result=resp.text)


@app.route('/forecast')
def forecast():
    return render_template('forecast.html')




if __name__ == '__main__':
    app.run()
