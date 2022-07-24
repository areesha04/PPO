import os
import json
import time
from flask import Flask, render_template, request, send_file, jsonify
import requests
from werkzeug.utils import secure_filename
from flask_restful import Api
from flask_cors import CORS
from optimization import suggestion


app = Flask(__name__)

CORS(app)
api = Api(app)

upload_folder = "uploads/"

if not os.path.exists(upload_folder):
    os.mkdir(upload_folder)

extensions = ['xlsx']
app.config['UPLOAD_FOLDER'] = upload_folder


def check_extension(filename):
    return filename.split('.')[-1] in extensions


@app.route('/upload', methods=['GET', 'POST'])
def uploadfile():

    if request.method == 'POST':
        f = request.files['file']

        if check_extension(f.filename):

            f.save(os.path.join(
                app.config['UPLOAD_FOLDER'], secure_filename("train_file.xlsx")))
            return ({"message": 'successfully uploaded !'})
        else:
            return ({"message": 'Please Upload Excel file!'})


@app.route('/download_sample', methods=['GET'])
def download_sample_file():
    return send_file('production.xlsx', as_attachment=True)


@app.route('/train')
def forecasting():
    try:
        if os.path.exists("./uploads/train_file.xlsx"):
            os.system("python lstm.py")
            time.sleep(1)
            os.remove("./uploads/train_file.xlsx")
            return({"message": "Forecasting - Done ✔ "})
        else:
            return({"message": "Please upload training file first!"})
    except Exception as e:
        return({"message": str(e)})


@app.route('/forecast_data')
def forecast_chart():
    # return send_file('forecast.json', as_attachment=True)
    with open("forecast.json", "r") as file:
        data = json.loads(file.read())
        return({"message": data["Forecast"]})


@app.route('/download_forecast', methods=['GET'])
def download_forecast_file():
    return send_file('forecast.xlsx', as_attachment=True)


@app.route('/machine_info', methods=['GET'])
def machine_info():
    return send_file('machine_info.json', as_attachment=True)


@app.route('/optimization', methods=['POST'])
def optimization():
    data = request.get_json()
    material_type = data.get('material_type', '')
    quantity = float(data.get('quantity', ''))
    deadline_days = float(data.get('deadline_days', ''))
    response = suggestion(material_type, quantity, deadline_days)
    response = ''.join(response)
    return response


if __name__ == '__main__':
    app.run()
