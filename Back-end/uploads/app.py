import os
import time
from flask_cors import CORS
from flask import Flask, request, json, send_file
from predictor import predictor
import constants as constant
app = Flask(__name__)
CORS(app)


@app.route('/sendData', methods=['POST'])
def create():
    try:
        request_data = json.loads(request.data)
        message = predictor(request_data)
        return message
    except:
        return "Request Failed!"


@app.route('/upload', methods=["POST"])
def upload():
    if request.method.lower() == 'post':
        if request.files:
            data_file = request.files["file"]
            data_file.save(os.path.join(
                constant.path, constant.data_dir, data_file.filename))
            print(data_file.filename.upper(),
                  " has been saved to data directory")
            return({"message": "File uploaded!"})
        return({"message": "Error! No file received!"})
    return({"message": "Method name not matched!"})


@app.route('/train')
def train():
    os.system("python data_cleansing.py")
    time.sleep(1)
    os.system("python lstm.py")
    time.sleep(1)
    os.system("python fragmentation.py")
    time.sleep(1)
    os.system("python ann.py")
    time.sleep(1)
    return({"message": "Training Completed!<br /><br /> Data Cleansing - Done ✔<br /> Forecasting - Done ✔<br /> Data Fragmentation - Done ✔<br /> Model Compilation - Done ✔"})


@app.route("/get-csv") 
def get_csv():
    try:
        return send_file(os.path.join(constant.path, constant.data_dir, constant.template_filename), as_attachment=True)
    except:
        return {'message': "Error! Something went wrong."}


if __name__ == "__main__":
    app.run()
