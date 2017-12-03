import numpy as np 
from flask import Flask, abort, request, jsonify
import os as os 
#import cPickle as pickle

#my_model = pickle.load(open("model.pkl", "rb"))
app = Flask(__name__)

@app.route('/')
def home():
    response = "Route index for Survenet model API"
    return jsonify(result= response)

@app.route('/model',methods=['POST'])
def make_a_prediction():

    # data = request.get_json(force=true)
    # prediction_request = null #/*** Insert parameters here ***/
    # prediction_request = np.array(prediction_request)

    response = "IM WORKING"

    return jsonify(result = response)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
