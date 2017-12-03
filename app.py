import numpy as np 
from flask import Flask, abort, request, jsonify
#import cPickle as pickle

#my_model = pickle.load(open("model.pkl", "rb"))

app = Flask(__name__)

@app.route('/model', method=['POST'])
def make_a_prediction():

    data = request.get_json(force=true)
    prediction_request = null #/*** Insert parameters here ***/
    prediction_request = np.array(prediction_request)



    return jsonify("----- works ---")

if __name__ == '__main__':
    app.run(port=2345, debug = true)
