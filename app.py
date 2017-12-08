import numpy as np 
import os as os 
import keras.models
import sys
import re
sys.path.append(os.path.abspath('./model'))
from load import *
from scipy.misc.pilutil import imsave, imread, imresize
from flask import Flask, abort, request, jsonify

app = Flask(__name__)

global model, graph
model, graph = init()

def convertImage(imgData1):

    print('======= GOT INTO CONVERT IMAGE=========')


    imgstr = re.search(r'base64,(.*)',imgData1).group(1)
    with open('output.png','wb') as output:
        output.write(imgstr.decode('base64'))


@app.route('/')
def home():
    response = "Route index for Survenet model API"
    return jsonify(result= response)

@app.route('/model',methods=['POST'])
def make_a_prediction():
    print('======= 1 Request made by surveyapp =========')

    imageData = request.get_data()
    convertImage(imageData)

    print('======= 2 GOT PAST CONVERTIMAGE =========')

    x = imread('output.png',mode='L')

    print('======= 3 =========')
    
    x = np.invert(x)
    
    print('======= 4 =========')

    x = imresize(x, 64, 64)

    print('======= 5 =========')

    x = x.reshape(1, 64, 64, 1)

    print('======= 6 =========')

    with graph.as_result():
        out = model.predict(x)
        response = np.array_str(np.argmax(out))
        
        print('======= 7 =========')

        print('============RESPONSE' + response)
        return response.read().decode('utf-8')

 

    # print('-------LOADED MODEL----------')
    # loaded_model.load_weights("/output/out.h5")
    # loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # score = loaded_model.evaluate(X, Y, verbose=0)
    # print("Loaded model from disk")
    # print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
    # data = request.get_json(force=true)
    # prediction_request = null #/*** Insert parameters here ***/
    # prediction_request = np.array(prediction_request)
    # response = "IM WORKING"

    # return jsonify(result = response)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 6000))
    app.run(host='0.0.0.0', port=port)
