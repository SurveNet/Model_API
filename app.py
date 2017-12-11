import numpy as np 
import os as os 
import keras.models
import sys
import re
sys.path.append(os.path.abspath('./model'))
from load import *
from scipy.misc.pilutil import imsave, imread, imresize
from flask import Flask, abort, request, jsonify
import base64
from keras.preprocessing import image  


app = Flask(__name__)

global model, graph
model, graph = init()

def convertImage(imgData1):
    imgstr = re.search(b'base64,(.*)',imgData1).group(1)
    with open('output.jpg','wb') as output:
         output.write(base64.b64decode(imgstr))

@app.route('/')
def home():
    response = "Route index for Survenet model API"
    return jsonify(result= response)

@app.route('/model',methods=['GET','POST'])
def make_a_prediction():
    response = ("undefined")
    print('1: Request made by surveyapp')

    imageData = request.get_data()

    print(imageData)

    convertImage(imageData)
    print('2: Image was converted')

    test_image = image.load_img('output.jpg',target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    print('3: image converted to array')

    test_image = np.expand_dims(test_image, axis = 0)
    print('4: Dimension added')

    # with graph.as_default():
    result = model.predict(test_image)
    print(result[0][0])

    if result[0][0] > 0.5:
        response = 'Happy'
    else:
        response = 'Sad'
    return response 

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 6000))
    app.run(host='0.0.0.0', port=port)
