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

global model 
model = init()

'''
Converts the base64 image to a JPG that will
be later read for prediction by the trained model
'''
def convertImage(imgData1):
    imgstr = re.search(b'base64,(.*)',imgData1).group(1)
    with open('output.jpg','wb') as output:
         output.write(base64.b64decode(imgstr))

@app.route('/')
def home():
    response = "Route index for Survenet model API"
    return jsonify(result= response)

'''
App route for API to recieve payload
from SurveyApp 

A POST method is accepted and handled, returning
the prediction back to the app
'''
@app.route('/model',methods=['GET','POST'])
def make_a_prediction():
    response = ("undefined")
    print('1: Request made by surveyapp')

    #Get data from the request and send it to
    #be converted to an image
    imageData = request.get_data()
    convertImage(imageData)
    print('2: Image was converted')

    #Change the image to an array
    test_image = image.load_img('output.jpg',target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    print('3: image converted to array')

    #Expand the dimensions of the array 
    test_image = np.expand_dims(test_image, axis = 0)
    print('4: Dimension added')

    #Make a prediction with the image
    result = model.predict(test_image)

    #A threshold is set for the models output
    #0 meaning sad and 1 meaning happy
    if result[0][0] > 0.5:
        response = 'Happy'
        print(result[0][0])
    else:
        response = 'Sad'
        print(result[0][0])

    #model response is then parsed to JSON and returned to app
    return jsonify(response)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 6000))
    app.run(host='0.0.0.0', port=port)
