import numpy as np 
import keras.models
from keras.models import model_from_json
from scipy.misc.pilutil import imread, imresize, imshow
import tensorflow as tf
import os

'''
Initialise the model. 

Load the model architecture from the JSON file
Initialise the weights using saved_weights.h5
compile the model and return it
'''

def init():
    #Open the json model file and read
    json_file = open('model/model.json', 'r')
    loaded_json = json_file.read()
    json_file.close() 
    #initialize model
    loaded_model = model_from_json(loaded_json)
    loaded_model.load_weights('model/saved_weights.h5')
    print("Model succesffully loaded")
    #evaluate
    loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return loaded_model
