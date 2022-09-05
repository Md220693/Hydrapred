
import tensorflow as tf
import streamlit as st
import pandas as pd
import numpy as np
import cv2
from keras_preprocessing.image import load_img, img_to_array
from PIL import Image, ImageOps

def preprocess_image(file):
    #Image = load_img(path, target_size = (img_height, img_width))
    image = Image.open(file)
    image = tf.image.resize(image, (64, 64))
    a = img_to_array(image)
    a = np.expand_dims(a, axis = 0)
    a /= 255.
    return a

def import_and_predict(image_data, model):
    
        size = (64,64)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = (cv2.resize(img, dsize=(64, 64),    interpolation=cv2.INTER_CUBIC))/255.
        
        img_reshape = img_resize[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        answer = np.argmax(prediction, axis=1)
        return answer

model = tf.keras.models.load_model('CNN_aug_best_weights.h5')


file = st.file_uploader("Please upload an image file", type=["jpg", "png"])
#
if file is None:
    st.text("You haven't uploaded an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    
    
    dict = {"0" : "A" , "1" : "B" , "2" : "C", "3" : "D", "4" : "E", "5" : "F", "6" : "G", "7" : "H", "8" : "I", "9" : "J", "10" : "K", "11" : "L", "12" : "M", "13" : "N", "14" : "O", "15" : "P", "16" : "Q", "17" : "R", "18" : "S"}
    prediction = import_and_predict(image, model)
    result = dict[str(prediction[0])]
    st.write(result)
