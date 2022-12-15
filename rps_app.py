import tensorflow as tf

# load_model_sample.py
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow_hub as hub

import cv2
from PIL import Image, ImageOps
import numpy as np




model = tf.keras.models.load_model('my_model.hdf5', custom_objects={'KerasLayer':hub.KerasLayer})
import streamlit as st



st.write("""
         # Prediction for pose
         """
         )

st.write("This is a simple image classification web app to predict your pose and rate it")
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])
####################################
def import_and_predict(image_data, model):
    img_tensor = image.img_to_array(image_data)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.          

    prediction = model.predict(img_tensor)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = image.load_img(file, target_size=(224, 2224))
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    
    if np.argmax(prediction) == 0:
        st.write("It is a Bench!")
    elif np.argmax(prediction) == 1:
        st.write("It is a Deadlift!")
    else:
        st.write("It is a scissor!")
    
    st.text("Probability (0: Bench , 1: Deadlift, 2: Squat")
    st.write(prediction)
    
    # image = Image.open(file)
    # st.image(image, use_column_width=True)
    # prediction = import_and_predict(image, model)
    
    # if np.argmax(prediction) == 0:
    #     st.write("It is a Bench!")
    # elif np.argmax(prediction) == 1:
    #     st.write("It is a Deadlift!")
    # else:
    #     st.write("It is a scissor!")
    
    # st.text("Probability (0: Bench , 1: Deadlift, 2: Squat")
    # st.write(prediction)





# def load_image(img_path, show=False):

#     img = image.load_img(img_path, target_size=(150, 150))
#     img_tensor = image.img_to_array(img)                    # (height, width, channels)
#     img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
#     img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

#     if show:
#         plt.imshow(img_tensor[0])                           
#         plt.axis('off')
#         plt.show()

#     return img_tensor


# if __name__ == "__main__":

#     # load model
#     model = load_model("my_model.h5")

#     # image path
#     img_path = '/media/data/dogscats/test1/3867.jpg'    # dog
#     #img_path = '/media/data/dogscats/test1/19.jpg'      # cat

#     # load a single image
#     new_image = load_image(img_path)

#     # check prediction
#     pred = model.predict(new_image)