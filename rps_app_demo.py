# Import the necessary modules
import streamlit as st
import tensorflow as tf
from PIL import Image
import tensorflow_hub as hub
import random
import json
import time

def show_message():
    st.write("Waiting for the ViTPose to run on the image")
    time.sleep(5)
    st.write("")

# Define a function to classify an image using a trained model
def classify_image(image, model):
    # Pre-process the image for the model
    image = tf.image.resize(image, (224, 224))  # Resize the image
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)  # Pre-process the image for the model
    image = image[None, ...]  # Add a batch dimension to the image

    # Use the model to classify the image
    predictions = model.predict(image)

    # Return the predicted class and probability
    return tf.argmax(predictions[0]), tf.nn.softmax(predictions)[0, tf.argmax(predictions[0])]

# Load the trained model
# model = tf.keras.models.load_model('model.h5')
model = tf.keras.models.load_model('my_model.hdf5', custom_objects={'KerasLayer':hub.KerasLayer})

st.write("""
         # Prediction for pose
         """
         )

st.write("This is a simple image classification web app to predict your pose and rate it")

# Use the file_uploader widget to upload an image
uploaded_file = st.file_uploader('Choose an image to classify:')


# Display the uploaded image
# working
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image')

    # Use the classify_image function to classify the uploaded image
    class_index, class_probability = classify_image(image, model)

    index_list = ["Bench", "Deadlift", "Squat"]
    # Display the predicted class and probability
    st.write(f'Predicted class: {index_list[class_index]}')
    st.write(f'Class probability: {class_probability:.2f}')

    # wait message
    show_message()
  
    image = Image.open("C:\\Users\\amart50\\Documents\\Test_demo\\PoseEstimations\\vis_17.jpg")
    st.image(image, caption='ViTPose output')
    st.write("According to pose estimation, your wrist joints are too closes")
