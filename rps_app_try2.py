# Import the necessary modules
import streamlit as st
import tensorflow as tf
from PIL import Image
import tensorflow_hub as hub
import random
import json

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
## working
# if uploaded_file is not None:
#     image = Image.open(uploaded_file)
#     st.image(image, caption='Uploaded Image')

#     # Use the classify_image function to classify the uploaded image
#     class_index, class_probability = classify_image(image, model)

#     index_list = ["Bench", "Deadlift", "Squat"]
#     # Display the predicted class and probability
#     st.write(f'Predicted class: {index_list[class_index]}')
#     st.write(f'Class probability: {class_probability:.2f}')

initial_json = '' #TODO: add this json path
# New Try with Vitpose
bench_dict = {}

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image')

    # Use the classify_image function to classify the uploaded image
    class_index, class_probability = classify_image(image, model)

    index_list = ["Bench", "Deadlift", "Squat"]
    # Display the predicted class and probability
    st.write(f'Predicted class: {index_list[class_index]}')
    st.write(f'Class probability: {class_probability:.2f}')

    filename = uploaded_file.name
    file_number = filename.split(".")[0]

    bench_dict[file_number] = image


    json_file = json.load(open(initial_json))

    # create JSON
    image_dict = {}
    image_dict['file_name'] = filename
    padded_id = [i.rjust(12,'0') for i in list( bench_dict.keys() ) ]
    height = [bench_dict[i].shape[0] for i in list( bench_dict.keys() ) ]
    width = [bench_dict[i].shape[1] for i in list( bench_dict.keys() ) ]
    image_dict['height'] = height
    image_dict['width'] = width
    image_dict['id'] = int(file_number)

    json_file['images'].append(image_dict)


    # json_dict["images"] = []
    # for i in range(len(bench_dict)):
    #     temp = {}
    #     temp["license"] = None
    #     temp["file_name"] = padded_id[i] +'.jpg'
    #     temp["coco_url"] = None
    #     temp["height"] = height[i]
    #     temp["width"] = width[i]
    #     temp["date_captured"] = None
    #     temp["flickr_url"] = None
    #     temp["id"] = int( list( bench_dict.keys() )[i])

    #     json_dict["images"].append(temp)

    # json_dict["annotations"] = []

    annotation = json_file['annotation']
    xxx = min(width, height)
    annotation['bbox'] = [[100, xxx, xxx, 100]]
    annotation['id'] = int(file_number)
    annotation['image_id'] = int(file_number)
    annotation['area'] = xxx*xxx

    # for i in range(len(bench_dict)):
    #     temp = {}
    #     random_l = list(random.sample(range(100, 300), 32) )
    #     temp["segmentation"] = [random_l]
    #     temp["num_keypoints"] = 17
    #     temp["iscrowd"] = 0
    #     temp["keypoints"] = [10,10,2] * 17
    #     temp["image_id"] = int( list( bench_dict.keys() )[i])
    #     xxx = min(width[i], height[i])
    #     temp["area"] = xxx * xxx

    #     l6_new =[100, xxx, xxx, 100]
    #     temp["bbox"] = l6_new #100, width[i]-100, 100, height[i]-100
    #     temp["category_id"] = 1
    #     temp["id"] = int( list( bench_dict.keys() )[i])
    #     json_dict["annotations"].append(temp)
    
    # json_object = json.dumps(json_dict, indent=4)


    with open(initial_json, "w") as outfile:
        json.dump(json_file, outfile, indent=4)

# python demo/top_down_img_demo.py configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192.py 
# C:\Users\mkingsl6\Desktop\vitpose-b-multi-coco.pth --img-root C:\Users\mkingsl6\Desktop\ViTPose-main\LiftingData\Deadlift --json-file C:\Users\mkingsl6\Desktop\ViTPose-main\output\deadlift_annotations.json --out-img-root C:\Users\mkingsl6\Desktop\output_deadlift

#TODO:
#import top_down_img_demo from ViTPose, 
#call demo class with (config, .pth, root, json, out-root)
