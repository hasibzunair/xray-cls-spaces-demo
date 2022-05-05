"""Streamlit web app for radiological condition prediction from chest X-ray images"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True
import numpy as np
import time
import cv2
import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
from keras.utils.data_utils import get_file
from tensorflow import keras
from PIL import Image
from explain import get_gradcam

st.set_option("deprecation.showfileUploaderEncoding", False)

IMAGE_SIZE = 224

classes = [
    'Normal', # No Finding
    'Enlarged- \nCardiomediastinum',
    'Cardiomegaly',
    'Lung Opacity',
    'Lung Lesion',
    'Edema',
    'Consolidation',
    'Pneumonia',
    'Atelectasis',
    'Pneumothorax',
    'Pleural Effusion',
    'Pleural Other',
    'Fracture',
    'Support Devices'
]


@st.cache(allow_output_mutation=True)
def cached_model():
    URL = "https://github.com/hasibzunair/cxr-predictor/releases/latest/download/CheXpert_DenseNet121_res224.h5"
    weights_path = get_file(
               "CheXpert_DenseNet121_res224.h5",
               URL)
    model = load_model(weights_path, compile = False)
    return model


def preprocess_image(uploaded_file):
    # Load image
    img_array = np.array(Image.open(uploaded_file))
    # Normalize to [0,1]
    img_array = img_array.astype('float32')
    img_array /= 255
    # Check that images are 2D arrays
    if len(img_array.shape) > 2:
        img_array = img_array[:, :, 0]
    # Convert to 3-channel
    img_array = np.stack((img_array, img_array, img_array), axis=-1)
    # Convert to array
    img_array = cv2.resize(img_array, (IMAGE_SIZE, IMAGE_SIZE))
    return img_array


def make_prediction(file):
    # Preprocess input image
    image = preprocess_image(file)
    # Add batch axis
    image = np.expand_dims(image, 0)
    # Predict
    predictions = model.predict(image)
    return predictions


# Get model
model = cached_model()

if __name__ == '__main__':
    
    logo = np.array(Image.open("media/logo_rs.png"))
    st.image(logo, use_column_width=True)

    st.write("""
    # AI Assisted Radiology Tool
    :red_circle: NOT FOR MEDICAL USE!
    
    This is a prototype application which demonstrates how artifical intelligence based systems can identify
    medical conditions from images. Using this tool, medical professionals can process an image to 
    confirm or aid in their diagnosis which may serve as a second opinion.
    
    The tool predicts the presence of 14 different radiological conditions from a given chest X-ray image. This is built using data from a large public
    [database](https://stanfordmlgroup.github.io/competitions/chexpert/).
    
    Questions? Email me at `hasibzunair@gmail.com`.

    If you continue, you assume all liability when using the system.
    Please upload a posterior to anterior (PA) view chest X-ray image file (PNG, JPG, JPEG)
    to predict the presence of the radiological conditions. Here's an example.
    """)

    example_image = np.array(Image.open("media/example.jpg"))
    st.image(example_image, caption="An example input.", width=100)

    uploaded_file = st.file_uploader("Upload file either by clicking 'Browze Files' or drag and drop the image.", type=None)

    if uploaded_file is not None:
        # Uploaded image
        original_image = np.array(Image.open(uploaded_file))

        st.image(original_image, caption="Input chest X-ray image", use_column_width=True)
        st.write("")
        st.write("Analyzing the input image. Please wait...")

        start_time = time.time()

        # Preprocess input image
        image = preprocess_image(uploaded_file)
        image = np.expand_dims(image, 0)
        
        # Predict
        predictions = make_prediction(uploaded_file)

        st.write("Took {} seconds to run.".format(
            round(time.time() - start_time, 3)))
        
        # Convert probabilty scores to percent for easy interpretation
        predictions_percent = [x*100 for x in predictions[0]]

        df = pd.DataFrame({'classes' : classes, 'predictions' : predictions_percent})
        df = df.sort_values('predictions')

        # Top predicted class
        top_predicted_class = list(df['classes'])[-1]
        
        fig, ax = plt.subplots()
        ax.grid(False)
        ax.set_xlim(0, 1)
        ax.set_xticks([x for x in range(0,110, 10)])
        #ax.set_xticklabels(['Low','Average','High'])
        ax.tick_params(axis='y', labelcolor='r', which='major', labelsize=8)
        ax.barh(df['classes'], df['predictions'], color='green')
        ax.set_xlabel('Confidence (in percent)', fontsize=15)
        ax.set_ylabel('Radiological conditions', fontsize=15)
        st.pyplot(fig)

        st.write("""
        For more information about the top most predicted radiological finding,
        you can click on 'Get Heatmap' button which will highlight the most influential features 
        in the image affecting the prediction.        
        """)
        if st.button('Get Heatmap'):
            st.write('Generating heatmap for regions predictive of {}. Indicated by red.'.format(top_predicted_class))
            heatmap = get_gradcam(uploaded_file, model, "conv5_block16_2_conv", predictions)
            orig_heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
            # Convert to 3 channel if not
            if len(original_image.shape) > 2:
                original_image = original_image[:, :, 0]
            original_image = np.stack((original_image, original_image, original_image), axis=-1)
            st.image(np.concatenate((original_image,orig_heatmap),axis=1), caption="Input image + Heatmap ", use_column_width=True)