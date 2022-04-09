import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf
import pandas as pd

from PIL import Image

import yaml
from yaml import SafeLoader

with open('launch.yaml') as f:
    
    data = yaml.load(f, Loader=SafeLoader)

IMAGE_WIDTH = data['IMAGE_WIDTH']
IMAGE_HEIGHT = data['IMAGE_HEIGHT']
IMAGE_DEPTH = data['IMAGE_DEPTH']
MODEL_DIR = data['MODEL_DIR']

dict_class = {0 : 'ATR', 1 : 'Airbus', 2 : 'Antonov', 3 : 'Beechcraft', 4 : 'Boeing', 
              5 :'Bombardier Aerospace',6 : 'British Aerospace', 7 : 'Canadair', 
              8 : 'Cessna', 9 : 'Cirrus Aircraft', 10 : 'Dassault Aviation', 
              11 : 'Dornier', 12 : 'Douglas Aircraft Company', 13 : 'Embraer', 
              14 : 'Eurofighter', 15 : 'Fairchild', 16 : 'Fokker', 17 : 'Gulfstream Aerospace', 
              18 : 'Ilyushin', 19 : 'Lockheed Corporation',20 : 'Lockheed Martin', 
              21 : 'McDonnell Douglas', 22 : 'Panavia', 23 : 'Piper', 24 : 'Robin',
              25 : 'Saab', 26 : 'Supermarine', 27 : 'Tupolev', 28 : 'Yakovlev', 29 : 'de Havilland'}


def load_image(path):
    """Load an image as numpy array
    """
    return plt.imread(path)
    

def predict_image(path, model):
    
    """Predict plane identification from image.
    
    Parameters
    ----------
    path (Path): path to image to identify
    model (keras.models): Keras model to be used for prediction 
    
    Returns
    -------
    Predicted class
    """
    images = np.array([np.array(Image.open(path).resize((IMAGE_WIDTH, IMAGE_HEIGHT)))])
    prediction_vector = model.predict(images)
    predicted_class = np.argmax(prediction_vector, axis=1)[0]
    proba = np.round(prediction_vector[:,predicted_class][0],2)*100
    proba = round(proba,2)
  
    df = pd.DataFrame()   
    df.index = dict_class.values()
    df["proba"] = prediction_vector[0]
    df.sort_values(by = "proba", ascending = True).plot.barh(y = "proba", title = "Répartition des probabilités par classe")
    st.pyplot(fig = plt)
    
    
    return predicted_class, proba


def load_model(path):
    """Load tf/Keras model for prediction
    """
    return tf.keras.models.load_model(path)
    

model = load_model(MODEL_DIR + '/manufacturer.h5')
model.summary()

st.title("Identification d'avion")

uploaded_file = st.file_uploader("Charger une image d'avion") #, accept_multiple_files=True)#

if uploaded_file:
    loaded_image = load_image(uploaded_file)
    st.image(loaded_image)

predict_btn = st.button("Identifier", disabled=(uploaded_file is None))
if predict_btn:
    prediction, proba = predict_image(uploaded_file, model)
    st.write(f"L'avion est un '{dict_class[prediction]}' à {proba}% de probabilité.")
    # Exemple si les f-strings ne sont pas dispo.
    #st.write("C'est un : {}".format(prediction)
