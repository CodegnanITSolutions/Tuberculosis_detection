import numpy as np
import pandas as pd
import tensorflow as tf

import streamlit as st

import cv2
import sklearn

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from os import listdir
import tensorflow as tf


input_file_name = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"]).name

if input_file_name is not None:

	image = cv2.imread(input_file_name)
	st.image(image, caption="Chest X-Ray", use_column_width=True)
	image = cv2.resize(image,(64,64))
	image = image /255
	image = np.array([image])

	model = tf.keras.models.load_model('t_model.pt')
	prediction= model.predict(image)

	if prediction*100>=1:
		st.write("Tuberculosis Detected")
	else:
		st.write("Tuberculosis Not Detected")