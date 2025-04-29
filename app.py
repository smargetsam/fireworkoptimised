
import streamlit as st
import os
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2 as cv
import joblib
from tqdm.notebook import tqdm
import tensorflow as tf
from tensorflow import keras
import zipfile
# Load the model within your Streamlit app

# Create three columns
col1, col2, col3 = st.columns([1,2,1])
with col2:

  def showpic(image):
    plt.figure()
    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    plt.show()


  st.title("Let's check for fire!")
  st.write("Upload an image to determine if there is fire, smoke, smoke and fire, or neither fire nor smoke. This is the optimised version of the  app uses a lightweight convolutional neural network to read the image and make a prediction.The performance of this model was enhanced by data augmentation (blur).")

  uploaded_files = st.file_uploader(
      "Choose an image file", accept_multiple_files=True
  )

  #Create a directory to store uploaded images if it doesn't exist
  if not os.path.exists("uploaded_images"):
      os.makedirs("uploaded_images")

  for uploaded_file in uploaded_files:
      file_bytes = uploaded_file.read()
      file_bytesname = uploaded_file.name
      #Convert the uploaded file (bytes) to a NumPy array
      file_bytes = np.asarray(bytearray(file_bytes), dtype=np.uint8)
      # Decode the image
      image = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
      # Save the image
      cv.imwrite(os.path.join("uploaded_images", uploaded_file.name), image)
      original = cv.cvtColor(image, cv.COLOR_BGR2RGB)
      st.write("Your upload:")
      st.image(original, caption=f"{file_bytesname} picture uploaded", width = 400)

  
  # Unzip the model
  with zipfile.ZipFile("bestmodel.zip", 'r') as zip_ref:
    zip_ref.extractall("model_dir")

  # Path to your model after extraction
  model_path = os.path.join("model_dir", "bestmodel.h5")

  # Load the model
  model = tf.keras.models.load_model(model_path)







  
  fire_label_mapping = {0: 'smoke', 1: 'fire', 2: 'bothfireandsmoke', 3: 'neitherfirenorsmoke'}


  def preprocess_image(image):
      # Resize the image
      image = cv.resize(image, (128, 128))
      # Convert to NumPy array and normalize
      image = image.astype(np.float32) / 255.0
      # Assuming you want to expand dimensions for single image prediction
      image = np.expand_dims(image, axis=0)
      return image





  for uploaded_file in (uploaded_files):
      file_bytes = uploaded_file.read()
      # ... (other existing file handling code) ...

      # Preprocess the image
      processed_image = preprocess_image(image)


      # Display "Please hold" message
      with st.spinner('Please hold while fire detection is in progress...'):
          # Make prediction
          prediction = model.predict(processed_image)
          predicted_class = np.argmax(prediction, axis=-1)
          final_prediction = [fire_label_mapping[label] for label in predicted_class]

      # Display the prediction (after processing is done)
      if final_prediction == ['fire']:
          st.subheader("Fire detected!")
      elif final_prediction == ['smoke']:
          st.subheader("Smoke detected!")
      elif final_prediction == ['bothfireandsmoke']:
          st.subheader("Fire and smoke detected!")
      else:
          st.subheader("No fire or smoke detected.")





  
