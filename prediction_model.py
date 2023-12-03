import pandas as pd
import numpy as np
import streamlit as st
import pickle


loaded_model = pickle.load(open("C:/Users/every/OneDrive/Desktop/machine learning programs/trained_model.sav"))

input_data = (7.3,0.65,0,1.2,0.065,15,21,0.9946,3.39,0.47,10)

input_data_as_array = np.asarray(input_data)

input_data_reshaped = input_data_as_array.reshape(1, -1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
    print('Wine quality is not good')
else:
    print('Wine quality is good')

