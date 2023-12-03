import numpy as np
import pickle
import streamlit as st

# loading a saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))


# creating a function for Prediction
def prediction(input_data):
    
    input_data = ()

    input_data_as_array = np.asarray(input_data)

    input_data_reshaped = input_data_as_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return 'Wine quality is not good'
    else:
        return 'Wine quality is good'
    

def main():

    # giving a title
    st.title('Wine Quality Web App')

    # user input data

    fixed_acidity = st.text_input("Fixed acidity Percentage")
    volatile_acidity = st.text_input("Volatile acidity Percentage")
    citric_acid = st.text_input("Citric acid Percentage")
    residual_sugar = st.text_input('Residual Sugar Percentage')
    chlorides = st.text_input("chloride Percentage")
    free_sulphur_dioxide = st.text_input('Free Sulphur Dioxide Percentage')
    total_sulphur_dioxide = st.text_input('Total Sulphur Dioxide Percentage')
    density = st.text_input('Density Percentage')
    pH = st.text_input("pH Percentage")
    sulphates = st.text_input('Sulphate Percentage')
    alcohol = st.text_input('Alcohol Percentage')


    # code for prediction
    quality = ''

    # creating a button for prediction
    if st.button('Wine Quality Test Result'):
        quality = prediction([fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulphur_dioxide, total_sulphur_dioxide, density, pH, sulphates, alcohol])


    st.success(quality)


if __name__ == '__main__':
    main()