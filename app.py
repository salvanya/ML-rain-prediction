# app.py
import streamlit as st
import joblib
import pandas as pd
import os
from clases_y_funciones import columnas, columnas_seleccionadas, process_data, split_datasets, DataProcessor, NeuralNetworkPipeline, ClassificationPipeline 

# Se carga el pipeline del modelo.
path_dir=os.path.dirname(os.path.abspath(__file__))
pkl_path=os.path.join(path_dir, 'rain_prediction_classification.pkl')
classification_pipeline = joblib.load(pkl_path)
pkl_path_reg=os.path.join(path_dir, 'rain_prediction_regression.pkl')
regression_pipeline = joblib.load(pkl_path_reg)

st.title('RainfallTomorrow and RainTomorrow Predictor Models')


def get_user_input():
    """
    esta función genera los inputs del frontend de streamlit para que el usuario pueda cargar los valores.
    Además, contiene el botón para hacer el submit y obtener la predicción.
    """
    input_dict = {}

    with st.form(key='my_form'):
        for feat in columnas:
            if feat in columnas_seleccionadas:
                if feat == 'wet_month':
                    answer = st.radio(
                            feat + '(is it rainy season?): select an option',
                            ["Yes",  "No"])
                else:
                    answer = st.radio(
                                feat + ': select an option',
                                ["Yes",  "No"])
                if answer == 'Yes':
                    input_value = float(1.0)
                else:
                    input_value = float(0.0)
            elif feat in ['Sunshine', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm']:
                if feat == 'Sunshine':
                    try:
                        input_value = st.number_input(f"Enter value for {feat}. Values allowed: 0 to 24", value=0.0, step=0.01, min_value=0.0, max_value=24.0)
                    except ValueError:
                        st.error("Please enter a valid number within the specified range.")
                        st.stop() 
                elif feat == 'Humidity9am' or feat == 'Humidity3pm':
                    try:
                        input_value = st.number_input(f"Enter value for {feat}. Values allowed: 0 to 100", value=0.0, step=0.01, min_value=0.0, max_value=100.0)
                    except ValueError:
                        st.error("Please enter a valid number within the specified range.")
                        st.stop() 
                elif feat == 'Pressure9am' or feat == 'Pressure3pm':
                    try:
                        input_value = st.number_input(f"Enter value for {feat}. Values allowed: 900 to 1200", value=900.0, step=0.01, min_value=900.0, max_value=1200.0)
                    except ValueError:
                        st.error("Please enter a valid number within the specified range.")
                        st.stop() 
                else:
                    try:
                        input_value = st.number_input(f"Enter value for {feat}. Values allowed: 0 to 8", value=0.0, step=0.01, min_value=0.0, max_value=8.0)
                    except ValueError:
                        st.error("Please enter a valid number within the specified range.")
                        st.stop() 
            else:
                    input_value = st.number_input(f"Enter value for {feat}", value=0.0, step=0.01)
            input_dict[feat] = float(input_value)
        submit_button = st.form_submit_button(label='Submit')

    return pd.DataFrame([input_dict], columns=columnas), submit_button


user_input, submit_button = get_user_input()


# When the 'Submit' button is pressed, perform the prediction
if submit_button:
    # Predict rainfall and class
    prediction_class = classification_pipeline.predict(user_input)[0]
    prediction_reg = regression_pipeline.predict(user_input)[0][0]

    # Display the prediction
    st.header("Predicted Class")
    if prediction_class == 1:
        prediction_class = ' tomorrow it will rain'
    else:
        prediction_class = ' tomorrow it will not rain'

    if prediction_reg < 0:
        prediction_reg = 0.0
    st.write(f'The model has resulted: {prediction_class}')
    st.write(f'The probable amount of rain for tomorrow is: {prediction_reg:.2f} mm')
    

st.markdown(
    """
        Fernández, Florencia
        Salvañá, Leandro
    """, unsafe_allow_html=True
)