import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

classes=pd.read_excel("class.xlsx")

st.header("Crop Recommendation data")
stacking_classifier = joblib.load('stacking_classifier_model.pkl')

st.image("https://i.pinimg.com/564x/ec/ab/a3/ecaba3b64e853d7eda45658da3ebff50.jpg", use_column_width=True, width=400)

N_input = st.text_input("Enter N value:")
st.write("You entered:", N_input)

P_input = st.text_input("Enter P value:")
st.write("You entered:", P_input)

K_input = st.text_input("Enter K value:")
st.write("You entered:", K_input)

temperature_input = st.text_input("Enter temperature input:")
st.write("You entered:", temperature_input)

humidity_input = st.text_input("Enter humidity input:")
st.write("You entered:", humidity_input)

ph_input = st.text_input("Enter ph value:")
st.write("You entered:", ph_input)

rainfall_input = st.text_input("Enter rainfall value:")
st.write("You entered:", rainfall_input)

scaling_params = joblib.load('scaling_params.pkl')

# Create a new MinMaxScaler and set the parameters
scaler = MinMaxScaler()
scaler.data_min_ = scaling_params["min"]
scaler.data_max_ = scaling_params["max"]
scaler.min_ = scaling_params["min_"]
scaler.scale_ = scaling_params["scaler_"]

predict = st.button("Predict")
if predict:
    value=stacking_classifier.predict(scaler.transform([[N_input,P_input,K_input,temperature_input,humidity_input,ph_input,rainfall_input]]))[0]
    st.write(classes["class"][value:value+1])