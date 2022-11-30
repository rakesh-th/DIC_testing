import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#from imblearn.over_sampling import ADASYN
from sklearn.metrics import *
from xgboost import XGBClassifier

st.header("Credit Card Approval Prediction")
st.text_input("Enter your Name: ", key="name")
my_data = pd.read_csv("studentscores.csv")

# load model
#best_xgboost_model = XGBClassifier()
#best_xgboost_model.load_model("best_model.json")

if st.checkbox('Show Training Dataframe'):
    my_data
 

input_hours = st.slider('Number of Hours:', 0, max(my_data["Hours"]), 2.5)

X = my_data.iloc[ : ,   : 1 ].values
Y = my_data.iloc[ : , 1 ].values

X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 1/4, random_state = 0) 

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor = regressor.fit(X_train, Y_train)


if st.button('Make Prediction'):
    inputs = [input_hours]
    prediction = regressor.predict(X_test)(inputs)
    st.write(f"SCore: {prediction:.2f}")

    st.write(f"Thank you {st.session_state.name}! I hope you liked it.")
