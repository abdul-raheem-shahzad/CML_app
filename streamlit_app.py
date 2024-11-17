#creating a stremlit app from trained model
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dt pickled model
# dt_model = pickle.load(open('dt_model.pkl', 'rb'))
# # Load the rf pickled model
# rf_model = pickle.load(open('rf_model.pkl', 'rb'))
# Load the xgb pickled model
xgb_model = pickle.load(open('xtreme_model.pkl','rb'))
# Load the svc pickled model
svc_model = pickle.load(open('svc_model.pkl', 'rb'))
# Load the knn pickled model
knn_model = pickle.load(open('knn_model.pkl', 'rb'))
# Load the lr pickled model
lr_model = pickle.load(open('lr_model.pkl', 'rb'))

# heading of the app
st.header('Machine Learning Based Leukemia Cancer Prediction Using Protein Sequential Data')
st.write('This app predicts weather a patient has Leukemia or not using the protein sequential data')

# button to upload patient data
uploaded_file = st.file_uploader("Upload Paitent Data", type="csv")
#slider for the user to select the model
model = st.sidebar.selectbox('Select the model', ('Decision Tree', 'Random Forest', 'XG Boost', 'Support Vector Machine', 'K-Nearest Neighbors', 'Logistic Regression'))
# if the user selects the decision tree model
if model == 'Decision Tree':
    st.write('You have selected Decision Tree model')
    # if the file is not uploaded
    if uploaded_file is None:
        st.write('Please upload the patient data')
    elif uploaded_file is not None:
        # Read the uploaded file
        df = pd.read_csv(uploaded_file)
        # Display the uploaded file
        st.write(df)
        # Predict the output
        prediction = dt_model.predict(df)
        # Display the prediction
        # Display the prediction by pressing button
        if st.button('Predict'):
            st.write('The predicted output is: ', prediction)
            if prediction == 1:
                st.write('The patient has Leukemia')
            else:
                st.write('The patient does not have Leukemia')
        st.image(["dt_roc.png"])
# if the user selects the random forest model
elif model == 'Random Forest':
    st.write('You have selected Random Forest model')
    # if the file is not uploaded
    if uploaded_file is None:
        st.write('Please upload the patient data')
    elif uploaded_file is not None:
        # Read the uploaded file
        df = pd.read_csv(uploaded_file)
        # Display the uploaded file
        st.write(df)
        # Predict the output
        prediction = rf_model.predict(df)
        # Display the prediction by pressing button
        if st.button('Predict'):
            st.write('The predicted output is: ', prediction)
            if prediction == 1:
                st.write('The patient has Leukemia')
            else:
                st.write('The patient does not have Leukemia')
        st.image(["rf_roc.png"])
        
# if the user selects the xgboost model
elif model == 'XG Boost':
    st.write('You have selected XGBoost model')
    # if the file is not uploaded
    if uploaded_file is None:
        st.write('Please upload the patient data')
    elif uploaded_file is not None:
        # Read the uploaded file
        df = pd.read_csv(uploaded_file)
        # Display the uploaded file
        st.write(df)
        # Predict the output
        # if selected model is XGBoost
        
        prediction = xgb_model.predict(df)
        # Display the prediction by pressing button
        if st.button('Predict'):
            st.write('The predicted output is: ', prediction)
            if prediction == 1:
                st.write('The patient has Leukemia')
            else:
                st.write('The patient does not have Leukemia')
        st.image(["xgb_roc.png"])
        
# if the user selects the support vector machine model
elif model == 'Support Vector Machine':
    st.write('You have selected Support Vector Machine model')
    # if the file is not uploaded
    if uploaded_file is None:
        st.write('Please upload the patient data')
    elif uploaded_file is not None:
        # Read the uploaded file
        df = pd.read_csv(uploaded_file)
        # Display the uploaded file
        st.write(df)
        # Predict the output
        prediction = svc_model.predict(df)
        # Display the prediction by pressing button
        if st.button('Predict'):
            st.write('The predicted output is: ', prediction)
            if prediction == 1:
                st.write('The patient has Leukemia')
            else:
                st.write('The patient does not have Leukemia')
                #printing the graph in image format
    
                
        
# if the user selects the k-nearest neighbors model
elif model == 'K-Nearest Neighbors':
    st.write('You have selected K-Nearest Neighbors model')
    # if the file is not uploaded
    if uploaded_file is None:
        st.write('Please upload the patient data')
    
    elif uploaded_file is not None:
        # Read the uploaded file
        df = pd.read_csv(uploaded_file)
        # Display the uploaded file
        st.write(df)
        # Predict the output
        prediction = knn_model.predict(df)
        # Display the prediction by pressing button
        if st.button('Predict'):
            st.write('The predicted output is: ', prediction)
            if prediction == 1:
                st.write('The patient has Leukemia')
            else:
                st.write('The patient does not have Leukemia')
    
        
# if the user selects the logistic regression model
elif model == 'Logistic Regression':
    st.write('You have selected Logistic Regression model')
    # if the file is not uploaded
    if uploaded_file is None:
        st.write('Please upload the patient data')
    elif uploaded_file is not None:
        # Read the uploaded file
        df = pd.read_csv(uploaded_file)
        # Display the uploaded file
        st.write(df)
        # Predict the output
        prediction = lr_model.predict(df)
        # Display the prediction by pressing button
        if st.button('Predict'):
            st.write('The predicted output is: ', prediction)
            if prediction == 1:
                st.write('The patient has Leukemia')
            else:
                st.write('The patient does not have Leukemia')
        


        


