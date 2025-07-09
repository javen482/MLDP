import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the model
model = joblib.load('model_hdb.pkl')

#streamlit app
st.title("HDB Resale Price Prediction")

#define the input options
towns = ['Tampines', 'Bedok', 'Punggol']
flat_types = ['2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM']
storey_ranges = ['01 to 03','04 TO 06', '07 TO 09']

# user inputs
town_selected = st.selectbox("Select Town", towns)
flat_type_selected = st.selectbox("Select Flat Type", flat_types)
storey_range_selected = st.selectbox("Select Storey", storey_ranges)
floor_area_selected = st.slider("Select Floor Area (sqm)", min_value=30,
                                max_value=200, value=70)

#Predict button
if st.button("Predict HDB price"):

    # Create dict for input features
    input_data = {
        'town': town_selected,
        'flat_type': flat_type_selected,
        'storey_range': storey_range_selected,
        'floor_area_sqm': floor_area_selected
    }

    # Convert dict to DataFrame
    df_input = pd.DataFrame({
        'town': [town_selected],
        'flat_type': [flat_type_selected],
        'storey_range': [storey_range_selected],
        'floor_area_sqm': [floor_area_selected]
    })

    # One hot encode
    df_input = pd.get_dummies(df_input,
                              columns = ['town','flat_type','storey_range']
                              )
    
    # df_input = df_input.to_numpy()

    df_input = df_input.reindex(columns=model.feature_names_in_, 
                                fill_value=0)
    
    # Predict 
    y_unseen_pred = model.predict(df_input)[0]
    st.success(f"Predicted Resale Price: ${y_unseen_pred:,.2f}")

st.markdown(
    f"""
    <style>
    .stApp {{
        background: url("https://www.google.com/imgres?q=image&imgurl=https%3A%2F%2Fimages.ctfassets.net%2Fhrltx12pl8hq%2F28ECAQiPJZ78hxatLTa7Ts%2F2f695d869736ae3b0de3e56ceaca3958%2Ffree-nature-images.jpg%3Ffit%3Dfill%26w%3D1200%26h%3D630&imgrefurl=https%3A%2F%2Fwww.shutterstock.com%2Fdiscover%2Ffree-nature-images&docid=uEeA4F2Pf5UbvM&tbnid=cVgA8oYynNpqQM&vet=12ahUKEwikz_Dr6q6OAxUvUGwGHdUqH2kQM3oECBkQAA..i&w=1200&h=630&hcb=2&ved=2ahUKEwikz_Dr6q6OAxUvUGwGHdUqH2kQM3oECBkQAA")
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
)