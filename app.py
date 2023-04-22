import streamlit as st
import numpy as np, pandas as pd
import pickle, base64

# load the trained model
model = pickle.load(open('trained_model.pkl', 'rb'))

# define a function to make prediction
def predict_target(gravity, ph, osmo, cond, urea, calc, osmo_cond_ratio, urea_calc_diff, osmo_urea_interaction, gravity_bin, ph_bin, osmo_bin, cond_bin, urea_bin, calc_bin):
    
    # create a numpy array with the input values and reshape it for the model
    input_data = np.array([[gravity, ph, osmo, cond, urea, calc, osmo_cond_ratio, urea_calc_diff, osmo_urea_interaction, gravity_bin, ph_bin, osmo_bin, cond_bin, urea_bin, calc_bin]])
    input_data = np.reshape(input_data, (input_data.shape[0], input_data.shape[1], 1))
    
    # make prediction using the model
    prediction = model.predict(input_data)
    
    # return the predicted target value
    return prediction[0][0]
    # 'prediction' gets returned as 1*1 matrix
    # since we need only the value we are using 'prediction[0][0]'

# define a function to read and derive minimum/maximum values for features from the dataset
data = pd.read_csv('kidney_stone_data.csv')
data['osmo_cond_ratio'] = data['osmo'] / data['cond']
data['urea_calc_diff'] = data['urea'] - data['calc']
data['osmo_urea_interaction'] = data['osmo'] * data['urea']
data['gravity_bin'] = pd.qcut(data['gravity'], 5, labels=False)
data['ph_bin'] = pd.qcut(data['ph'], 5, labels=False)
data['osmo_bin'] = pd.qcut(data['osmo'], 5, labels=False)
data['cond_bin'] = pd.qcut(data['cond'], 5, labels=False)
data['urea_bin'] = pd.qcut(data['urea'], 5, labels=False)
data['calc_bin'] = pd.qcut(data['calc'], 5, labels=False)

# adding a background cover
def add_bg_img(image_file):
    with open(image_file, 'rb') as image_file:
        encoded_string = base64.b64encode(image_file.read())
        st.markdown(
                f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
        )
add_bg_img('./Images/pexels-8325982.jpg')

# add a title and a header
st.markdown("<h1 style='text-align: center; color: black;'>Kidney Stone Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: justify; color: black;'>The <i>six</i> physical characteristics of urine, <i>specific gravity</i>, <i>pH value</i>, <i>osmolarity</i>, <i>conductivity</i>, <i>urea</i> and <i>calcium  concentration</i> alongside few secondary characteristics have been provided below.</p>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: black;'>Kindly drag the slider to enter the values and click on 'Predict' button</h5>", unsafe_allow_html=True)

# add input fields for the features
gravity = st.slider('specific gravity', min_value=float(data.gravity.min()), max_value=float(data.gravity.max()), value=float(data.gravity.mean()))
ph = st.slider("pH value", min_value=float(data.ph.min()), max_value=float(data.ph.max()), value=float(data.ph.mean()))
osmo = st.slider("osmolarity value", min_value=int(data.osmo.min()), max_value=int(data.osmo.max()), value=int(data.osmo.mean()))
cond = st.slider("conductivity value", min_value=float(data.cond.min()), max_value=float(data.cond.max()), value=float(data.cond.mean()))
urea = st.slider("urea concentration", min_value=int(data.urea.min()), max_value=int(data.urea.max()), value=int(data.urea.mean()))
calc = st.slider("calcium concentration", min_value=float(data.calc.min()), max_value=float(data.calc.max()), value=float(data.calc.mean()))
osmo_cond_ratio = st.slider("ratio of osmolarity versus conductivity", min_value=float(data.osmo_cond_ratio.min()), max_value=float(data.osmo_cond_ratio.max()), value=float(data.osmo_cond_ratio.mean()))
urea_calc_diff = st.slider("difference between urea and calcium concentration", min_value=float(data.urea_calc_diff.min()), max_value=float(data.urea_calc_diff.max()), value=float(data.urea_calc_diff.mean()))
osmo_urea_interaction = st.slider("osmolarity urea interaction value", min_value=float(data.osmo_urea_interaction.min()), max_value=float(data.osmo_urea_interaction.max()), value=float(data.osmo_urea_interaction.mean()))
gravity_bin = st.slider("gravity_bin value", min_value=float(data.gravity_bin.min()), max_value=float(data.gravity_bin.max()), value=float(data.gravity_bin.mean()))
ph_bin = st.slider("ph_bin value", min_value=float(data.ph_bin.min()), max_value=float(data.ph_bin.max()), value=float(data.ph_bin.mean()))
osmo_bin = st.slider("osmolarity_bin value", min_value=float(data.osmo_bin.min()), max_value=float(data.osmo_bin.max()), value=float(data.osmo_bin.mean()))
cond_bin = st.slider("conductivity_bin value", min_value=float(data.cond_bin.min()), max_value=float(data.cond_bin.max()), value=float(data.cond_bin.mean()))
urea_bin = st.slider("urea_bin value", min_value=float(data.urea_bin.min()), max_value=float(data.urea_bin.max()),value=float(data.urea_bin.mean()))
calc_bin = st.slider("calc_bin value", min_value=float(data.calc_bin.min()), max_value=float(data.calc_bin.max()), value=float(data.calc_bin.mean()))

# add a button to make the prediction
if st.button("Predict"):
    prediction = predict_target(gravity, ph, osmo, cond, urea, calc, osmo_cond_ratio, urea_calc_diff, osmo_urea_interaction, gravity_bin, ph_bin, osmo_bin, cond_bin, urea_bin, calc_bin)
    st.write("Predicted Value: ", prediction)
    st.divider()
    if prediction > 0.3:
        st.markdown("<h5 style='text-align: center; color: black;'><i>Report suggests presence of stones in your kidney</i></h5>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: black;'><i>Please do not panic. Consider visiting a urologist/nephrologist for further consultation.</i></p>", unsafe_allow_html=True)
        st.divider()
    else:
        st.markdown("<h5 style='text-align: center; color: black;'><i>Report doesn't account for presence of stones in your kidney</i></h5>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: black;'><i>Personal advise would be to visit a urologist/nephrologist for final confirmation.</i></p>", unsafe_allow_html=True)
        st.divider()