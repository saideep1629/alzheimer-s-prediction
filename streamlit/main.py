import streamlit as st
import tensorflow as tf
import numpy as np


#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("streamlit/alzheimer.h5")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(256,256))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    confidence = round(100 * (np.max(predictions)), 2)
    return np.argmax(predictions), confidence #return index of max element


#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Alzheimer Prediction"])

#Main Page
if(app_mode=="Home"):
    st.header("ALZHEIMER DISEASE PREDICTION")
    image_path = "streamlit/home.jpg"
    st.image(image_path,use_column_width=True)
    st.markdown("""
    "Welcome to the Alzheimer's Disease Prediction System! 🧠🔍

Our mission is to assist in the early prediction of Alzheimer's disease. Upload brain imaging data, and our system will analyze it to detect any potential signs of Alzheimer's. Together, let's strive for early intervention and better management of this condition!

### How It Works
1. **Upload Data:** Navigate to the **Prediction** page and upload brain imaging data for analysis.
2. **Analysis:** Our system will employ advanced algorithms to scrutinize the data, identifying potential indicators of Alzheimer's disease.
3. **Results:** Review the results and recommendations for further evaluation or action.

### Why Choose Us?
- **Accuracy:** Our system leverages cutting-edge machine learning techniques to provide accurate predictions.
- **User-Friendly:** Enjoy a simple and intuitive interface designed for ease of use.
- **Fast and Efficient:** Receive predictions swiftly, facilitating prompt decision-making and intervention.

### Get Started
Click on the **Prediction** page in the menu to upload your data and harness the capabilities of our Alzheimer's Disease Prediction System!

### About Us
Explore more about our project, team, and objectives on the **About** page."
    """)

#About Project
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
                #### Content
                1. Train (4096 images)
                2. Test (544 images)
                3. Validation (512 images)
                4. Model accuracy : 99.10%              

                """)
    col1, col2 = st.columns(2)  # Create two columns for side-by-side display
    with col1:
        st.image("streamlit/training.png", caption="Training and Validation Loss", use_column_width=False)
    with col2:
        st.image("streamlit/test.png", caption="Training and Validation Accuracy", use_column_width=False)

#Prediction Page
elif(app_mode=="Alzheimer Prediction"):
    st.header("Alzheimer Disease Prediction")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
    #Predict button
    if(st.button("Predict")):
        st.write("Our Prediction")
        result_index,confidence = model_prediction(test_image)
        #Reading Labels
        class_name = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
        #st.success("The result is  {}".format(class_name[result_index],confidence))
        st.success(f"The result is {class_name[result_index]} with confidence {confidence}.")
