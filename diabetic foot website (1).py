#!/usr/bin/env python
# coding: utf-8

# In[31]:


#%%writefile thermal.py
import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageOps
import base64
def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://cdn.pixabay.com/photo/2017/03/09/11/47/insulin-syringe-2129490__340.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
add_bg_from_url()
html_temp = """ 
  <div style="background-color:pink ;padding:10px">
  <h2 style="color:white;text-align:center;">Integrated system for detection of diabetes using deep learning </h2>
  </div>
  """ 
st.markdown(html_temp,unsafe_allow_html=True)
activities=['Introduction','Diagnostic toolkit','About the team']
option=st.sidebar.selectbox('Select the section you want to go through',activities)
if option=='Introduction':
    st.write('This website is an integrated diagnostic application for diabetes. This website comprises of medical imaging based diagnosis of diabetes.')
    st.write('Diabetes is a chronic (long-lasting) health condition that affects how your body turns food into energy.Your body breaks down most of the food you eat into sugar (glucose) and releases it into your bloodstream. When your blood sugar goes up, it signals your pancreas to release insulin. Insulin acts like a key to let the blood sugar into your body’s cells for use as energy.With diabetes, your body doesn’t make enough insulin or can’t use it as well as it should. When there isn’t enough insulin or cells stop responding to insulin, too much blood sugar stays in your bloodstream. Over time, that can cause serious health problems, such as heart disease, vision loss, and kidney disease.')
elif option=='Diagnostic toolkit':
    st.write('This section of the website comprises of tests related to the diagnosis of diabetes')
    dia_act=['Thermal foot image','OCT','Fundus','Skin']
    dia_opt=st.selectbox('Select any one of the below tests',dia_act)
    if dia_opt=='Thermal foot image':
        st.subheader('The user is requested to upload the thermal image of their foot.')
        @st.cache(allow_output_mutation=True)
        def load_model():
            model=tf.keras.models.load_model(r"C:\Users\sairam\Desktop\research works\diabetes project\diabetic foot thermal images.h5")
            return model
        with st.spinner('Model is being loaded..'):
            model=load_model()
        file = st.file_uploader("Please upload the image of suspicion in the allocated dropdown box", type=["jpg", "png","jpeg"])
        st.set_option('deprecation.showfileUploaderEncoding', False)
        if file is None:
             st.text("Please upload an image file within the allotted file size")
        else:
            img = Image.open(file)
            st.image(img, use_column_width=False)
            size = (224,224)    
            image = ImageOps.fit(img, size, Image.ANTIALIAS)
            imag = np.asarray(image)
            imaga = np.expand_dims(imag,axis=0) 
            predictions = model.predict(imaga)
            a=predictions[0]
            if st.button('Click to get the results:'):
                if a<0.50:
                    st.success('The subject under observation appears to be normal.')
                
                else:
                    st.error('The subject under consideration is suspected to be Diabetic Foot Ulcer.')
    elif dia_opt=='OCT':
        st.subheader('The user is requested to upload the OCT image of their eye.')
        @st.cache(allow_output_mutation=True)
        def load_model():
            model=tf.keras.models.load_model(r"C:\Users\sairam\Desktop\research works\diabetes project\diabetic OCT images(2.9).h5")
            return model
        with st.spinner('Model is being loaded..'):
            model=load_model()
        file = st.file_uploader("Please upload the image of suspicion in the allocated dropdown box", type=["jpg", "png","jpeg"])
        st.set_option('deprecation.showfileUploaderEncoding', False)
        if file is None:
             st.text("Please upload an image file within the allotted file size")
        else:
            img = Image.open(file)
            st.image(img, use_column_width=False)
            size = (224,224)    
            image = ImageOps.fit(img, size, Image.ANTIALIAS)
            imag = np.asarray(image)
            imaga = np.expand_dims(imag,axis=0) 
            predictions = model.predict(imaga)
            a=predictions[0]
            if st.button('Click to get the results:'):
                if a<0.50:
                    st.error('The subject under observation is suspected to be Diabetic Macular Oedema.')
                
                else:
                    st.success('The subject under consideration is observed to be normal.')
    elif dia_opt=='Fundus':
        st.subheader('The user is requested to upload the fundus image of their eye.')
        @st.cache(allow_output_mutation=True)
        def load_model():
            model=tf.keras.models.load_model(r"C:\Users\sairam\Desktop\research works\diabetes project\diabetic retinopathy classification gaussian filtered.h5")
            return model
        with st.spinner('Model is being loaded..'):
            model=load_model()
        file = st.file_uploader("Please upload the image of suspicion in the allocated dropdown box", type=["jpg", "png","jpeg"])
        st.set_option('deprecation.showfileUploaderEncoding', False)
        if file is None:
             st.text("Please upload an image file within the allotted file size")
        else:
            img = Image.open(file)
            st.image(img, use_column_width=False)
            size = (224,224)    
            image = ImageOps.fit(img, size, Image.ANTIALIAS)
            imag = np.asarray(image)
            imag=imag/255
            imaga = np.expand_dims(imag,axis=0) 
            predictions = model.predict(imaga)
            a=np.argmax(predictions,axis=1)
            st.write(a)
            if st.button('Click to get the results:'):
                if a==0:
                    st.warning('The subject under observation appears to be mildly diabetic retinopathy.')
                elif a==1:
                    st.warning('The subject under observation appears to be moderately diabetic retinopathy.')
                elif a==2:
                    st.success('The subject under consideration is suspected to be normal.')
                elif a==3:
                    st.error('The subject under observation appears to be proliferately diabetic retinopathy.')
                else:
                    st.error('The subject under observation appears to be severely diabetic retinopathy.')
    elif dia_opt=='Skin':
        st.subheader('The user is requested to upload the camera image of the skin.')
        @st.cache(allow_output_mutation=True)
        def load_model():
            model=tf.keras.models.load_model(r"C:\Users\sairam\Desktop\research works\diabetes project\skin image diabetes prediction.h5")
            return model
        with st.spinner('Model is being loaded..'):
            model=load_model()
        file = st.file_uploader("Please upload the image of suspicion in the allocated dropdown box", type=["jpg", "png","jpeg"])
        st.set_option('deprecation.showfileUploaderEncoding', False)
        if file is None:
             st.text("Please upload an image file within the allotted file size")
        else:
            img = Image.open(file)
            st.image(img, use_column_width=False)
            size = (224,224)    
            image = ImageOps.fit(img, size, Image.ANTIALIAS)
            imag = np.asarray(image)
            imaga = np.expand_dims(imag,axis=0) 
            predictions = model.predict(imaga)
            a=np.argmax(predictions,axis=1)
            st.write(a)
            if st.button('Click to get the results:'):
                if a==0:
                    st.error('The subject under observation is suspected to have diabetes.')
                elif a==1:
                    st.success('The subject under observation appears to be normal.')
                else:
                    st.warning('The subject under observation appears to be have other skin disease and is not diabetic.')
elif option=='About the team':
    st.success('1. V.A.Sairam, Department of Biomedical Engineering, Rajalakshmi Engineering College, Chennai, India')
    st.success('2. Sameena Alam, Department of Biomedical Engineering, Rajalakshmi Engineering College, Chennai, India')
    st.success('3. H.Sruthi, Department of Biomedical Engineering, Rajalakshmi Engineering College, Chennai, India')
    st.success('1. R.Saranya, Department of Biomedical Engineering, Rajalakshmi Engineering College, Chennai, India')

