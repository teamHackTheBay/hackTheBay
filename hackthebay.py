import streamlit as st
import numpy as np
import pandas as pd
import datetime
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from PIL import Image

#set title

image = Image.open('images/oyster.png')
st.image(image, width = 800)

def main():
    activities = ['Intro to the Chesapeake Bay Challenge', 'Data Preparation',
    'Data Visualization', 'Total Nitrogen Model']
    option = st.sidebar.selectbox('Selection Option:', activities)

#Intro
    if option == 'Intro to the Chesapeake Bay Challenge':
        st.title('Intro to the Chesapeake Bay Challenge')
        crossref_temp = """
        <div style="background-color:#E31C3C;padding:1px">
        <h3 style="color:#313F3D;text-align:center;">Intro to the Chesapeake Bay Challenge</h3>
        </div>
        """
        st.markdown(title_page,unsafe_allow_html=True)

        crossref_write = """
        put writing here Jen
        """

        st.markdown(title_page,unsafe_allow_html=True)


#Data Preparation
    elif option == 'Data Preparation':
        st.title('Data Preparation')
        html_temp = """
        <div style="background-color:#E31C3C;padding:1px">
        <h3 style="color:#212F3D;text-align:center;">Data Preparation</h3>
        </div>
        """
        st.markdown(html_temp,unsafe_allow_html=True)


        explorationwrite_up = """
        Jen write here
        """
        st.markdown(explorationwrite_up, unsafe_allow_html=True)

        image = Image.open('images/oyster2.png')
        st.image(image, width = 800)


#Data Visualization
    elif option == 'Data Visualization':
        st.title('Data Visualization')
        html_temp = """
        <div style="background-color:#E31C3C;padding:1px">
        <h3 style="color:#212F3D;text-align:center;">Data Visualization</h3>
        </div>
        """
        st.markdown(html_temp,unsafe_allow_html=True)


        vizwrite_up = """
        Jen write here
        ```python
        This is how I write code here.
        ```
        """
        st.markdown(vizwrite_up, unsafe_allow_html=True)

        image = Image.open('images/oyster2.png')
        st.image(image, width = 800)

#Nitrogen Modeling
    elif option == 'Total Nitrogen Model':
        st.title('Total Nitrogen Model')
        html_temp = """
        <div style="background-color:#E31C3C;padding:1px">
        <h3 style="color:#212F3D;text-align:center;">Total Nitrogen Model</h3>
        </div>
        """
        st.markdown(html_temp,unsafe_allow_html=True)


        modelwrite_up = """
        Jen write here
        """
        st.markdown(modelwrite_up, unsafe_allow_html=True)

        image = Image.open('images/oyster2.png')
        st.image(image, width = 800)


if __name__ == '__main__':
    main()
