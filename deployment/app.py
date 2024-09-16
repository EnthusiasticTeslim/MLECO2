import os, sys
import streamlit as st
import numpy as np
import pandas as pd
cwd=os.path.dirname(__file__)
main_directory = os.path.abspath(os.path.join(cwd, '..'))
sys.path.append(main_directory)
from src.model import predict
from src.utils import cu_fraction, get_weight, preprocessing

layers = [6, 16, 16, 16, 3]
model_directory = f'{main_directory}/neuralnetwork/results'

style = """
        <style>
        .result-container {
            display: flex;
            flex-direction: column;
            align-items: flex-start;
            margin-top: 20px;
        }
        .result-item {
            background-color: #f0f0f0;
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
            width: 100%;
        }
        </style>
        """

st.set_page_config(page_title='FE calculator', page_icon='🏗️')
st.title('FE Calculator')
st.info('This app calculates Faradaic efficiency of the Sn/Cu CO2RR catalysis')

with st.expander('About this app'):
    st.markdown('**What can this app do?**')
    st.info('This app allow users to predict the faradiac efficiencies based on a neural network based ML model.')

    st.markdown('**How to use the app?**')
    st.warning('To engage with the app, set the input parameters as desired by adjusting the various slider widgets. As a result, this would call the trained model and then, display the model results.')

    st.markdown('**Under the hood**')
    st.markdown('ML model:')
    st.code('- Trained pytorch-based NN model to predict FE of HCOOH, C2H5OH and H2 :', language='markdown')
    
    st.markdown('Libraries used:')
    st.code('- numpy for data wrangling\n- pytorch for building a machine learning model\n- Streamlit for user interface', language='markdown')

    st.markdown('**Input Parameters**')
    st.code(" - Current Density\n- Potential\n- Sn (fraction)\n- pH", language='markdown')

    st.markdown('**Code**')
    url = "https://github.com/EnthusiasticTeslim/mleco2"
    st.code(f"{url}", language='markdown')
    
with st.sidebar:
    option = st.radio('Do you want to upload a file?', 
                ['No', 'Yes'],
                captions = ["file must have header: cDen, Pot, Sn %, pH",])
    
    if option == 'Yes':
        pass
        uploaded_file = st.file_uploader("Choose a file", type=['xlsx', 'csv', 'txt'])
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.txt'):
                data = pd.read_csv(uploaded_file, sep='\t')
            elif uploaded_file.name.endswith('.xls') or uploaded_file.name.endswith('.xlsx'):
                data = pd.read_excel(uploaded_file)
            else:
                st.write('Please upload a valid file')
            st.write('Number of Samples:', data.shape[0])
            st.write('Data:', data)
        else:
            st.write('Please upload a file')
        
    else:
        st.write('### Enter Input Parameters')
        cDen = st.number_input('Current Density', min_value=141.00, max_value=450.00, value=200.00)
        Pot = st.number_input('Potential', min_value=2.80, max_value=5.10, value=3.50)
        Sn = st.number_input('Sn (%)', min_value=0.0, max_value=100.0, value=50.0)/100
        pH = st.number_input('pH', min_value=8.02, max_value=14.05, value=9.0)

if st.button('Calculate'):
    max_cDen = 450.00
    max_Pot = 5.10
    max_Sn = 100
    max_pH = 14.05
    max_weight = 118.71
    
    if option == 'No':
        df = np.array([cDen/max_cDen, Pot/max_Pot, Sn/max_Sn, pH/max_pH, get_weight(Sn)/max_weight, cu_fraction(Sn)]).reshape(1, -1)
        prediction = predict(data=preprocessing(df), layer_model=layers, dir=model_directory)

        results = {
            'HCOOH': f"{100*np.mean([prediction[i][:, 0] for i in range(len(prediction))]):.2f} +/ {100*np.std([prediction[i][:, 0] for i in range(len(prediction))]):.2f}",
            'Ethanol': f"{100*np.mean([prediction[i][:, 1] for i in range(len(prediction))]):.2f} +/ {100*np.std([prediction[i][:, 1] for i in range(len(prediction))]):.2f}",
            'H2': f"{100*np.mean([prediction[i][:, 2] for i in range(len(prediction))]):.2f} +/ {100*np.std([prediction[i][:, 2] for i in range(len(prediction))]):.2f}"
        }

        st.write('## Faradaic Efficiency')
        st.markdown(style, unsafe_allow_html=True)
                    
        st.markdown('<div class="result-container">', unsafe_allow_html=True)
        for key, value in results.items():
            st.markdown(f'<div class="result-item"><strong>{key}:</strong> {value}%</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        if 'cDen' in data.columns and 'Pot' in data.columns and 'Sn %' in data.columns and 'pH' in data.columns:
            df = data[['cDen', 'Pot', 'Sn %', 'pH']].copy()
            df['cDen'] /= max_cDen
            df['Pot'] /= max_Pot
            df['Sn %'] /= max_Sn
            df['pH'] /= max_pH
            df['Cu %'] = df['Sn %'].apply(cu_fraction)
            df['weight'] = df['Sn %'].apply(get_weight) / max_weight

            df_ = np.array(df[['cDen', 'Pot', 'Sn %', 'pH', 'weight', 'Cu %']])
            prediction = predict(data=preprocessing(df_), layer_model=layers, dir=model_directory)

            data['HCOO_mean'] = 100*np.mean([prediction[i][:, 0] for i in range(len(prediction))])
            data['HCOO_std'] = 100*np.std([prediction[i][:, 0] for i in range(len(prediction))])
            data['Ethanol_mean'] = 100*np.mean([prediction[i][:, 1] for i in range(len(prediction))])
            data['Ethanol_std'] = 100*np.std([prediction[i][:, 1] for i in range(len(prediction))])
            data['H2_mean'] = 100*np.mean([prediction[i][:, 2] for i in range(len(prediction))])
            data['H2_std'] = 100*np.std([prediction[i][:, 2] for i in range(len(prediction))])

            st.download_button(
                label="Download data as CSV",
                data=data.to_csv().encode("utf-8"),
                file_name="result.csv",
                mime="text/csv",
            )
        else:
            st.warning('Please upload a file with the required columns: cDen, Pot, Sn %, pH', icon="⚠️")