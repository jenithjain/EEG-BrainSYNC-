import streamlit as st
import os
import json
import sys
import torch
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.dirname(script_path))

from app.pdf_parser import parse_pdf_to_json
from predict.predict_cognitive import predict_disorders
from model.eeg_classifier import EEGCognitiveClassifier

def load_model():
    try:
        model_path = "d:/NeuroGPT/models/cognitive_model.pt"
        checkpoint = torch.load(model_path)
        
        model = EEGCognitiveClassifier(
            input_features=1024,
            num_classes=7
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        return model, checkpoint['label_encoder']
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def main():
    st.set_page_config(page_title="EEG Analysis Pipeline", layout="wide")
    
    st.title("🧠 EEG Analysis Pipeline")
    st.write("Upload an EEG PDF report for comprehensive analysis")

    os.makedirs("d:/NeuroGPT/outputs", exist_ok=True)
    os.makedirs("d:/NeuroGPT/data/reports", exist_ok=True)

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        steps = ["Converting PDF to JSON", "Loading Model", "Making Prediction"]
        current_step = 0
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Parse PDF to JSON
            status_text.text(steps[current_step])
            pdf_content = uploaded_file.read()
            json_data, json_path = parse_pdf_to_json(pdf_content)
            current_step += 1
            progress_bar.progress(current_step / len(steps))
            
            if json_data:
                with st.expander("View Parsed JSON Data"):
                    st.json(json_data)
                
                # Step 2: Load Model
                status_text.text(steps[current_step])
                model, label_encoder = load_model()
                current_step += 1
                progress_bar.progress(current_step / len(steps))
                
                if model and label_encoder:
                    # Step 3: Make Prediction
                    status_text.text(steps[current_step])
                    
                    eeg_data = pd.DataFrame([json_data['eeg_data']])
                    eeg_data['age'] = json_data['patient_info']['age']
                    eeg_data['sex'] = 1 if json_data['patient_info']['sex'] == 'M' else 0
                    eeg_data['education'] = 16
                    eeg_data['iq'] = 100
                    
                    model.input_features = len([col for col in eeg_data.columns 
                                              if col.startswith(('AB.', 'COH.'))])
                    
                    labels, probabilities = predict_disorders(model, eeg_data, label_encoder)
                    current_step += 1
                    progress_bar.progress(current_step / len(steps))
                    
                    # Display Results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Patient Information")
                        st.write(f"ID: {json_data['patient_info']['id']}")
                        st.write(f"Age: {json_data['patient_info']['age']}")
                        st.write(f"Sex: {json_data['patient_info']['sex']}")
                        
                        st.subheader("Primary Diagnosis")
                        st.write(f"Condition: {labels[0]}")
                        st.write(f"Confidence: {np.max(probabilities[0])*100:.2f}%")
                    
                    with col2:
                        st.subheader("EEG Band Averages")
                        for band in ['delta', 'theta', 'alpha', 'beta', 'highbeta', 'gamma']:
                            cols = [col for col in eeg_data.columns if band in col.lower()]
                            if cols:
                                avg = eeg_data[cols].mean(axis=1).values[0]
                                st.write(f"{band.capitalize()}: {avg:.2f} μV²")
                    
                    progress_bar.progress(1.0)
                    status_text.text("Analysis Complete! 🎉")
                    
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()