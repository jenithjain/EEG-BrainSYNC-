import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json
import glob
import warnings

# Suppress PDF-related warnings
warnings.filterwarnings("ignore", category=UserWarning, module="PyPDF2")

script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.dirname(script_path))

from model.eeg_classifier import EEGCognitiveClassifier
import streamlit as st
import google.generativeai as genai

# Import the PDF parser functionality
sys.path.append("d:/NeuroGPT/app")
from pdf_parser import parse_pdf_to_json

# Configure Google API key
genai.configure(api_key="AIzaSyCl-bptq__Xvdl-hE2pIkpD7WLkLkXETjw")

# Add these imports for EEG visualization
from mne.viz import plot_topomap
from mne import create_info
from mne.channels import make_standard_montage

def prepare_eeg_data(data):
    """Extract EEG features from data"""
    if isinstance(data, pd.DataFrame):
        eeg_columns = [col for col in data.columns if any(band in col for band in 
                      ['delta', 'theta', 'alpha', 'beta', 'gamma'])]
        
        if not eeg_columns:
            return torch.FloatTensor(data[['time_len']].values)
        
        return torch.FloatTensor(data[eeg_columns].values)
    
    return torch.FloatTensor(data)

def extract_band_values(eeg_data):
    """Extract band values from EEG data"""
    band_values = {}
    for band in ['delta', 'theta', 'alpha', 'beta', 'highbeta', 'gamma']:
        band_cols = [col for col in eeg_data.columns if band.lower() in col.lower()]
        if band_cols:
            values = eeg_data[band_cols].values[0]
            band_values[band] = np.array(values, dtype=float)
        else:
            band_values[band] = np.zeros(19)
    return band_values


def create_brain_visualization(data, prediction, band_values):
    """Create EEG brain visualization with improved heatmaps"""
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(15, 10))
    
    # Create channel positions
    montage = make_standard_montage('standard_1020')
    ch_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 
                'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
    
    info = create_info(ch_names=ch_names, sfreq=256., ch_types='eeg')
    info.set_montage(montage)
    pos = np.array([montage.get_positions()['ch_pos'][ch] for ch in ch_names])[:, :2]
    
    # Define bands with updated settings
    bands = {
        'delta': {'max': 25, 'title': 'Delta Band (uV^2)', 'cmap': 'RdBu_r'},
        'theta': {'max': 14, 'title': 'Theta Band (uV^2)', 'cmap': 'RdBu_r'},
        'alpha': {'max': 10, 'title': 'Alpha Band (uV^2)', 'cmap': 'RdBu_r'},
        'beta': {'max': 8, 'title': 'Beta Band (uV^2)', 'cmap': 'RdBu_r'},
        'highbeta': {'max': 6, 'title': 'Highbeta Band (uV^2)', 'cmap': 'RdBu_r'},
        'gamma': {'max': 4, 'title': 'Gamma Band (uV^2)', 'cmap': 'RdBu_r'}
    }
    
    # Plot each frequency band
    for idx, (band, settings) in enumerate(bands.items(), 1):
        ax = fig.add_subplot(2, 3, idx)
        
        # Get band values and add random variation for visualization
        data_array = np.zeros(19)
        for i, ch in enumerate(ch_names):
            key = f"AB.{band}.{ch}"
            if key in data.columns:
                base_value = data[key].values[0]
                data_array[i] = base_value + np.random.normal(0, base_value * 0.1)
        
        # Ensure non-negative values
        data_array = np.maximum(data_array, 0)
        
        # Plot topomap with correct interpolation settings (removed duplicate plot_topomap call)
        im = plot_topomap(
            data_array, 
            pos, 
            axes=ax, 
            show=False,
            cmap=settings['cmap'],
            vlim=(0, settings['max']),
            outlines='head',
            sensors=True,
            contours=10,
            image_interp='cubic',  # Using only cubic interpolation
            extrapolate='local'
        )
        
        # Set matplotlib interpolation for smoother display
        im[0].set_interpolation('bicubic')
        
        # Add colorbar with custom styling
        cbar = plt.colorbar(im[0], ax=ax)
        cbar.set_label('μV²', rotation=270, labelpad=15, color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        
        # Enhanced title styling
        ax.set_title(settings['title'], color='white', pad=10, fontsize=10)
    
    plt.tight_layout()
    plt.suptitle(f'EEG Analysis: {prediction}', fontsize=16, color='white', y=1.02)
    
    return fig

def display_eeg_visualization(fig):
    """Display EEG visualization in Streamlit"""
    st.subheader("🧠 EEG Brain Activity Mapping")
    
    # Create columns for the visualization
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.pyplot(fig)
    
    with col2:
        st.markdown("""
        ### Legend
        - **Red**: High activity
        - **White**: Medium activity
        - **Blue**: Low activity
        
        ### Bands Information
        - **Delta**: 0.5-4 Hz (Sleep)
        - **Theta**: 4-8 Hz (Drowsy)
        - **Alpha**: 8-13 Hz (Relaxed)
        - **Beta**: 13-30 Hz (Active)
        - **Gamma**: >30 Hz (Focus)
        """)

# Remove this line as it's causing the error
# display_eeg_visualization(fig)  # <- Remove this line

def get_eeg_expert_analysis(eeg_data, prediction_results):
    """Get AI analysis of EEG data"""
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Calculate average band powers
    band_averages = {}
    for band in ['delta', 'theta', 'alpha', 'beta', 'highbeta', 'gamma']:
        band_values = [v for k, v in eeg_data.items() if f'AB.{band}.' in k]
        band_averages[band] = np.mean(band_values) if band_values else 0.0

    expert_prompt = f"""
    Analyze this EEG data as an expert:

    Band Averages:
    Delta: {band_averages['delta']:.2f} uV^2
    Theta: {band_averages['theta']:.2f} uV^2
    Alpha: {band_averages['alpha']:.2f} uV^2
    Beta: {band_averages['beta']:.2f} uV^2
    High Beta: {band_averages['highbeta']:.2f} uV^2
    Gamma: {band_averages['gamma']:.2f} uV^2

    Diagnosis: {prediction_results['primary_diagnosis']}
    Confidence: {prediction_results['confidence']:.2f}%

    Provide a brief clinical analysis focusing on:
    1. Key findings in each frequency band
    2. Clinical significance
    3. Brief recommendations
    """

    response = model.generate_content(expert_prompt)
    return response.text, band_averages

def predict_disorders(model, data, label_encoder):
    """Predict disorders from EEG data"""
    model.eval()
    with torch.no_grad():
        tensor_data = prepare_eeg_data(data)
        outputs = model(tensor_data)
        probabilities = F.softmax(outputs, dim=1)
        predictions = torch.argmax(outputs, dim=1)
        labels = label_encoder.inverse_transform(predictions.cpu().numpy())
    
    # Extract band values
    band_values = extract_band_values(data)
    
    # Create and save visualization
    os.makedirs('d:/NeuroGPT/outputs', exist_ok=True)
    fig = create_brain_visualization(data, labels[0], band_values)
    plt.savefig('d:/NeuroGPT/outputs/eeg_analysis.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return labels, probabilities.cpu().numpy(), band_values

def simulate_treatment_effect(model, data, label_encoder, treatment_type='medication'):
    """Simulate effect of treatment on EEG patterns"""
    # Get baseline prediction
    original_labels, original_probs, original_band_values = predict_disorders(model, data, label_encoder)
    
    # Simulate treatment effects on EEG
    treated_data = data.copy()
    
    # Treatment effects dictionary to track changes
    effect_description = {
        'increased': [],
        'decreased': [],
        'normalized': []
    }
    
    if treatment_type == 'medication':
        # Simulate medication effect (increase alpha, decrease theta)
        alpha_cols = [col for col in treated_data.columns if 'alpha' in col]
        theta_cols = [col for col in treated_data.columns if 'theta' in col]
        beta_cols = [col for col in treated_data.columns if 'beta' in col and 'high' not in col.lower()]
        
        if alpha_cols:
            for col in alpha_cols:
                treated_data[col] = treated_data[col] * 1.2  # Increase alpha by 20%
            effect_description['increased'].append('Alpha waves (relaxation)')
            
        if theta_cols:
            for col in theta_cols:
                treated_data[col] = treated_data[col] * 0.8  # Decrease theta by 20%
            effect_description['decreased'].append('Theta waves (drowsiness)')
            
        if beta_cols:
            for col in beta_cols:
                treated_data[col] = treated_data[col] * 0.9  # Slight decrease in beta
            effect_description['decreased'].append('Beta waves (stress/anxiety)')
    
    elif treatment_type == 'therapy':
        # Simulate therapy effect (normalize beta, increase alpha)
        beta_cols = [col for col in treated_data.columns if 'beta' in col]
        alpha_cols = [col for col in treated_data.columns if 'alpha' in col]
        
        if beta_cols:
            for col in beta_cols:
                mean_val = treated_data[col].mean()
                std_val = treated_data[col].std()
                if std_val > 0:  # Avoid division by zero
                    treated_data[col] = (treated_data[col] - mean_val) / std_val * 0.8 + mean_val
            effect_description['normalized'].append('Beta waves (stress reduction)')
            
        if alpha_cols:
            for col in alpha_cols:
                treated_data[col] = treated_data[col] * 1.15  # Increase alpha by 15%
            effect_description['increased'].append('Alpha waves (relaxation)')
    
    elif treatment_type == 'meditation':
        # Simulate meditation effect (increase alpha, decrease beta, slight increase in theta)
        alpha_cols = [col for col in treated_data.columns if 'alpha' in col]
        beta_cols = [col for col in treated_data.columns if 'beta' in col]
        theta_cols = [col for col in treated_data.columns if 'theta' in col]
        
        if alpha_cols:
            for col in alpha_cols:
                treated_data[col] = treated_data[col] * 1.4  # Significant increase in alpha
            effect_description['increased'].append('Alpha waves (deep relaxation)')
            
        if beta_cols:
            for col in beta_cols:
                treated_data[col] = treated_data[col] * 0.7  # Significant decrease in beta
            effect_description['decreased'].append('Beta waves (mental activity)')
            
        if theta_cols:
            for col in theta_cols:
                treated_data[col] = treated_data[col] * 1.1  # Slight increase in theta
            effect_description['increased'].append('Theta waves (meditative state)')
    
    elif treatment_type == 'sleep':
        # Simulate sleep effect (increase delta, decrease beta)
        delta_cols = [col for col in treated_data.columns if 'delta' in col]
        beta_cols = [col for col in treated_data.columns if 'beta' in col]
        
        if delta_cols:
            for col in delta_cols:
                treated_data[col] = treated_data[col] * 1.5  # Significant increase in delta
            effect_description['increased'].append('Delta waves (deep sleep)')
            
        if beta_cols:
            for col in beta_cols:
                treated_data[col] = treated_data[col] * 0.5  # Significant decrease in beta
            effect_description['decreased'].append('Beta waves (wakefulness)')
    
    # Get post-treatment prediction
    treated_labels, treated_probs, treated_band_values = predict_disorders(model, treated_data, label_encoder)
    
    # Calculate band changes for visualization
    band_changes = {}
    for band in treated_band_values:
        if band in original_band_values:
            original_avg = np.mean(original_band_values[band])
            treated_avg = np.mean(treated_band_values[band])
            percent_change = ((treated_avg - original_avg) / original_avg) * 100 if original_avg != 0 else 0
            band_changes[band] = {
                'original': original_avg,
                'treated': treated_avg,
                'percent_change': percent_change
            }
    
    return {
        'original_diagnosis': original_labels[0],
        'original_confidence': np.max(original_probs[0]) * 100,
        'treated_diagnosis': treated_labels[0],
        'treated_confidence': np.max(treated_probs[0]) * 100,
        'treatment_type': treatment_type,
        'effect_description': effect_description,
        'band_changes': band_changes,
        'original_band_values': original_band_values,
        'treated_band_values': treated_band_values
    }

def main():
    """Main function for Streamlit app"""
    st.set_page_config(page_title="EEG Cognitive Analysis", layout="wide")
    
    st.title("🧠 EEG Cognitive Analysis System")
    st.write("Advanced EEG analysis and cognitive disorder prediction")
    
    try:
        # Simplified interface - only PDF upload
        st.subheader("📄 EEG Report Input")
        uploaded_file = st.file_uploader("Upload EEG PDF Report", type="pdf")
        
        if uploaded_file is not None:
            with st.spinner("Parsing PDF to JSON..."):
                try:
                    pdf_content = uploaded_file.read()
                    json_data, output_path = parse_pdf_to_json(pdf_content)
                    
                    if not json_data:
                        # If PDF parsing fails, use default values
                        st.warning("Could not extract text from PDF. Using default values.")
                        
                        # Create default JSON data with sample values
                        json_data = {
                            "patient_info": {
                                "name": "Sample Patient",
                                "age": 35,
                                "sex": "M",
                                "date": "2025-04-13"
                            },
                            "eeg_data": {
                                "AB.delta.Fp1": 15.2, "AB.delta.Fp2": 14.8, "AB.delta.F7": 16.1,
                                "AB.delta.F3": 15.7, "AB.delta.Fz": 15.9, "AB.delta.F4": 15.8,
                                "AB.delta.F8": 16.0, "AB.delta.T3": 14.9, "AB.delta.C3": 15.3,
                                "AB.delta.Cz": 15.5, "AB.delta.C4": 15.4, "AB.delta.T4": 15.0,
                                "AB.delta.T5": 14.7, "AB.delta.P3": 15.6, "AB.delta.Pz": 15.2,
                                "AB.delta.P4": 15.1, "AB.delta.T6": 14.6, "AB.delta.O1": 14.5,
                                "AB.delta.O2": 14.4,
                                
                                "AB.theta.Fp1": 8.2, "AB.theta.Fp2": 8.1, "AB.theta.F7": 8.5,
                                "AB.theta.Fp2": 8.1, "AB.theta.F7": 8.5, "AB.theta.F3": 8.4,
                                "AB.theta.F8": 8.7, "AB.theta.T3": 8.0, "AB.theta.C3": 8.8,
                                "AB.theta.Cz": 8.9, "AB.theta.C4": 8.2, "AB.theta.T4": 8.1,
                                "AB.theta.T5": 8.3, "AB.theta.P3": 8.4, "AB.theta.Pz": 8.5,
                                "AB.theta.P4": 8.6, "AB.theta.T6": 8.7, "AB.theta.O1": 8.8,
                                "AB.theta.O2": 8.9,
                                
                                "AB.alpha.Fp1": 12.1, "AB.alpha.Fp2": 12.2, "AB.alpha.F7": 12.3,
                                "AB.alpha.Fp2": 12.2, "AB.alpha.F7": 12.3, "AB.alpha.F3": 12.4,
                                "AB.alpha.Fz": 12.5, "AB.alpha.F4": 12.6,
                                "AB.alpha.F8": 12.7, "AB.alpha.T3": 12.8, "AB.alpha.C3": 12.9,
                                "AB.alpha.Cz": 13.0, "AB.alpha.C4": 13.1, "AB.alpha.T4": 13.2,
                                "AB.alpha.T5": 13.3, "AB.alpha.P3": 13.4, "AB.alpha.Pz": 13.5,
                                "AB.alpha.P4": 13.6, "AB.alpha.T6": 13.7, "AB.alpha.O1": 13.8,
                                "AB.alpha.O2": 13.9,
                                
                                "AB.beta.Fp1": 18.1, "AB.beta.Fp2": 18.2, "AB.beta.F7": 18.3,
                                "AB.beta.Fp2": 18.2, "AB.beta.F7": 18.3, "AB.beta.F3": 18.4,
                                "AB.beta.Fz": 18.5, "AB.beta.F4": 18.6,
                                "AB.beta.F8": 18.7, "AB.beta.T3": 18.8, "AB.beta.C3": 18.9,
                                "AB.beta.Cz": 19.0, "AB.beta.C4": 19.1, "AB.beta.T4": 19.2,
                                "AB.beta.T5": 19.3, "AB.beta.P3": 19.4, "AB.beta.Pz": 19.5,
                                "AB.beta.P4": 19.6, "AB.beta.T6": 19.7, "AB.beta.O1": 19.8,
                                "AB.beta.O2": 19.9,
                                
                                "AB.highbeta.Fp1": 10.1, "AB.highbeta.Fp2": 10.2, "AB.highbeta.F7": 10.3,
                                "AB.highbeta.Fp2": 10.2, "AB.highbeta.F7": 10.3, "AB.highbeta.F8": 10.7,
                                "AB.highbeta.T3": 10.8, "AB.highbeta.C3": 10.9,
                                "AB.highbeta.Cz": 11.0, "AB.highbeta.C4": 11.1, "AB.highbeta.T4": 11.2,
                                "AB.highbeta.T5": 11.3, "AB.highbeta.P3": 11.4, "AB.highbeta.Pz": 11.5,
                                "AB.highbeta.P4": 11.6, "AB.highbeta.T6": 11.7, "AB.highbeta.O1": 11.8,
                                "AB.highbeta.O2": 11.9,
                                
                                "AB.gamma.Fp1": 4.1, "AB.gamma.Fp2": 4.2, "AB.gamma.F7": 4.3,
                                "AB.gamma.Fp1": 4.1, "AB.gamma.Fp2": 4.2, "AB.gamma.F7": 4.3,
                                "AB.gamma.F3": 4.4, "AB.gamma.Fz": 4.5, "AB.gamma.F4": 4.6,
                                "AB.gamma.F8": 4.7, "AB.gamma.T3": 4.8, "AB.gamma.C3": 4.9,
                                "AB.gamma.Cz": 5.0, "AB.gamma.C4": 5.1, "AB.gamma.T4": 5.2,
                                "AB.gamma.T5": 5.3, "AB.gamma.P3": 5.4, "AB.gamma.Pz": 5.5,
                                "AB.gamma.P4": 5.6, "AB.gamma.T6": 5.7, "AB.gamma.O1": 5.8,
                                "AB.gamma.O2": 5.9
                            }
                        }
                        
                        # Save default data to a temporary file
                        os.makedirs("d:/NeuroGPT/data/reports", exist_ok=True)
                        output_path = f"d:/NeuroGPT/data/reports/EEG_default_{uploaded_file.name}.json"
                        with open(output_path, 'w') as f:
                            json.dump(json_data, f, indent=4)
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
                    return
                
                selected_file = output_path
        else:
            st.info("Please upload a PDF file to continue.")
            return
        
        # Rest of the code remains the same
        with st.spinner("Analyzing EEG data..."):
            # Load and process data
            with open(selected_file, 'r') as f:
                json_data = json.load(f)
            
            eeg_data = pd.DataFrame([json_data['eeg_data']])
            eeg_data['age'] = json_data['patient_info']['age']
            eeg_data['sex'] = 1 if json_data['patient_info']['sex'] == 'M' else 0
            
            # Load model
            model_path = "d:/NeuroGPT/models/cognitive_model.pt"
            checkpoint = torch.load(model_path)
            
            total_features = len([col for col in eeg_data.columns 
                                if col.startswith(('AB.', 'COH.'))])
            
            model = EEGCognitiveClassifier(
                input_features=total_features,
                num_classes=7
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            label_encoder = checkpoint['label_encoder']
            
            # Make prediction
            labels, probabilities, band_values = predict_disorders(model, eeg_data, label_encoder)
            
            # Display results in columns
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📋 Patient Information")
                st.json(json_data["patient_info"])
                
                st.subheader("🎯 Primary Diagnosis")
                st.markdown(f"""
                *Predicted Condition:* {labels[0]}  
                *Confidence:* {np.max(probabilities[0])*100:.2f}%
                """)
            
            with col2:
                st.subheader("📊 Disorder Probabilities")
                for disorder, prob in zip(label_encoder.classes_, probabilities[0]):
                    prob_pct = prob * 100
                    color = "green" if disorder == labels[0] else "gray"
                    st.markdown(f"""
                    <div style='display: flex; justify-content: space-between; color: {color};'>
                        <span>{disorder}</span>
                        <span>{prob_pct:.2f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Display EEG visualization
            st.image('d:/NeuroGPT/outputs/eeg_analysis.png')
            
            # Expert Analysis
            st.subheader("🔬 Expert EEG Analysis")
            
            # Update the prediction results dictionary
            prediction_results = {
                'primary_diagnosis': labels[0],
                'confidence': np.max(probabilities[0]) * 100
            }
            
            # Get expert analysis
            analysis, band_averages = get_eeg_expert_analysis(json_data['eeg_data'], prediction_results)
            
            # Display analysis
            st.subheader("🧠 EEG Band Analysis")
            
            # Create a more visually appealing band analysis with gauges
            cols = st.columns(3)
            for i, (band, value) in enumerate(band_averages.items()):
                # Determine if the value is high, normal, or low
                if band == 'alpha':
                    status = "Low" if value < 10 else "Normal" if value < 25 else "High"
                    delta_color = "inverse" if value < 10 else "normal" if value < 25 else "inverse"
                    info = "Relaxation" if value >= 10 else "Anxiety/Stress"
                elif band == 'beta':
                    status = "Low" if value < 5 else "Normal" if value < 20 else "High"
                    delta_color = "normal" if value < 20 else "inverse"
                    info = "Focus" if value < 20 else "Stress/Anxiety"
                elif band == 'theta':
                    status = "Low" if value < 5 else "Normal" if value < 15 else "High"
                    delta_color = "inverse" if value < 5 else "normal" if value < 15 else "inverse"
                    info = "Meditation/Creativity" if value >= 5 and value < 15 else "Drowsiness"
                else:
                    status = "Normal"
                    delta_color = "normal"
                    info = ""
                
                cols[i % 3].metric(
                    f"{band.title()} Band", 
                    f"{value:.2f} μV²", 
                    delta=status,
                    delta_color=delta_color
                )
                cols[i % 3].caption(f"{info}")
            
            # Create a table with band information
            st.subheader("📊 EEG Band Information")
            band_info = {
                "Band": ["Delta (0.5-4 Hz)", "Theta (4-8 Hz)", "Alpha (8-13 Hz)", "Beta (13-30 Hz)", "Gamma (>30 Hz)"],
                "Associated With": ["Deep sleep, healing", "Drowsiness, meditation", "Relaxation, calmness", "Active thinking, focus", "Higher cognitive processing"],
                "Your Level": [
                    f"{band_averages['delta']:.2f} μV² ({'High' if band_averages['delta'] > 20 else 'Normal' if band_averages['delta'] > 5 else 'Low'})",
                    f"{band_averages['theta']:.2f} μV² ({'High' if band_averages['theta'] > 15 else 'Normal' if band_averages['theta'] > 5 else 'Low'})",
                    f"{band_averages['alpha']:.2f} μV² ({'High' if band_averages['alpha'] > 25 else 'Normal' if band_averages['alpha'] > 10 else 'Low'})",
                    f"{band_averages['beta']:.2f} μV² ({'High' if band_averages['beta'] > 20 else 'Normal' if band_averages['beta'] > 5 else 'Low'})",
                    f"{band_averages['gamma']:.2f} μV² ({'High' if band_averages['gamma'] > 5 else 'Normal' if band_averages['gamma'] > 1 else 'Low'})"
                ]
            }
            st.table(pd.DataFrame(band_info))
            
            st.subheader("📋 Clinical Analysis")
            st.markdown(analysis)
            
            # Treatment Simulation - Enhanced Interactive Version
            st.subheader("💊 Brain Intervention Simulator")
            st.markdown("See how different interventions might affect your brain patterns and mental state.")
            
            # Create interactive treatment selection with icons
            col1, col2, col3, col4 = st.columns(4)
            
            treatment_selected = None
            
            with col1:
                med_btn = st.button("💊 Medication")
                if med_btn:
                    treatment_selected = "medication"
                st.caption("Simulates effect of psychiatric medication")
                
            with col2:
                therapy_btn = st.button("🛋 Therapy")
                if therapy_btn:
                    treatment_selected = "therapy"
                st.caption("Simulates effect of cognitive behavioral therapy")
                
            with col3:
                meditation_btn = st.button("🧘 Meditation")
                if meditation_btn:
                    treatment_selected = "meditation"
                st.caption("Simulates effect of regular meditation practice")
                
            with col4:
                sleep_btn = st.button("😴 Sleep")
                if sleep_btn:
                    treatment_selected = "sleep"
                st.caption("Simulates effect of proper sleep")
            
            if treatment_selected:
                with st.spinner(f"Simulating effects of {treatment_selected}..."):
                    treatment_results = simulate_treatment_effect(
                        model, eeg_data, label_encoder, treatment_selected
                    )
                    
                    # Create tabs for different views of the results
                    tab1, tab2, tab3 = st.tabs(["Summary", "Brain Wave Changes", "Detailed Analysis"])
                    
                    with tab1:
                        # Summary view
                        st.subheader(f"✨ {treatment_selected.title()} Simulation Results")
                        
                        # Create two columns for before/after
                        before_col, after_col = st.columns(2)
                        
                        with before_col:
                            st.markdown("### Before Intervention")
                            st.markdown(f"""
                            *Diagnosis:* {treatment_results['original_diagnosis']}  
                            *Confidence:* {treatment_results['original_confidence']:.1f}%
                            """)
                            
                            # Create a simple gauge for stress level (based on beta/alpha ratio)
                            original_beta = np.mean(treatment_results['original_band_values']['beta'])
                            original_alpha = np.mean(treatment_results['original_band_values']['alpha'])
                            stress_ratio = original_beta / original_alpha if original_alpha > 0 else 5
                            
                            # Normalize to 0-100 scale
                            stress_level = min(100, max(0, (stress_ratio - 1) * 20))
                            st.markdown(f"*Stress Level:* {stress_level:.0f}%")
                            st.progress(stress_level/100)
                        
                        with after_col:
                            st.markdown("### After Intervention")
                            st.markdown(f"""
                            *Diagnosis:* {treatment_results['treated_diagnosis']}  
                            *Confidence:* {treatment_results['treated_confidence']:.1f}%
                            """)
                            
                            # Create a simple gauge for stress level after treatment
                            treated_beta = np.mean(treatment_results['treated_band_values']['beta'])
                            treated_alpha = np.mean(treatment_results['treated_band_values']['alpha'])
                            treated_stress_ratio = treated_beta / treated_alpha if treated_alpha > 0 else 5
                            
                            # Normalize to 0-100 scale
                            treated_stress_level = min(100, max(0, (treated_stress_ratio - 1) * 20))
                            st.markdown(f"*Stress Level:* {treated_stress_level:.0f}%")
                            st.progress(treated_stress_level/100)
                            
                            # Show improvement
                            improvement = stress_level - treated_stress_level
                            if improvement > 0:
                                st.success(f"🎉 {improvement:.0f}% Stress Reduction")
                        
                        # Personalized recommendation
                        st.markdown("### 💡 Personalized Insight")
                        
                        if treatment_selected == "meditation":
                            st.info("""
                            *Your stress level is high.* Based on your EEG pattern, regular meditation could help 
                            increase your alpha waves and reduce excessive beta activity. This simulation shows how 
                            your brain might respond to a consistent meditation practice.
                            
                            *Recommendation:* Try 10-15 minutes of guided meditation daily for 4 weeks.
                            """)
                        elif treatment_selected == "medication":
                            st.info("""
                            *Your EEG shows imbalanced activity.* This simulation suggests medication might help 
                            normalize your brain wave patterns, particularly by reducing excessive beta activity 
                            and supporting healthy alpha wave production.
                            
                            *Recommendation:* Discuss medication options with a psychiatrist.
                            """)
                        elif treatment_selected == "therapy":
                            st.info("""
                            *Your brain activity suggests cognitive stress.* This simulation shows how cognitive 
                            behavioral therapy might help balance your brain wave patterns over time, teaching your 
                            brain to maintain healthier activity patterns.
                            
                            *Recommendation:* Consider 8-12 sessions of cognitive behavioral therapy.
                            """)
                        elif treatment_selected == "sleep":
                            st.info("""
                            *Your EEG suggests sleep deprivation.* This simulation shows how proper sleep hygiene 
                            could significantly improve your brain wave patterns, particularly by increasing delta 
                            waves and allowing your brain to properly recover.
                            
                            *Recommendation:* Aim for 7-8 hours of uninterrupted sleep nightly.
                            """)
                    
                    with tab2:
                        # Brain Wave Changes
                        st.subheader("🧠 Brain Wave Changes")
                        
                        # Create a bar chart showing before/after for each band
                        band_data = {
                            'Band': [],
                            'Before': [],
                            'After': [],
                            'Change (%)': []
                        }
                        
                        for band, values in treatment_results['band_changes'].items():
                            band_data['Band'].append(band.capitalize())
                            band_data['Before'].append(values['original'])
                            band_data['After'].append(values['treated'])
                            band_data['Change (%)'].append(values['percent_change'])
                        
                        band_df = pd.DataFrame(band_data)
                        
                        # Plot the changes
                        st.bar_chart(band_df.set_index('Band')[['Before', 'After']])
                        
                        # Show a table with the percent changes
                        st.table(band_df[['Band', 'Before', 'After', 'Change (%)']])
                        
                        # Explain the changes
                        st.subheader("📝 What Changed?")
                        
                        for change_type, bands in treatment_results['effect_description'].items():
                            if bands:
                                if change_type == 'increased':
                                    st.markdown(f"*Increased:* {', '.join(bands)}")
                                elif change_type == 'decreased':
                                    st.markdown(f"*Decreased:* {', '.join(bands)}")
                                elif change_type == 'normalized':
                                    st.markdown(f"*Normalized:* {', '.join(bands)}")
                    
                    with tab3:
                        # Detailed Analysis
                        st.subheader("🔍 Detailed Analysis")
                        
                        # Generate a detailed analysis using the Gemini model
                        treatment_prompt = f"""
                        As an EEG expert system, analyze the effects of {treatment_selected} on this patient:

                        Original EEG Band Averages:
                        Delta (0.5-4 Hz): {np.mean(treatment_results['original_band_values']['delta']):.2f} μV²
                        Theta (4-8 Hz): {np.mean(treatment_results['original_band_values']['theta']):.2f} μV²
                        Alpha (8-13 Hz): {np.mean(treatment_results['original_band_values']['alpha']):.2f} μV²
                        Beta (13-30 Hz): {np.mean(treatment_results['original_band_values']['beta']):.2f} μV²
                        High Beta (20-30 Hz): {np.mean(treatment_results['original_band_values']['highbeta']):.2f} μV²
                        Gamma (>30 Hz): {np.mean(treatment_results['original_band_values']['gamma']):.2f} μV²

                        Post-{treatment_selected} EEG Band Averages:
                        Delta (0.5-4 Hz): {np.mean(treatment_results['treated_band_values']['delta']):.2f} μV²
                        Theta (4-8 Hz): {np.mean(treatment_results['treated_band_values']['theta']):.2f} μV²
                        Alpha (8-13 Hz): {np.mean(treatment_results['treated_band_values']['alpha']):.2f} μV²
                        Beta (13-30 Hz): {np.mean(treatment_results['treated_band_values']['beta']):.2f} μV²
                        High Beta (20-30 Hz): {np.mean(treatment_results['treated_band_values']['highbeta']):.2f} μV²
                        Gamma (>30 Hz): {np.mean(treatment_results['treated_band_values']['gamma']):.2f} μV²

                        Original Diagnosis: {treatment_results['original_diagnosis']}
                        Post-{treatment_selected} Diagnosis: {treatment_results['treated_diagnosis']}

                        Provide a detailed clinical analysis of how {treatment_selected} affected this patient's brain activity.
                        Explain the significance of the changes in each frequency band and what it means for the patient's mental health.
                        Include specific recommendations for the patient based on these results.
                        """
                        
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        detailed_response = model.generate_content(treatment_prompt)
                        detailed_analysis = detailed_response.text
                        
                        st.markdown(detailed_analysis)
            
            # Add this import near the top with other imports
            from datetime import datetime
            import sys
            sys.path.append("d:/NeuroGPT/report")
            from doctor_report import display_report_section
            
            # Then in your main function, after displaying the analysis and before the "Additional Resources" section,
            # add this code to integrate the report generation feature:
            
            # Add Doctor's Report section
            if 'treatment_selected' in locals() and treatment_selected:
                display_report_section(
                    json_data["patient_info"],
                    prediction_results,
                    band_averages,
                    analysis,
                    treatment_results
                )
            else:
                display_report_section(
                    json_data["patient_info"],
                    prediction_results,
                    band_averages,
                    analysis
                )
            
            # Additional Resources
            st.subheader("📚 Additional Resources")
            st.markdown("""
            - [National Institute of Mental Health](https://www.nimh.nih.gov)
            - [American Academy of Neurology](https://www.aan.com)
            - [International League Against Epilepsy](https://www.ilae.org)
            - [Brain & Behavior Research Foundation](https://www.bbrfoundation.org)
            """)
            
    except Exception as e:
        st.error(f"Visualization Error: {str(e)}")
        st.info("Please ensure all required files and models are available.")

if __name__ == "__main__":
    main()