import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
import sys

script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.dirname(script_path))

from model.eeg_classifier import EEGCognitiveClassifier
from visualize_eeg import create_eeg_visualization, save_visualizations

def prepare_eeg_data(data):
    # Handle the complex EEG features from the BRMH dataset
    if isinstance(data, pd.DataFrame):
        # Extract all EEG features (Delta, Theta, Alpha, Beta, Gamma bands)
        eeg_columns = [col for col in data.columns if any(band in col for band in 
                      ['delta', 'theta', 'alpha', 'beta', 'gamma'])]
        
        if not eeg_columns:
            # Fallback to time_len if no EEG features found
            return torch.FloatTensor(data[['time_len']].values)
        
        return torch.FloatTensor(data[eeg_columns].values)
    
    return torch.FloatTensor(data)

# Add these imports at the top
import matplotlib.pyplot as plt
from mne.viz import plot_topomap
from mne import create_info, pick_types
from mne.channels import make_standard_montage
import mne
import seaborn as sns

def create_brain_visualization(data, prediction, band_values):
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(20, 12))
    
    # Create channel positions
    montage = make_standard_montage('standard_1020')
    ch_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 
                'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
    
    info = create_info(ch_names=ch_names, sfreq=256., ch_types='eeg')
    info.set_montage(montage)
    pos = np.array([montage.get_positions()['ch_pos'][ch] for ch in ch_names])[:, :2]
    
    # Plot each frequency band
    bands = ['delta', 'theta', 'alpha', 'beta', 'highbeta', 'gamma']
    for idx, band in enumerate(bands, 1):
        ax = fig.add_subplot(2, 4, idx)
        data = band_values[band][:19]
        
        im = plot_topomap(data, pos, axes=ax, show=False,
                         cmap='RdYlBu_r', 
                         sensors=True,
                         outlines='head',
                         contours=10,
                         image_interp='cubic')
        
        plt.colorbar(im[0], ax=ax)
        ax.set_title(f'{band.capitalize()} Band\n({min(data):.1f}-{max(data):.1f} μV²)',
                    color='white', pad=15)
    
    # Add power spectrum
    ax_spectrum = fig.add_subplot(2, 4, 7)
    mean_powers = [np.mean(band_values[band][:19]) for band in bands]
    
    bars = sns.barplot(x=bands, y=mean_powers, ax=ax_spectrum, 
                      palette='RdYlBu_r', alpha=0.8)
    ax_spectrum.set_title('Average Band Powers', color='white')
    ax_spectrum.set_xticklabels(bands, rotation=45)
    ax_spectrum.set_ylabel('Power (μV²)')
    ax_spectrum.grid(True, alpha=0.2)
    
    # Add prediction info
    ax_text = fig.add_subplot(2, 4, 8)
    ax_text.text(0.5, 0.5, f'Predicted Condition:\n{prediction}', 
                ha='center', va='center', fontsize=14,
                color='white', weight='bold')
    ax_text.axis('off')
    
    plt.tight_layout()
    return fig

def save_visualization(fig, output_path):
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='black')
    plt.close(fig)

def predict_disorders(model, data, label_encoder):
    model.eval()
    with torch.no_grad():
        # Keep original data for visualization
        tensor_data = prepare_eeg_data(data)
        outputs = model(tensor_data)
        probabilities = F.softmax(outputs, dim=1)
        predictions = torch.argmax(outputs, dim=1)
        labels = label_encoder.inverse_transform(predictions.numpy())
    
    # Use DataFrame for band values extraction
    band_values = {
        'delta': np.array([data[col].values[0] for col in data.columns if 'delta' in col.lower()]),
        'theta': np.array([data[col].values[0] for col in data.columns if 'theta' in col.lower()]),
        'alpha': np.array([data[col].values[0] for col in data.columns if 'alpha' in col.lower()]),
        'beta': np.array([data[col].values[0] for col in data.columns if 'beta' in col.lower()]),
        'highbeta': np.array([data[col].values[0] for col in data.columns if 'highbeta' in col.lower()]),
        'gamma': np.array([data[col].values[0] for col in data.columns if 'gamma' in col.lower()])
    }
    
    # Create and save visualization
    # Create visualization with the new function
    fig = create_brain_visualization(data, labels[0], band_values)
    save_visualization(fig, 'd:/NeuroGPT/outputs/eeg_analysis.png')
    
    return labels, probabilities.numpy()

def visualize_prediction(data, label, probability, top_n=7):  # Changed to show all disorders
    plt.figure(figsize=(12, 6))
    
    # Create bar chart of all probabilities
    classes = label_encoder.classes_
    probs = probability[0]
    
    # Sort all predictions by probability
    indices = np.argsort(probs)[::-1]
    sorted_probs = probs[indices]
    sorted_classes = classes[indices]
    
    # Plot
    colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(sorted_classes))]
    sns.barplot(x=sorted_probs*100, y=sorted_classes, palette=colors)
    plt.xlabel('Probability (%)')
    plt.ylabel('Disorder Type')
    plt.title(f'Diagnosis: {label[0]} (Confidence: {np.max(probs)*100:.1f}%)')
    plt.tight_layout()
    
    # Save visualization
    plt.savefig('d:/NeuroGPT/outputs/diagnosis_result.png')
    plt.close()
    
    return 'd:/NeuroGPT/outputs/diagnosis_result.png'

def explain_prediction(model, data, label_encoder):
    """
    Generate SHAP values to explain model prediction
    """
    # Prepare background data for SHAP
    background = data.sample(min(50, len(data)))
    background_tensor = prepare_eeg_data(background)
    
    # Initialize explainer
    explainer = shap.DeepExplainer(model, background_tensor)
    
    # Get SHAP values for a single prediction
    sample = prepare_eeg_data(data.iloc[0:1])
    shap_values = explainer.shap_values(sample)
    
    # Get feature names
    if 'delta' in data.columns[0]:
        feature_names = [col for col in data.columns if any(band in col for band in 
                        ['delta', 'theta', 'alpha', 'beta', 'gamma'])]
    else:
        feature_names = data.columns.tolist()
    
    # Plot SHAP values
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, sample.numpy(), feature_names=feature_names, show=False)
    plt.title('EEG Features Importance for Diagnosis')
    plt.tight_layout()
    
    # Save explanation
    plt.savefig('d:/NeuroGPT/outputs/diagnosis_explanation.png')
    plt.close()
    
    return 'd:/NeuroGPT/outputs/diagnosis_explanation.png'

def simulate_treatment_effect(model, data, label_encoder, treatment_type='medication'):
    """
    Simulate effect of treatment on EEG patterns and resulting diagnosis
    """
    # Get baseline prediction
    original_labels, original_probs = predict_disorders(model, data, label_encoder)
    
    # Simulate treatment effects on EEG (simplified example)
    treated_data = data.copy()
    
    if treatment_type == 'medication':
        # Simulate medication effect (e.g., increase alpha, decrease theta)
        alpha_cols = [col for col in treated_data.columns if 'alpha' in col]
        theta_cols = [col for col in treated_data.columns if 'theta' in col]
        
        if alpha_cols:
            treated_data[alpha_cols] *= 1.2  # Increase alpha by 20%
        if theta_cols:
            treated_data[theta_cols] *= 0.8  # Decrease theta by 20%
    
    elif treatment_type == 'therapy':
        # Simulate therapy effect (e.g., normalize beta)
        beta_cols = [col for col in treated_data.columns if 'beta' in col]
        if beta_cols:
            treated_data[beta_cols] = treated_data[beta_cols].apply(
                lambda x: (x - x.mean()) / x.std() * 0.8 + x.mean())
    
    # Get post-treatment prediction
    treated_labels, treated_probs = predict_disorders(model, treated_data, label_encoder)
    
    return {
        'original_diagnosis': original_labels[0],
        'original_confidence': np.max(original_probs[0]) * 100,
        'treated_diagnosis': treated_labels[0],
        'treated_confidence': np.max(treated_probs[0]) * 100,
        'treatment_type': treatment_type
    }


if __name__ == "__main__":
    # Create comprehensive sample input data
    channels = ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'T3', 'C3', 'CZ', 
                'C4', 'T4', 'T5', 'P3', 'PZ', 'P4', 'T6', 'O1', 'O2']
    
    sample_data = {}
    
    # Add Absolute Power (AB) features
    band_values = {
        'delta': {  # 1-4 Hz
            'range': (20, 40),
            'values': [35.2, 34.8, 33.9, 34.3, 34.5, 34.2, 33.8, 33.5, 33.7, 34.0,
                      33.6, 33.2, 32.9, 33.3, 33.5, 33.2, 32.8, 32.5, 32.7]
        },
        'theta': {  # 4-8 Hz
            'range': (15, 25),
            'values': [22.3, 22.1, 21.8, 22.2, 22.4, 22.1, 21.7, 21.4, 21.6, 21.9,
                      21.5, 21.1, 20.8, 21.2, 21.4, 21.1, 20.7, 20.4, 20.6]
        },
        'alpha': {  # 8-13 Hz
            'range': (10, 20),
            'values': [18.7, 18.5, 18.2, 18.6, 18.8, 18.5, 18.1, 17.6, 18.0, 18.3,
                      17.9, 17.5, 17.2, 17.6, 17.8, 17.5, 17.1, 16.8, 17.0]
        },
        'beta': {  # 13-30 Hz
            'range': (5, 15),
            'values': [12.5, 12.3, 12.0, 12.4, 12.6, 12.3, 11.9, 11.4, 11.8, 12.1,
                      11.7, 11.3, 11.0, 11.4, 11.6, 11.3, 10.9, 10.6, 10.8]
        },
        'highbeta': {  # 20-30 Hz
            'range': (3, 10),
            'values': [8.1, 7.9, 7.6, 8.0, 8.2, 7.9, 7.5, 7.0, 7.4, 7.7,
                      7.3, 6.9, 6.6, 7.0, 7.2, 6.9, 6.5, 6.2, 6.4]
        },
        'gamma': {  # >30 Hz
            'range': (1, 5),
            'values': [3.8, 3.6, 3.3, 3.7, 3.9, 3.6, 3.2, 2.7, 3.1, 3.4,
                      3.0, 2.6, 2.3, 2.7, 2.9, 2.6, 2.2, 1.9, 2.1]
        }
    }
    
    # Add AB features
    for band, data in band_values.items():
        for ch_idx, ch in enumerate(channels):
            sample_data[f'AB.{band}.{ch}'] = data['values'][ch_idx]
    
    # Add Coherence (COH) features
    for band in band_values.keys():
        for i, ch1 in enumerate(channels):
            for ch2 in channels[i+1:]:
                # Higher coherence for adjacent channels
                if abs(channels.index(ch1) - channels.index(ch2)) <= 1:
                    sample_data[f'COH.{band}.{ch1}.{ch2}'] = np.random.uniform(0.6, 0.9)
                else:
                    sample_data[f'COH.{band}.{ch1}.{ch2}'] = np.random.uniform(0.2, 0.5)
    
    # Add demographic data
    sample_data.update({
        'age': 35,
        'education': 16,
        'iq': 105,
        'sex': 'M'
    })
    
    # Convert to DataFrame
    test_data = pd.DataFrame([sample_data])
    
    # Create output directory
    os.makedirs("d:/NeuroGPT/outputs", exist_ok=True)
    
    try:
        # Load model
        model_path = "d:/NeuroGPT/models/cognitive_model.pt"
        checkpoint = torch.load(model_path)
        
        # Get total number of features (AB + COH)
        total_features = len([col for col in test_data.columns 
                            if col.startswith(('AB.', 'COH.'))])
        
        # Initialize model with correct number of features
        model = EEGCognitiveClassifier(
            input_features=total_features,  # This matches the trained model's input size
            num_classes=7
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        label_encoder = checkpoint['label_encoder']
        
        # Make prediction
        labels, probabilities = predict_disorders(model, test_data, label_encoder)
        
        # Display results
        print("\nPrediction Results:")
        print("-" * 50)
        print(f"Input Data Summary:")
        print("EEG Bands (Average Values):")
        for band in ['delta', 'theta', 'alpha', 'beta', 'highbeta', 'gamma']:
            cols = [col for col in test_data.columns if band in col.lower()]
            if cols:
                avg = test_data[cols].mean(axis=1).values[0]
                print(f"{band.capitalize()}: {avg:.2f}")
        
        print("\nDemographic Information:")
        print(f"Age: {test_data['age'].values[0]}")
        print(f"Education: {test_data['education'].values[0]} years")
        print(f"IQ: {test_data['iq'].values[0]}")
        print(f"Sex: {test_data['sex'].values[0]}")
        
        print("\nDiagnosis:")
        print(f"Predicted Disorder: {labels[0]}")
        print(f"Confidence: {np.max(probabilities[0])*100:.2f}%")
        
        print("\nProbabilities for all disorders:")
        for disorder, prob in zip(label_encoder.classes_, probabilities[0]):
            print(f"{disorder}: {prob*100:.2f}%" + 
                  (" (Predicted)" if disorder == labels[0] else ""))
        
        # Generate visualizations
        viz_path = visualize_prediction(test_data, labels, probabilities)
        print(f"\nVisualization saved to: {viz_path}")
        
        # Generate explanation
        explanation_path = explain_prediction(model, test_data, label_encoder)
        print(f"Explanation plot saved to: {explanation_path}")
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        print("Please ensure the model is trained and saved correctly.")


def generate():
    # After getting predictions
    band_values = {
        'delta': eeg_data['delta'],
        'theta': eeg_data['theta'],
        'alpha': eeg_data['alpha'],
        'beta': eeg_data['beta'],
        'highbeta': eeg_data['highbeta'],
        'gamma': eeg_data['gamma']
    }
    
    # Create visualizations
    create_eeg_visualization(eeg_data, prediction, band_values)
    
    # Save if needed
    save_visualizations('eeg_analysis_results.png')