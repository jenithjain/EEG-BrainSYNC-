from predict_cognitive import predict_disorders
from model.eeg_classifier import EEGCognitiveClassifier
from visualize_eeg import create_eeg_visualization, save_visualizations
import torch
import pandas as pd
import sys
import os
import numpy as np

# Add parent directory to path for imports
script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.dirname(script_path))

# Load your model and data
model_path = "d:/NeuroGPT/models/cognitive_model.pt"
checkpoint = torch.load(model_path)

# Get input features from the first layer's weight shape
input_features = checkpoint['model_state_dict']['layers.0.weight'].shape[1]

# Initialize model with correct input size
model = EEGCognitiveClassifier(
    input_features=input_features,
    num_classes=7
)
model.load_state_dict(checkpoint['model_state_dict'])
label_encoder = checkpoint['label_encoder']

# Use sample data instead of reading from CSV
# Create sample EEG data with clinically relevant values
# Create comprehensive sample input data
# Update channel names to match MNE's standard naming
channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 
            'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']

sample_data = {}

# Add Absolute Power (AB) features with specific values
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

# Get predictions
labels, probabilities = predict_disorders(model, test_data, label_encoder)

# Prepare band values for visualization
band_values = {}
for band in ['delta', 'theta', 'alpha', 'beta', 'highbeta', 'gamma']:
    cols = [col for col in test_data.columns if band in col.lower()]
    band_values[band] = test_data[cols].values[0]

# Create and save visualizations
create_eeg_visualization(test_data, labels[0], band_values)
save_visualizations('d:/NeuroGPT/outputs/eeg_analysis.png')