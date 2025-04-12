import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from captum.attr import IntegratedGradients

def analyze_feature_importance(model, data, save_dir):
    # Create results directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Prepare input tensors
    X_eeg = torch.FloatTensor(data['X_eeg'])
    X_demo = torch.FloatTensor(data['X_demo'])
    
    # Initialize integrated gradients
    ig = IntegratedGradients(model)
    
    # Analyze EEG features
    eeg_importance = np.abs(ig.attribute(X_eeg, target=data['y']).detach().numpy())
    demo_importance = np.abs(ig.attribute(X_demo, target=data['y']).detach().numpy())
    
    # Plot EEG feature importance
    plt.figure(figsize=(15, 10))
    
    # EEG bands plot
    plt.subplot(2, 1, 1)
    bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'High Beta', 'Gamma']
    band_importance = np.mean(eeg_importance.reshape(len(eeg_importance), len(bands), -1), axis=2)
    sns.boxplot(data=band_importance)
    plt.title('EEG Band Importance')
    plt.xticks(range(len(bands)), bands)
    
    # Demographic features plot
    plt.subplot(2, 1, 2)
    demo_features = ['Age', 'Education', 'IQ', 'Sex']
    demo_mean = np.mean(demo_importance, axis=0)
    plt.bar(demo_features, demo_mean)
    plt.title('Demographic Feature Importance')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/feature_importance.png")
    
    # Save numerical results
    with open(f"{save_dir}/feature_importance.txt", 'w') as f:
        f.write("EEG Band Importance (mean):\n")
        for band, imp in zip(bands, np.mean(band_importance, axis=0)):
            f.write(f"{band}: {imp:.4f}\n")
        
        f.write("\nDemographic Feature Importance:\n")
        for feat, imp in zip(demo_features, demo_mean):
            f.write(f"{feat}: {imp:.4f}\n")