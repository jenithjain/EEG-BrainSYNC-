import os
import sys
import torch

# Add parent directory to Python path
script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.dirname(script_path))

from evaluate import evaluate_model
from analyze_features import analyze_feature_importance
from preprocess import preprocess_data
from model.eeg_classifier import EEGCognitiveClassifier

def run_complete_analysis():
    # Load data and model
    data = preprocess_data("d:/NeuroGPT/data/EEG.machinelearing_data_BRMH.csv")
    checkpoint = torch.load("d:/NeuroGPT/models/best_model.pt")
    
    # Create model instance
    model = EEGCognitiveClassifier(
        eeg_features=data['eeg_shape'],
        demo_features=data['demo_shape'],
        num_classes=7
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Run evaluations
    evaluate_model(model, data, "d:/NeuroGPT/results")
    analyze_feature_importance(model, data, "d:/NeuroGPT/results")

if __name__ == "__main__":
    run_complete_analysis()