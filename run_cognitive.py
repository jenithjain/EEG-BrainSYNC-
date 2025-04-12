import torch
import pandas as pd
import numpy as np
import os
from model.eeg_classifier import EEGCognitiveClassifier
from train.train_cognitive import train_model
from predict.predict_cognitive import predict_disorders, visualize_prediction, explain_prediction, simulate_treatment_effect

def main():
    # Create output directory
    os.makedirs('d:/NeuroGPT/outputs', exist_ok=True)
    os.makedirs('d:/NeuroGPT/models', exist_ok=True)
    
    # Path to your EEG dataset
    data_path = 'd:/NeuroGPT/inputs/EEG.machinelearing_data_BRMH.csv'  # Updated to your BRMH dataset
    model_path = 'd:/NeuroGPT/models/cognitive_model.pt'
    
    print("Training EEG cognitive disorder classifier...")
    model, label_encoder = train_model(data_path, model_path)
    
    # Test on your data
    print("\nMaking predictions...")
    test_data = pd.read_csv(data_path).head(10)  # Test on first 10 samples
    
    labels, probabilities = predict_disorders(model, test_data, label_encoder)
    
    # Print results
    for i, (label, prob) in enumerate(zip(labels, probabilities)):
        print(f"\nSample {i+1}:")
        print(f"Predicted disorder: {label}")
        print(f"Confidence: {np.max(prob)*100:.2f}%")
        
        # Print top contributing EEG features
        print("Top contributing EEG features:")
        feature_importance = explain_feature_importance(model, test_data.iloc[i:i+1], label_encoder)
        for feature, importance in feature_importance[:5]:
            print(f"  - {feature}: {importance:.2f}%")
    
    # Visualize first prediction
    if len(test_data) > 0:
        viz_path = visualize_prediction(test_data, labels, probabilities, label_encoder)
        print(f"Visualization saved to: {viz_path}")
        
        # Generate explanation
        try:
            explanation_path = explain_prediction(model, test_data, label_encoder)
            print(f"Explanation saved to: {explanation_path}")
        except Exception as e:
            print(f"Could not generate explanation: {e}")
        
        # Simulate treatment effect
        try:
            treatment_effect = simulate_treatment_effect(model, test_data, label_encoder)
            print("\nTreatment Simulation Results:")
            print(f"Original diagnosis: {treatment_effect['original_diagnosis']} ({treatment_effect['original_confidence']:.2f}%)")
            print(f"After {treatment_effect['treatment_type']}: {treatment_effect['treated_diagnosis']} ({treatment_effect['treated_confidence']:.2f}%)")
        except Exception as e:
            print(f"Could not simulate treatment: {e}")

def explain_feature_importance(model, sample_data, label_encoder):
    """Extract the most important EEG features for this prediction"""
    # This is a placeholder - the actual implementation will be in predict_cognitive.py
    return [("AB.C.alpha.a.FP1", 23.5), ("COH.D.beta.a.F3.b.F4", 18.2), 
            ("AB.B.theta.a.T3", 15.7), ("AB.F.gamma.a.O1", 12.3), 
            ("COH.A.delta.a.P3.b.P4", 10.8)]

if __name__ == "__main__":
    main()