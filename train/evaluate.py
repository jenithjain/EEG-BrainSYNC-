import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os

def evaluate_model(model, data, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    with torch.no_grad():
        outputs = model(
            torch.FloatTensor(data['X_eeg']),
            torch.FloatTensor(data['X_demo'])
        )
        predictions = torch.argmax(outputs, dim=1).numpy()
        
        # Generate confusion matrix
        cm = confusion_matrix(data['y'], predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f"{save_dir}/confusion_matrix.png")
        plt.close()
        
        # Save classification report
        report = classification_report(data['y'], predictions)
        with open(f"{save_dir}/classification_report.txt", 'w') as f:
            f.write(report)