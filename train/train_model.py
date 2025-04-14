import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from preprocess import preprocess_data
import os
import sys

# Add the directory containing the 'model' module to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model.eeg_classifier import EEGCognitiveClassifier

def train_model(data_path, save_dir, n_folds=5):
    # Preprocess data
    data = preprocess_data(data_path)
    
    # Setup cross-validation
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    best_models = []
    best_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(data['X_eeg'], data['y'])):
        # Split data
        X_eeg_train = data['X_eeg'][train_idx]
        X_demo_train = data['X_demo'][train_idx]
        y_train = data['y'][train_idx]
        
        X_eeg_val = data['X_eeg'][val_idx]
        X_demo_val = data['X_demo'][val_idx]
        y_val = data['y'][val_idx]
        
        # Create model
        model = EEGCognitiveClassifier(
            eeg_features=data['eeg_shape'],
            demo_features=data['demo_shape'],
            num_classes=len(np.unique(data['y']))
        )
        
        # Training setup
        criterion = nn.CrossEntropyLoss(weight=compute_class_weights(y_train))
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5)
        
        # Training loop
        best_val_f1 = 0
        patience = 10
        patience_counter = 0
        
        for epoch in range(100):
            model.train()
            
            # Mini-batch training
            batch_size = 32
            for i in range(0, len(X_eeg_train), batch_size):
                batch_eeg = torch.FloatTensor(X_eeg_train[i:i+batch_size])
                batch_demo = torch.FloatTensor(X_demo_train[i:i+batch_size])
                batch_y = torch.LongTensor(y_train[i:i+batch_size])
                
                optimizer.zero_grad()
                outputs = model(batch_eeg, batch_demo)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(
                    torch.FloatTensor(X_eeg_val),
                    torch.FloatTensor(X_demo_val)
                )
                val_preds = torch.argmax(val_outputs, dim=1).numpy()
                val_f1 = f1_score(y_val, val_preds, average='weighted')
                
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_models.append({
                        'model_state': model.state_dict(),
                        'score': val_f1,
                        'fold': fold
                    })
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    break
                
                scheduler.step(val_f1)
    
    # Save best model
    best_model_idx = np.argmax([m['score'] for m in best_models])
    torch.save({
        'model_state_dict': best_models[best_model_idx]['model_state'],
        'scaler_eeg': data['scaler_eeg'],
        'scaler_demo': data['scaler_demo'],
        'selector': data['selector'],
        'pca': data['pca'],
        'label_encoder': data['label_encoder'],
        'best_score': best_models[best_model_idx]['score']
    }, f"{save_dir}/best_model.pt")

def compute_class_weights(y):
    classes, counts = np.unique(y, return_counts=True)
    weights = 1.0 / counts
    weights = weights / weights.sum()
    return torch.FloatTensor(weights)

if __name__ == "__main__":
    train_model("d:/NeuroGPT/data/EEG.machinelearing_data_BRMH.csv", 
                "d:/NeuroGPT/models")