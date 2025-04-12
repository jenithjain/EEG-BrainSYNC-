import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from torch.nn import functional as F
import sys
import os
script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(script_path)))

# Update the import statement
from model.eeg_classifier import EEGCognitiveClassifier
import os
import re
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def plot_training_metrics(train_losses, val_losses, accuracies, save_dir):
    """Plot and save training metrics"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(accuracies, label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_metrics.png'))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes, save_dir):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()

def train_model(data_path, model_save_path):
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # Load dataset
    print(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Extract EEG features
    spectral_columns = [col for col in df.columns if col.startswith('AB.')]
    coherence_columns = [col for col in df.columns if col.startswith('COH.')]
    
    # Group features by frequency bands
    delta_columns = [col for col in df.columns if '.A.delta.' in col]
    theta_columns = [col for col in df.columns if '.B.theta.' in col]
    alpha_columns = [col for col in df.columns if '.C.alpha.' in col]
    beta_columns = [col for col in df.columns if '.D.beta.' in col]
    highbeta_columns = [col for col in df.columns if '.E.highbeta.' in col]
    gamma_columns = [col for col in df.columns if '.F.gamma.' in col]
    
    print(f"Found {len(spectral_columns)} spectral features and {len(coherence_columns)} coherence features")
    print(f"Frequency bands: Delta: {len(delta_columns)}, Theta: {len(theta_columns)}, Alpha: {len(alpha_columns)}")
    print(f"Beta: {len(beta_columns)}, High Beta: {len(highbeta_columns)}, Gamma: {len(gamma_columns)}")
    
    # Combine all EEG features
    eeg_columns = spectral_columns + coherence_columns
    
    if not eeg_columns:
        X = df[['time_len']].values
        print("Using time_len as feature (no EEG bands found)")
    else:
        X = df[eeg_columns].values
        print(f"Using {len(eeg_columns)} EEG features")
    
    # Extract disorder labels
    if 'main.disorder' in df.columns:
        y = df['main.disorder'].values
        print("Using main.disorder as target")
    elif 'specific.disorder' in df.columns:
        y = df['specific.disorder'].values
        print("Using specific.disorder as target")
    else:
        y = df['filename'].apply(lambda x: x.split('_')[2][:8]).values
        print("Using filename pattern as target")
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Feature shape: {X.shape}")
    print(f"Classes: {le.classes_}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)
    
    # Initialize model
    model = EEGCognitiveClassifier(
        input_features=X.shape[1],  # Use the actual feature dimension
        num_classes=len(np.unique(y))
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Training loop
    num_epochs = 100
    batch_size = 32
    best_loss = float('inf')
    
    # Lists to store metrics
    train_losses = []
    val_losses = []
    accuracies = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        # Batch training
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test)
            val_loss = criterion(val_outputs, y_test)
            predictions = torch.argmax(val_outputs, dim=1)
            accuracy = (predictions == y_test).float().mean()
            
            # Store metrics
            train_losses.append(total_loss/len(range(0, len(X_train), batch_size)))
            val_losses.append(val_loss.item())
            accuracies.append(accuracy.item())
            
            print(f'Epoch {epoch+1}/{num_epochs}')
            print(f'Training Loss: {train_losses[-1]:.4f}')
            print(f'Validation Loss: {val_losses[-1]:.4f}, Accuracy: {accuracy:.4f}')
            
            scheduler.step(val_loss)
            
            if val_loss < best_loss:
                best_loss = val_loss
                # Save complete model state
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'label_encoder': le,
                    'accuracy': accuracy,
                    'feature_names': eeg_columns,
                    'scaler': scaler,
                    'spectral_columns': spectral_columns,
                    'coherence_columns': coherence_columns,
                    'frequency_bands': {
                        'delta': delta_columns,
                        'theta': theta_columns,
                        'alpha': alpha_columns,
                        'beta': beta_columns,
                        'highbeta': highbeta_columns,
                        'gamma': gamma_columns
                    },
                    'training_history': {
                        'train_losses': train_losses,
                        'val_losses': val_losses,
                        'accuracies': accuracies
                    }
                }, model_save_path)
                print(f"Model saved with accuracy: {accuracy:.4f}")
    
    # Load best model
    checkpoint = torch.load(model_save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Generate final evaluation metrics
    model.eval()
    with torch.no_grad():
        final_outputs = model(X_test)
        final_predictions = torch.argmax(final_outputs, dim=1)
        
        # Plot training metrics
        plot_training_metrics(train_losses, val_losses, accuracies, 
                            os.path.dirname(model_save_path))
        
        # Plot confusion matrix
        plot_confusion_matrix(y_test.numpy(), final_predictions.numpy(),
                            le.classes_, os.path.dirname(model_save_path))
        
        # Generate classification report
        report = classification_report(y_test.numpy(), final_predictions.numpy(),
                                    target_names=le.classes_, digits=4)
        
        # Save classification report
        with open(os.path.join(os.path.dirname(model_save_path), 
                              'classification_report.txt'), 'w') as f:
            f.write(report)
        
        print("\nFinal Classification Report:")
        print(report)
    
    return model, le

if __name__ == "__main__":
    # Example usage
    data_path = "d:/NeuroGPT/inputs/EEG.machinelearing_data_BRMH.csv"
    model_save_path = "d:/NeuroGPT/models/cognitive_model.pt"
    model, label_encoder = train_model(data_path, model_save_path)