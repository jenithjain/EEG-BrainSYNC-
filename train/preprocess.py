import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA

def preprocess_data(data_path):
    df = pd.read_csv(data_path)
    
    # EEG features
    eeg_features = [col for col in df.columns if col.startswith(('AB.', 'COH.'))]
    X_eeg = df[eeg_features].values
    
    # Demographic features
    demo_features = ['age', 'education', 'iq', 'sex']
    X_demo = df[demo_features].copy()
    
    # Handle missing values
    X_demo = X_demo.fillna(X_demo.mean())
    X_eeg = np.nan_to_num(X_eeg, nan=0)
    
    # Encode sex
    X_demo['sex'] = (X_demo['sex'] == 'M').astype(int)
    
    # Feature selection for EEG
    selector = SelectKBest(score_func=mutual_info_classif, k=500)
    X_eeg_selected = selector.fit_transform(X_eeg, df['main.disorder'])
    
    # PCA for dimensionality reduction
    pca = PCA(n_components=0.95)  # Keep 95% variance
    X_eeg_pca = pca.fit_transform(X_eeg_selected)
    
    # Scale features
    scaler_eeg = StandardScaler()
    scaler_demo = StandardScaler()
    X_eeg_scaled = scaler_eeg.fit_transform(X_eeg_pca)
    X_demo_scaled = scaler_demo.fit_transform(X_demo)
    
    # Prepare labels
    le = LabelEncoder()
    y = le.fit_transform(df['main.disorder'])
    
    return {
        'X_eeg': X_eeg_scaled,
        'X_demo': X_demo_scaled,
        'y': y,
        'scaler_eeg': scaler_eeg,
        'scaler_demo': scaler_demo,
        'selector': selector,
        'pca': pca,
        'label_encoder': le,
        'eeg_shape': X_eeg_scaled.shape[1],
        'demo_shape': X_demo_scaled.shape[1]
    }