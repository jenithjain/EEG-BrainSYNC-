import pandas as pd

def print_data_structure():
    df = pd.read_csv("d:/NeuroGPT/inputs/EEG.machinelearing_data_BRMH.csv")
    
    print("\nData Structure:")
    print("-" * 50)
    print(f"Total columns: {len(df.columns)}")
    
    # Group columns by type
    ab_cols = [col for col in df.columns if col.startswith('AB.')]
    coh_cols = [col for col in df.columns if col.startswith('COH.')]
    
    print(f"\nAB (Absolute Power) columns: {len(ab_cols)}")
    print(f"COH (Coherence) columns: {len(coh_cols)}")
    
    # Print first few column names of each type
    print("\nSample AB columns:")
    for col in ab_cols[:5]:
        print(f"- {col}")
    
    print("\nSample COH columns:")
    for col in coh_cols[:5]:
        print(f"- {col}")
    
    print("\nUnique disorders:")
    print(df['main.disorder'].unique())

if __name__ == "__main__":
    print_data_structure()