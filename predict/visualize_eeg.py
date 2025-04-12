import numpy as np
import matplotlib.pyplot as plt
from mne.viz import plot_topomap
from mne import create_info, pick_types
from mne.channels import make_standard_montage
import mne
import seaborn as sns

def create_eeg_visualization(eeg_data, prediction, band_values):
    # Set up the figure
    fig = plt.figure(figsize=(20, 15))
    
    # Create channel positions
    montage = make_standard_montage('standard_1020')
    ch_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 
                'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
    
    # Create MNE info object
    info = create_info(ch_names=ch_names, sfreq=256., ch_types='eeg')
    info.set_montage(montage)
    
    # Get positions for plotting
    pos = np.array([montage.get_positions()['ch_pos'][ch] for ch in ch_names])[:, :2]
    
    # Plot each frequency band
    bands = ['delta', 'theta', 'alpha', 'beta', 'highbeta', 'gamma']
    for idx, band in enumerate(bands, 1):
        ax = fig.add_subplot(3, 3, idx)
        data = np.array([band_values[band][i] for i in range(len(ch_names))])
        
        # Create interpolated topographic map
        plot_topomap(data, pos, axes=ax, show=False, 
                    cmap='RdBu_r', sensors=True, 
                    outlines='head', contours=6)
        
        ax.set_title(f'{band.capitalize()} Band\n({min(data):.1f}-{max(data):.1f} μV²)')
    
    # Add brain activity heatmap
    ax_heat = fig.add_subplot(3, 3, 7)
    total_activity = np.zeros(len(ch_names))
    for band in bands:
        total_activity += np.array([band_values[band][i] for i in range(len(ch_names))])
    
    plot_topomap(total_activity, pos, axes=ax_heat, show=False,
                 cmap='hot', sensors=True, 
                 outlines='head', contours=6)
    ax_heat.set_title('Total Brain Activity')
    
    # Add power spectrum
    ax_spectrum = fig.add_subplot(3, 3, 8)
    mean_powers = [np.mean([band_values[band][i] for i in range(len(ch_names))]) 
                  for band in bands]
    sns.barplot(x=bands, y=mean_powers, ax=ax_spectrum)
    ax_spectrum.set_title('Average Band Powers')
    ax_spectrum.set_xticklabels(bands, rotation=45)
    ax_spectrum.set_ylabel('Power (μV²)')
    
    # Add prediction info
    ax_text = fig.add_subplot(3, 3, 9)
    ax_text.text(0.5, 0.5, f'Predicted Condition:\n{prediction}', 
                ha='center', va='center', fontsize=12)
    ax_text.axis('off')
    
    plt.tight_layout()
    return fig

def save_visualizations(output_path):
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close('all')