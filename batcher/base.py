class EEGDataset(Dataset):
    def __init__(self, files, sample_keys, chunk_len, num_chunks, ovlp, 
                 root_path, gpt_only=False, normalization=True,
                 extract_frequency_features=False):
        super().__init__()
        self.files = files
        self.sample_keys = sample_keys
        self.chunk_len = chunk_len
        self.num_chunks = num_chunks
        self.ovlp = ovlp
        self.root_path = root_path
        self.gpt_only = gpt_only
        self.normalization = normalization
        self.extract_frequency_features = extract_frequency_features
        
    def __getitem__(self, idx):
        file = self.files[idx]
        data = np.load(os.path.join(self.root_path, file))
        
        # Extract EEG features
        eeg_data = data['eeg']
        
        if self.extract_frequency_features:
            # Extract frequency bands using scipy
            from scipy import signal
            
            fs = 250  # Sampling frequency
            freqs = {
                'Delta': (0.5, 4),
                'Theta': (4, 8),
                'Alpha': (8, 13),
                'Beta': (13, 30),
                'Gamma': (30, 100)
            }
            
            frequency_features = {}
            for band, (low, high) in freqs.items():
                # Apply bandpass filter
                b, a = signal.butter(4, [low/(fs/2), high/(fs/2)], btype='band')
                filtered = signal.filtfilt(b, a, eeg_data)
                # Get band power
                frequency_features[band] = np.mean(filtered**2, axis=0)
        
        # Prepare sample
        sample = {
            'inputs': torch.FloatTensor(eeg_data),
            'attention_mask': torch.ones(eeg_data.shape[0]),
        }
        
        if self.extract_frequency_features:
            sample['frequency_bands'] = torch.FloatTensor(
                np.array([frequency_features[band] for band in freqs.keys()])
            )
            
        return sample

    def __len__(self):
        return len(self.files)