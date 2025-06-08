from scipy.signal import coherence

def compute_coherence(ch1, ch2, sf, band):
    f, Cxy = coherence(ch1, ch2, sf, nperseg=256)
    idx_band = (f >= band[0]) & (f <= band[1])
    return np.mean(Cxy[idx_band]) if np.any(idx_band) else 0.0
