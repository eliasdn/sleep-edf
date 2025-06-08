import mne
import numpy as np
import cupy as cp
import math
from cupyx.scipy.fft import rfft as cp_rfft
from cupyx.scipy.fft import rfftfreq as cp_rfftfreq
from cupyx.scipy.signal.windows import hann as cp_hann
from cuml.cluster import KMeans as cuKMeans
from scipy.signal import welch
from tqdm import tqdm
import psutil

# Fonctions optimisées pour le GPU
def gpu_welch(x, fs=100.0, nperseg=256, noverlap=None):
    """Version GPU de la fonction welch avec CuPy."""
    if noverlap is None:
        noverlap = nperseg // 2
    
    window = cp_hann(nperseg, dtype=cp.float32)
    window_power = cp.sum(window ** 2)
    
    step = nperseg - noverlap
    n_segments = (len(x) - noverlap) // step
    
    psd = cp.zeros(nperseg // 2 + 1, dtype=cp.float32)
    
    for i in range(n_segments):
        start = i * step
        segment = x[start:start + nperseg] * window
        fft_seg = cp.abs(cp_rfft(segment, n=nperseg)) ** 2
        psd += fft_seg
    
    if n_segments > 0:
        psd /= (fs * window_power * n_segments)
    
    freqs = cp_rfftfreq(nperseg, 1.0/fs)
    return freqs, psd

def gpu_hjorth_parameters(x):
    """Version GPU des paramètres de Hjorth."""
    if len(x) < 2:
        return 0.0, 0.0, 0.0
    
    var0 = cp.var(x)
    dx = x[1:] - x[:-1]
    var1 = cp.var(dx)
    
    if var0 > 0:
        mobility = cp.sqrt(var1 / var0)
    else:
        return 0.0, 0.0, 0.0
    
    if len(dx) > 1:
        ddx = dx[1:] - dx[:-1]
        var2 = cp.var(ddx)
        if var1 > 0:
            complexity = cp.sqrt(var2 / var1) / (mobility + 1e-10)
        else:
            complexity = 0.0
    else:
        complexity = 0.0
    
    return var0, mobility, complexity
import umap
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from itertools import permutations
from scipy.stats import rankdata, entropy # Import entropy here
import math
import concurrent.futures
from functools import partial
import numba
import psutil
from tqdm import tqdm
from scipy.signal import welch, lfilter, get_window
from scipy.spatial.distance import pdist, squareform
from scipy.fft import rfft, rfftfreq
import nolds  # Pour le calcul d'entropie d'échantillon

# Configuration globale pour Numba
USE_NUMBA = True  # Activer/désactiver Numba selon les besoins

@numba.jit(nopython=True, fastmath=True, cache=True)
def _numba_welch(x, fs=1.0, window='hann', nperseg=256, noverlap=None):
    """Version optimisée de welch avec Numba"""
    if noverlap is None:
        noverlap = nperseg // 2
    
    # Fenêtrage
    if window == 'hann':
        win = 0.5 * (1 - np.cos(2 * np.pi * np.arange(nperseg) / (nperseg - 1)))
    else:
        win = np.ones(nperseg)
    
    # Nombre de segments
    n = len(x)
    step = nperseg - noverlap
    n_segments = (n - noverlap) // step
    
    # Calcul PSD
    psd = np.zeros(nperseg // 2 + 1)
    for i in range(n_segments):
        start = i * step
        segment = x[start:start + nperseg] * win
        spec = np.abs(np.fft.rfft(segment)) ** 2
        psd += spec
    
    psd /= (fs * np.sum(win ** 2) * n_segments)
    freqs = np.fft.rfftfreq(nperseg, 1/fs)
    return freqs, psd

@numba.jit(nopython=True, fastmath=True, cache=True)
def _hjorth_parameters_numba(x):
    """Version optimisée des paramètres de Hjorth avec Numba"""
    if len(x) < 2:
        return 0.0, 0.0, 0.0
    
    # Activité (variance)
    var0 = np.var(x)
    
    # Dérivée première
    dx = np.diff(x)
    var1 = np.var(dx)
    
    # Mobilité
    if var0 > 0:
        mobility = np.sqrt(var1 / var0)
    else:
        return 0.0, 0.0, 0.0
    
    # Complexité
    if len(dx) > 1:
        ddx = np.diff(dx)
        var2 = np.var(ddx)
        if var1 > 0:
            complexity = np.sqrt(var2 / var1) / mobility if var2 > 0 else 0.0
        else:
            complexity = 0.0
    else:
        complexity = 0.0
    
    return var0, mobility, complexity
from scipy.stats import entropy

# Fonctions utilitaires pour les nouvelles features
def _embed(x, order=3, delay=1):
    """Crée une matrice de séquences décalées"""
    n = len(x) - (order - 1) * delay
    if n <= 0:
        return np.array([])
    return np.asarray([x[i:i + order * delay:delay] for i in range(n)])

def _maxdist(x, y):
    """Distance de Chebyshev entre deux vecteurs"""
    return np.max(np.abs(x - y))

def _phi(x, m, r):
    """Fonction utilitaire pour l'entropie approximative"""
    n = len(x)
    x = np.asarray(x, dtype=np.float64)
    
    # Créer la matrice des séquences
    xm = _embed(x, m)
    if xm.size == 0:
        return 0.0
        
    # Calculer les distances
    d = pdist(xm, metric='chebyshev')
    d = squareform(d)
    
    # Compter les paires à distance < r
    c = np.sum(d <= r, axis=0) - 1.0  # -1 pour exclure la diagonale
    c[c == 0] = 1e-10  # Éviter log(0)
    
    # Moyenne des logs
    return np.mean(np.log(c / (n - m + 1.0)))

def approximate_entropy(x, m=2, r=None):
    """Calcule l'entropie approximative (ApEn) d'un signal.
    
    Paramètres:
    x : array_like
        Le signal d'entrée
    m : int, optionnel
        Longueur des motifs (embedding dimension)
    r : float, optionnel
        Seuil de similarité (fraction de l'écart-type du signal)
        Si None, r = 0.2 * std(x)
        
    Retourne:
    float
        Valeur de l'entropie approximative
    """
    x = np.asarray(x, dtype=np.float64)
    if r is None:
        r = 0.2 * np.std(x, ddof=0)
    
    if np.isclose(r, 0):
        return 0.0
        
    phi_m = _phi(x, m, r)
    phi_m1 = _phi(x, m + 1, r)
    
    return phi_m - phi_m1

def sample_entropy(x, m=2, r=None):
    """Calcule l'entropie d'échantillon (SampEn) d'un signal.
    
    Paramètres:
    x : array_like
        Le signal d'entrée
    m : int, optionnel
        Longueur des motifs (embedding dimension)
    r : float, optionnel
        Seuil de similarité (fraction de l'écart-type du signal)
        Si None, r = 0.2 * std(x)
        
    Retourne:
    float
        Valeur de l'entropie d'échantillon
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    
    if n <= m + 1:
        return 0.0
        
    if r is None:
        r = 0.2 * np.std(x, ddof=0)
    
    if np.isclose(r, 0):
        return 0.0
    
    # Créer les matrices de séquences
    xm = _embed(x, m)
    xm1 = _embed(x, m + 1)
    
    if xm.size == 0 or xm1.size == 0:
        return 0.0
    
    # Calculer les distances
    d_m = pdist(xm, metric='chebyshev')
    d_m1 = pdist(xm1, metric='chebyshev')
    
    # Compter les paires similaires
    A = np.sum(d_m1 <= r) / 2.0  # /2 car squareform est symétrique
    B = np.sum(d_m <= r) / 2.0
    
    # Éviter la division par zéro
    if B == 0 or A == 0:
        return 0.0
    
    return -np.log(A / B)

@numba.jit(nopython=True, fastmath=True, cache=True)
def _embed_sequences(x, order, delay):
    """Crée les séquences décalées pour le calcul de l'entropie de permutation."""
    n = len(x) - (order - 1) * delay
    if n <= 0:
        return np.zeros((0, order))
    
    result = np.zeros((n, order))
    for i in range(order):
        result[:, i] = x[i*delay:i*delay+n]
    return result

@numba.jit(nopython=True, fastmath=True, cache=True)
def _argsort_sequence(seq):
    """Version Numba-compatible de np.argsort pour les séquences."""
    return np.argsort(seq)

@numba.jit(nopython=True, fastmath=True, cache=True)
def _count_ordinal_patterns(embedded):
    """Compte les motifs ordinaux pour le calcul de l'entropie de permutation."""
    n = len(embedded)
    if n == 0:
        return np.zeros(1, dtype=np.float64)
    
    # Taille maximale possible pour le tableau de motifs uniques
    max_patterns = 1000  # Suffisamment grand pour la plupart des cas
    unique_patterns = []
    pattern_counts = np.zeros(max_patterns, dtype=np.float64)
    
    for i in range(n):
        # Obtenir le motif de tri pour cette séquence
        current_pattern = _argsort_sequence(embedded[i])
        found = False
        
        # Vérifier si ce motif existe déjà
        for j, pattern in enumerate(unique_patterns):
            if np.all(current_pattern == pattern):
                pattern_counts[j] += 1.0
                found = True
                break
                
        # Si nouveau motif, l'ajouter
        if not found and len(unique_patterns) < max_patterns - 1:
            unique_patterns.append(current_pattern)
            pattern_counts[len(unique_patterns)-1] = 1.0
    
    # Retourner uniquement les comptes non nuls
    if len(unique_patterns) == 0:
        return np.zeros(1, dtype=np.float64)
        
    return pattern_counts[:len(unique_patterns)] / np.sum(pattern_counts[:len(unique_patterns)])

def extract_features(data, sf, bands, nperseg=256, use_gpu=True, batch_size=100):
    """Extrait les caractéristiques du signal EEG pour chaque époque en utilisant le GPU.
    
    Paramètres:
    data : array-like, shape (n_epochs, n_channels, n_times)
        Données EEG organisées en époques
    sf : float
        Fréquence d'échantillonnage
    bands : list of tuples
        Liste des bandes de fréquence à analyser (ex: [(0.5, 4), (4, 8), ...])
    nperseg : int
        Nombre de points par segment pour le calcul PSD
    use_gpu : bool
        Si True, utilise le GPU pour le calcul
    batch_size : int
        Taille des lots pour le traitement GPU
        
    Retourne:
    numpy.ndarray
        Tableau de caractéristiques de forme (n_epochs, n_features)
    """
    if not use_gpu:
        return _extract_features_cpu(data, sf, bands, nperseg)
        
    n_epochs = len(data)
    if n_epochs == 0:
        return np.array([])
        
    n_channels = data.shape[1]
    n_times = data.shape[2]
    n_bands = len(bands)
    n_features = 5 + n_bands  # 5 features de base + bandes de fréquence
    
    # Préparer le tableau de sortie
    X = np.zeros((n_epochs, n_channels * n_features), dtype=np.float32)
    
    # Traiter par lots pour économiser la mémoire GPU
    for i in tqdm(range(0, n_epochs, batch_size), desc="Traitement GPU"):
        batch_end = min(i + batch_size, n_epochs)
        batch_data = data[i:batch_end]
        
        # Convertir le lot en tenseur CuPy
        batch_gpu = cp.array(batch_data, dtype=cp.float32)
        batch_features = []
        
        # Traiter chaque canal
        for ch in range(n_channels):
            channel_data = batch_gpu[:, ch, :]
            
            # Calculer les paramètres de Hjorth sur GPU
            var0 = cp.var(channel_data, axis=1, keepdims=True)
            dx = channel_data[:, 1:] - channel_data[:, :-1]
            var1 = cp.var(dx, axis=1, keepdims=True)
            mobility = cp.sqrt(var1 / (var0 + 1e-10))
            
            ddx = dx[:, 1:] - dx[:, :-1]
            var2 = cp.var(ddx, axis=1, keepdims=True)
            complexity = cp.sqrt(var2 / (var1 + 1e-10)) / (mobility + 1e-10)
            
            # Calculer l'entropie de permutation (sur CPU pour l'instant)
            perm_entropy = np.zeros(len(batch_data))
            for j in range(len(batch_data)):
                try:
                    perm_entropy[j] = permutation_entropy(
                        cp.asnumpy(channel_data[j]), 
                        order=3, 
                        delay=1, 
                        normalize=True
                    )
                except:
                    perm_entropy[j] = 0.0
            
            # Calculer la PSD sur GPU
            psd_features = []
            for j in range(len(batch_data)):
                try:
                    freqs, psd = gpu_welch(channel_data[j], fs=sf, nperseg=min(nperseg, n_times))
                    
                    # Calculer les bandes de fréquence
                    band_powers = []
                    for b in range(len(bands)):
                        mask = (freqs >= bands[b][0]) & (freqs <= bands[b][1])
                        if cp.any(mask):
                            band_powers.append(cp.mean(psd[mask]))
                        else:
                            band_powers.append(0.0)
                    
                    # Concaténer toutes les features
                    features = cp.concatenate([
                        var0[j],  # Activité
                        mobility[j],  # Mobilité
                        complexity[j],  # Complexité
                        cp.array([perm_entropy[j]]),  # Entropie de permutation
                        cp.array([0.0]),  # Autre feature (peut être utilisée pour d'autres mesures)
                        cp.array(band_powers)  # Puissances des bandes
                    ])
                    psd_features.append(features)
                except Exception as e:
                    print(f"Erreur PSD GPU: {str(e)}")
                    psd_features.append(cp.zeros(n_features, dtype=cp.float32))
            
            batch_features.append(cp.stack(psd_features))
        
        # Concaténer les canaux et stocker les résultats
        if batch_features:
            batch_result = cp.concatenate(batch_features, axis=1)
            X[i:batch_end] = cp.asnumpy(batch_result)
    
    return X

def permutation_entropy(x, order=3, delay=1, normalize=True):
    """Calcule l'entropie de permutation d'un signal de manière optimisée.
    
    Paramètres:
    x : array_like
        Le signal d'entrée
    order : int, optionnel
        Ordre des motifs (embedding dimension), typiquement entre 3 et 7
    delay : int, optionnel
        Délai entre les échantillons
    normalize : bool, optionnel
        Si True, normalise le résultat entre 0 et 1
        
    Retourne:
    float
        Valeur de l'entropie de permutation
    """
    try:
        x = np.asarray(x, dtype=np.float64)
        if len(x) < order + 1:
            return 0.0
            
        # Créer les séquences décalées
        embedded = _embed_sequences(x, order, delay)
        if len(embedded) == 0:
            return 0.0
            
        # Compter les motifs de permutation
        probs = _count_ordinal_patterns(embedded)
        
        # Éviter log(0)
        probs = probs[probs > 0]
        if len(probs) == 0:
            return 0.0
            
        # Calculer l'entropie
        pe = -np.sum(probs * np.log(probs))
        
        # Normaliser si demandé
        if normalize:
            pe = pe / np.log(math.factorial(order))
            
        return float(pe)
        
    except Exception as e:
        print(f"Erreur dans permutation_entropy: {str(e)}")
        return 0.0

def _embed_sequences(x, order, delay):
    """Crée les séquences décalées pour le calcul de l'entropie de permutation."""
    n = len(x)
    if n < order + (order - 1) * delay:
        return np.array([])
    
    # Créer les indices pour chaque dimension
    indices = np.zeros((order, n - (order - 1) * delay), dtype=int)
    for i in range(order):
        indices[i] = np.arange(i * delay, i * delay + len(indices[0]))
    
    # Extraire les séquences
    return x[indices]

def _count_ordinal_patterns(embedded):
    """Compte les motifs ordinaux uniques dans les séquences."""
    if len(embedded) == 0 or len(embedded[0]) == 0:
        return np.array([])
    
    # Trier chaque séquence et garder les indices de tri
    sorted_idx = np.argsort(embedded, axis=0)
    
    # Compter les motifs uniques
    unique_patterns, counts = np.unique(sorted_idx.T, axis=0, return_counts=True)
    
    # Normaliser les comptes en probabilités
    return counts / np.sum(counts)

def _extract_features_cpu(data, sf, bands, nperseg=256, num_workers=None):
    """Version CPU de l'extraction de caractéristiques (fallback)"""
    if num_workers is None or num_workers < 1:
        num_workers = max(1, psutil.cpu_count(logical=False) - 1)
        
    n_epochs = len(data)
    if n_epochs == 0:
        return np.array([])
        
    n_channels = data.shape[1]
    n_bands = len(bands)
    n_features = 5 + n_bands  # 5 features de base + bandes de fréquence
    
    # Préparer le tableau de sortie
    X = np.zeros((n_epochs, n_channels * n_features), dtype=np.float32)
    
    # Fonction pour traiter un seul canal
    def process_channel(args):
        i, ch = args
        channel_data = data[i, ch, :]
        features = np.zeros(n_features, dtype=np.float32)
        
        try:
            # Calculer les paramètres de Hjorth
            var0 = np.var(channel_data)
            dx = np.diff(channel_data)
            var1 = np.var(dx)
            
            if var0 > 1e-10:  # Éviter la division par zéro
                mobility = np.sqrt(var1 / var0)
            else:
                mobility = 0.0
                
            if len(dx) > 1:
                ddx = np.diff(dx)
                var2 = np.var(ddx)
                if var1 > 1e-10 and mobility > 1e-10:
                    complexity = np.sqrt(var2 / var1) / mobility
                else:
                    complexity = 0.0
            else:
                complexity = 0.0
            
            # Calculer l'entropie de permutation
            try:
                pe = permutation_entropy(channel_data, order=3, delay=1, normalize=True)
            except:
                pe = 0.0
            
            # Calculer la PSD
            try:
                freqs, psd = welch(channel_data, fs=sf, nperseg=min(nperseg, len(channel_data)))
                
                # Calculer les bandes de fréquence
                band_powers = np.zeros(n_bands, dtype=np.float32)
                for b, (fmin, fmax) in enumerate(bands):
                    mask = (freqs >= fmin) & (freqs <= fmax)
                    if np.any(mask):
                        band_powers[b] = np.mean(psd[mask])
                
                # Rassembler toutes les features
                features[:5] = [var0, mobility, complexity, pe, 0.0]  # Dernier 0.0 pour une feature supplémentaire
                features[5:5+n_bands] = band_powers
                
            except Exception as e:
                print(f"Erreur PSD CPU: {str(e)}")
                
        except Exception as e:
            print(f"Erreur dans le traitement du canal: {str(e)}")
            
        return i, ch, features
    
    # Traitement parallèle
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Préparer les arguments
        args = [(i, ch) for i in range(n_epochs) for ch in range(n_channels)]
        
        # Exécuter en parallèle
        for i, ch, ch_features in tqdm(
            executor.map(process_channel, args), 
            total=len(args),
            desc="Extraction des caractéristiques"
        ):
            start_idx = ch * n_features
            end_idx = start_idx + n_features
            X[i, start_idx:end_idx] = ch_features
    
    return X

def hjorth_parameters(x):
    """Calcule les paramètres de Hjorth (activité, mobilité, complexité).
    
    Paramètres:
    x : array_like
        Le signal d'entrée
        
    Retourne:
    tuple
        (activité, mobilité, complexité)
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Dérivées première et seconde
    dx = np.diff(x)
    ddx = np.diff(dx)
    
    # Variance (activité)
    var0 = np.var(x, ddof=1)
    
    # Mobilité
    if var0 == 0:
        return 0.0, 0.0, 0.0
    
    var1 = np.var(dx, ddof=1)
    mobility = np.sqrt(var1 / var0)
    
    # Complexité
    if len(ddx) > 0 and var1 > 0:
        var2 = np.var(ddx, ddof=1)
        complexity = np.sqrt(var2 / var1) / mobility if var2 > 0 else 0.0
    else:
        complexity = 0.0
    
    return var0, mobility, complexity

# Charger le fichier EEG
# Assurez-vous que 'mon_fichier.edf' est dans le même répertoire que votre notebook
# ou fournissez le chemin complet vers le fichier.
# Vous pouvez télécharger un fichier d'exemple depuis le dataset mentionné
# dans les commentaires markdown au début de votre notebook si n'en avez pas.
try:
    raw = mne.io.read_raw_edf('SC4001E0-PSG.edf', preload=True)
    raw.filter(1., 40.)  # filtrage rapide

    # Extraire les features par fenêtre
    # Use the preload=True argument for make_fixed_length_epochs to match the first cell
    epochs = mne.make_fixed_length_epochs(raw, duration=5.0, preload=True)
    data = epochs.get_data()  # (n_epochs, n_channels, n_times)

    @numba.jit(nopython=True, cache=True)
    def bandpower_numba(data, sf, fmin, fmax):
        """Version optimisée avec Numba pour le calcul des bandes de fréquence"""
        # The import statement has been moved outside this function.
        # from scipy.signal import welch # REMOVE THIS IMPORT
        # Cannot use scipy.signal.welch directly in nopython mode
        # You would need a Numba-compatible implementation of welch or compute PSD differently
        # For now, remove the nopython=True decorator or use a different approach if welch is critical.
        # Let's remove nopython=True for now to fix the immediate import error.
        # If performance is critical, a Numba implementation of PSD calculation is needed.

        # Reverting to a standard Python function for now, as welch is not Numba-compatible in nopython mode.
        # If you still need Numba performance, you'll need to implement PSD calculation manually or find a Numba-compatible alternative.
        # Removing @numba.jit(nopython=True, cache=True) temporarily
        pass # Placeholder, the function definition will be outside the numba block

    # Version optimisée avec Numba pour le calcul des bandes de fréquence (without nopython=True if welch is used)
    # Or, if welch is not strictly necessary, reimplement PSD calculation within Numba.
    # For the sake of fixing the import error with the current structure, let's just use the standard Python welch.
    # If Numba optimization is a must for this function, a re-implementation or alternative is required.
    # Assuming standard welch is acceptable for now in a non-nopython context.

    # @numba.jit(nopython=True, cache=True) # Remove or adjust Numba decorator
    def bandpower_numba(data, sf, fmin, fmax):
         # Use the globally imported welch
         # Note: This function will NOT be jitted with nopython=True due to welch.
         # If you require nopython mode, replace welch with a Numba-compatible PSD calculation.
         # As a temporary fix to allow execution, remove nopython=True.
         # If you want to keep Numba, you MUST remove the welch call and replace it.
         # Let's keep the original welch call for functionality and remove the Numba decorator for now.
         psd, freqs = welch(data, sf, nperseg=min(256, len(data)))
         idx_band = (freqs >= fmin) & (freqs <= fmax)
         return np.mean(psd[idx_band]) if np.any(idx_band) else 0.0


    # Version de compatibilité pour le code existant
    def bandpower(data, sf, band):
        # This function now calls the potentially optimized bandpower_numba
        # but bandpower_numba currently uses welch, which might not be jittable.
        # If bandpower_numba uses standard welch, this call is fine.
        # If you reimplement bandpower_numba in nopython mode, this call works too.
        return bandpower_numba(data, sf, band[0], band[1])

    # Calcul du ratio alpha/theta pour une fenêtre
    # Also move welch import outside
    def calculate_alpha_theta_ratio(epoch, sf):
        # from scipy.signal import welch # REMOVE THIS IMPORT
        # Calculer le PSD
        freqs, psd = welch(epoch, sf, nperseg=min(256, len(epoch)))

        # Définir les bandes de fréquence
        alpha_band = (8, 12)  # Alpha
        theta_band = (4, 8)    # Theta

        # Calculer la puissance moyenne dans chaque bande
        # Use np.logical_and for boolean indexing consistency
        alpha_power = np.mean(psd[np.logical_and(freqs >= alpha_band[0], freqs <= alpha_band[1])])
        theta_power = np.mean(psd[np.logical_and(freqs >= theta_band[0], freqs <= theta_band[1])])


        # Éviter la division par zéro
        if theta_power > 0:
            return alpha_power / theta_power
        return 0.0

    # Calcul de la puissance gamma (30-100 Hz)
    # Also move welch import outside
    def calculate_gamma_power(epoch, sf):
        # from scipy.signal import welch # REMOVE THIS IMPORT
        freqs, psd = welch(epoch, sf, nperseg=min(256, len(epoch)))
        gamma_band = (30, 100)
        # Use np.logical_and for boolean indexing consistency
        idx_gamma = np.logical_and(freqs >= gamma_band[0], freqs <= gamma_band[1])
        return np.mean(psd[idx_gamma]) if np.any(idx_gamma) else 0.0

    # Calcul de l'entropie spectrale (mesure de complexité)
    # Also move welch and entropy imports outside
    def calculate_spectral_entropy(epoch, sf):
        # from scipy.signal import welch # REMOVE THIS IMPORT
        # from scipy.stats import entropy # REMOVE THIS IMPORT - entropy is imported globally
        # Calculer le PSD normalisé (probabilité)
        freqs, psd = welch(epoch, sf, nperseg=min(256, len(epoch)))
        psd_norm = psd / np.sum(psd)
        # Calculer l'entropie (en évitant les log(0))
        # from scipy.stats import entropy # REMOVE THIS IMPORT
        return entropy(psd_norm + 1e-12) # Use globally imported entropy

    def calculate_permutation_entropy(epoch, m=3, delay=1):
        """
        Calcule l'entropie de permutation d'une série temporelle.

        Paramètres:
        -----------
        epoch : array-like
            La série temporelle d'entrée
        m : int, optional (défaut=3)
            Longueur des motifs (embedding dimension)
        delay : int, optional (défaut=1)
        Délai entre les échantillons

        Retourne:
        --------
        pe : float
            Entropie de permutation normalisée (entre 0 et 1)
        """
        n = len(epoch)
        if n < m * delay:
            return 0.0

        # Generate patterns using numpy stride tricks
        # Ensure window_shape and step are valid
        try:
            patterns = np.lib.stride_tricks.sliding_window_view(epoch, window_shape=m)[::delay]
        except ValueError:
            # Handle cases where window_shape or step is invalid for the input array
            print(f"Warning: Could not create sliding window view for epoch. len(epoch)={len(epoch)}, m={m}, delay={delay}")
            return 0.0


        if patterns.shape[0] == 0: # Handle case where patterns cannot be formed
             return 0.0
        # Sort patterns along the last axis to get the permutation
        sorted_patterns_indices = np.argsort(patterns, axis=-1) # Use axis=-1 for the last dimension


        # Convert sorted patterns to a format that can be used for counting unique rows
        # Create a unique integer representation for each permutation/sorted pattern
        # This is a common way to count unique rows in numpy
        # Ensure the dtype is large enough to hold the integer representation
        # Calculate required byte size based on the number of columns (m) and item size
        required_bytes = sorted_patterns_indices.shape[1] * sorted_patterns_indices.dtype.itemsize
        dtype = np.dtype((np.void, required_bytes))

        # Ensure the array is contiguous before viewing
        contiguous_sorted_patterns = np.ascontiguousarray(sorted_patterns_indices)

        # Reshape or view to ensure it's a 2D array of bytes for unique
        try:
            # View as dtype with multiple items per row, then flatten or reshape if necessary for unique
            # A direct view might create a 1D array of structured dtypes, which is fine for np.unique
             contiguous_sorted_patterns_view = contiguous_sorted_patterns.view(dtype)
        except ValueError as e:
             print(f"Error creating view for unique counting: {e}")
             return 0.0 # Return 0.0 if view creation fails

        # Count unique patterns
        try:
            _, counts = np.unique(contiguous_sorted_patterns_view, return_counts=True)
        except TypeError as e:
            print(f"Error counting unique patterns: {e}")
            # This can happen if the dtype view is not compatible with np.unique
            # As an alternative, convert to a list of tuples and then use unique
            patterns_as_tuples = [tuple(row) for row in sorted_patterns_indices]
            unique_patterns_tuples, counts = np.unique(patterns_as_tuples, return_counts=True, axis=0) # Use axis=0 for tuples/rows
            # Note: np.unique with axis=0 on an array of tuples is not standard, ensure it works or use a loop/set.
            # A more robust way might be to convert rows to hashable types if needed.
            # Sticking to the view approach as it's more performant if it works.
            # If the view approach consistently fails, reconsider the unique counting method.
            print("Falling back to list of tuples for unique counting.")
            unique_elements = []
            counts = []
            for pattern in patterns_as_tuples:
                if pattern in unique_elements:
                    counts[unique_elements.index(pattern)] += 1
                else:
                    unique_elements.append(pattern)
                    counts.append(1)
            counts = np.array(counts) # Convert counts back to numpy array

        # Normaliser les comptes pour obtenir des probabilités
        probs = counts / np.sum(counts)

        # Calculer l'entropie de permutation
        # Use globally imported entropy
        # from scipy.stats import entropy # REMOVE THIS IMPORT
        # Ensure probs is not empty before calculating entropy
        if len(probs) == 0 or np.sum(probs) == 0:
             return 0.0 # Return 0.0 if no patterns or probabilities are 0

        # Calculate entropy using natural log
        pe = -np.sum(probs * np.log(probs + 1e-12)) # Add small constant to avoid log(0)


        # Normaliser par le log factoriel de m pour obtenir une valeur entre 0 et 1
        # Ensure m > 1 for log factorial calculation (factorial(1)=1, log(1)=0)
        if m > 1:
             # Calculate log factorial robustly
             log_factorial_m = np.sum(np.log(np.arange(1, m + 1))) if m > 0 else 0.0
             if log_factorial_m > 0:
                 pe_normalized = pe / log_factorial_m
             else:
                 pe_normalized = 0.0 # Avoid division by zero if log_factorial_m is 0 (for m=0 or m=1)
        else:
            pe_normalized = 0.0 # For m=0 or m=1, normalized entropy is 0

        # Clamp the normalized entropy to be within [0, 1] due to potential floating point inaccuracies
        pe_normalized = np.clip(pe_normalized, 0.0, 1.0)

        return pe_normalized


    # Calcul des caractéristiques temporelles (variation entre fenêtres)
    def calculate_temporal_features(epoch):
        # Écart-type du signal (variabilité temporelle)
        std_dev = np.std(epoch)

        # Différence absolue moyenne (MAD) - variation entre échantillons consécutifs
        # Ensure epoch has at least two elements for np.diff
        if len(epoch) > 1:
            mad = np.mean(np.abs(np.diff(epoch)))
        else:
            mad = 0.0 # Return 0.0 if diff cannot be calculated

        # Pente moyenne du signal
        # Ensure epoch has at least two elements for np.diff
        if len(epoch) > 1:
            slope = np.mean(np.diff(epoch))
        else:
            slope = 0.0 # Return 0.0 if diff cannot be calculated

        # Entropie de permutation (complexité du signal)
        # Provide default values for m and delay
        perm_entropy = calculate_permutation_entropy(epoch, m=3, delay=1)

        return [std_dev, mad, slope, perm_entropy]

    def process_epoch(epoch, sf, bands):
        """Traite une seule époque et retourne ses features"""
        feat = []
        for ch in epoch:
            # Utiliser la version optimisée de bandpower (which is now standard Python welch based)
            for fmin, fmax in bands:
                feat.append(bandpower_numba(ch, sf, fmin, fmax)) # Calls the standard welch function

            # Calculer les autres features
            # Add checks for empty channels before calculating features
            if len(ch) > 0:
                feat.extend([
                    calculate_alpha_theta_ratio(ch, sf),
                    calculate_gamma_power(ch, sf),
                    calculate_spectral_entropy(ch, sf)
                ])
            else:
                 feat.extend([0.0, 0.0, 0.0]) # Append default values if channel is empty


            # Caractéristiques temporelles
            # Add check for empty channels before calculating temporal features
            if len(ch) > 0:
                feat.extend(calculate_temporal_features(ch))
            else:
                feat.extend([0.0, 0.0, 0.0, 0.0]) # Append default values if channel is empty
        return feat

    def process_batch(batch, sf, bands):
        """Traite un lot d'époques"""
        # Ensure batch is not empty before processing
        if len(batch) == 0:
            return []
        return [process_epoch(epoch, sf, bands) for epoch in batch]

    # Configuration du traitement parallèle
    sf = raw.info['sfreq']
    bands = np.array([(0.5,4), (4,8), (8,12), (12,30)], dtype=np.float32)

    # Déterminer le nombre optimal de workers
    num_cores = psutil.cpu_count(logical=False)
    num_workers = min(4, num_cores)  # Limiter à 4 workers maximum pour éviter la surcharge

    print(f"Traitement de {len(data)} époques avec {num_workers} workers...")
    
    # Optimisation : Pré-calculer les fréquences pour welch
    from scipy.signal import get_window
    nperseg = min(256, len(data[0][0]))
    freqs = np.fft.rfftfreq(nperseg, 1.0/sf)
    
    # Optimisation : Fonction de traitement d'un seul canal
    def process_channel(ch, sf, bands, freqs=freqs):
        try:
            # Vérifier que le canal n'est pas vide
            if len(ch) < 10:  # Taille minimale pour des calculs significatifs
                return [0.0] * (len(bands) + 7)  # Ajuster selon le nombre total de features
            
            # Initialiser la liste des caractéristiques
            channel_feat = []
            
            # Calculer les paramètres de Hjorth
            if USE_NUMBA and len(ch) >= 128:
                try:
                    activity, mobility, complexity = _hjorth_parameters_numba(ch)
                except:
                    activity, mobility, complexity = hjorth_parameters(ch)
            else:
                activity, mobility, complexity = hjorth_parameters(ch)
            
            # Calculer les bandes de fréquence
            band_powers = []
            for fmin, fmax in bands:
                band_powers.append(bandpower_numba(ch, sf, fmin, fmax))
            
            # Ajouter les features de base
            channel_feat.extend([
                activity,
                mobility,
                complexity,
                calculate_alpha_theta_ratio(ch, sf) if len(ch) > 0 else 0.0,
                calculate_gamma_power(ch, sf) if len(ch) > 0 else 0.0,
                calculate_spectral_entropy(ch, sf) if len(ch) > 0 else 0.0
            ])
            
            # Ajouter les puissances des bandes
            channel_feat.extend(band_powers)
            
            # Ajouter les caractéristiques temporelles
            channel_feat.extend(calculate_temporal_features(ch) if len(ch) > 0 else [0.0, 0.0, 0.0, 0.0])
            
            return channel_feat
            
        except Exception as e:
            print(f"Erreur dans le traitement du canal: {str(e)}")
            return [0.0] * (len(bands) + 7)  # Valeurs par défaut en cas d'erreur

def process_batch(batch, sf, bands):
    """Traite un lot d'époques"""
    # Ensure batch is not empty before processing
    if len(batch) == 0:
        return []
    return [process_epoch(epoch, sf, bands) for epoch in batch]

# Configuration du traitement parallèle
if 'raw' in locals() and hasattr(raw, 'info') and 'sfreq' in raw.info:
    sf = raw.info['sfreq']
    bands = np.array([(0.5,4), (4,8), (8,12), (12,30)], dtype=np.float32)

    # Déterminer le nombre optimal de workers
    num_cores = psutil.cpu_count(logical=False)
    num_workers = min(4, num_cores)  # Limiter à 4 workers maximum pour éviter la surcharge

    print(f"Traitement de {len(data)} époques avec {num_workers} workers...")
        
    # Optimisation : Pré-calculer les fréquences pour welch
    nperseg = min(256, len(data[0][0]))
    freqs = np.fft.rfftfreq(nperseg, 1.0/sf)
    
    # Optimisation : Fonction de traitement d'un seul canal
    def process_channel(ch, sf, bands, freqs=freqs):
    # Vérifier que le canal n'est pas vide
    if len(ch) < 10:  # Taille minimale pour des calculs significatifs
        return [0.0] * (len(bands) + 7)  # Ajuster selon le nombre total de features
        
    features = []
        
    # Calculer PSD avec Numba si activé
    if USE_NUMBA and len(ch) >= 128:  # Seuil minimal pour que Numba soit efficace
        try:
            _, psd = _numba_welch(ch, fs=sf, nperseg=min(256, len(ch)))
        except:
            window = get_window('hann', min(256, len(ch)))
            _, psd = welch(ch, sf, window=window, nperseg=min(256, len(ch)), return_onesided=True)
    else:
        window = get_window('hann', min(256, len(ch)))
        _, psd = welch(ch, sf, window=window, nperseg=min(256, len(ch)), return_onesided=True)
        
    # Calculer les paramètres de Hjorth avec Numba si activé
    if USE_NUMBA:
        activity, mobility, complexity = _hjorth_parameters_numba(ch)
    else:
        activity, mobility, complexity = hjorth_parameters(ch)
            
    features.extend([activity, mobility, complexity])
        
    # Calculer l'entropie de permutation avec nolds
    try:
        pe = permutation_entropy(ch, order=3)
        features.append(pe if not np.isnan(pe) and not np.isinf(pe) else 0.0)
    except Exception as e:
        print(f"Erreur calcul entropie permutation: {str(e)}")
        features.append(0.0)
            
    # Calculer l'entropie d'échantillon
    try:
        sampen = sample_entropy(ch, m=2, r=0.2*np.std(ch))
        features.append(sampen)
    except:
        features.append(0.0)
        
    # Bands de fréquence
    for fmin, fmax in bands:
        idx_band = (freqs >= fmin) & (freqs <= fmax)
        features.append(np.mean(psd[idx_band]) if np.any(idx_band) else 0.0)
        
    # Ratio alpha/theta
    alpha_band = (8, 12)
    theta_band = (4, 8)
    alpha_power = np.mean(psd[(freqs >= alpha_band[0]) & (freqs <= alpha_band[1])])
    theta_power = np.mean(psd[(freqs >= theta_band[0]) & (freqs <= theta_band[1])])
    features.append(alpha_power / theta_power if theta_power > 0 else 0.0)
        
    # Puissance gamma
    gamma_band = (30, 100)
    idx_gamma = (freqs >= gamma_band[0]) & (freqs <= gamma_band[1])
    features.append(np.mean(psd[idx_gamma]) if np.any(idx_gamma) else 0.0)
        
    # Entropie spectrale
    psd_norm = psd / (np.sum(psd) + 1e-12)
    features.append(entropy(psd_norm + 1e-12))
        
    # Caractéristiques temporelles
    std_dev = np.std(ch)
    mad = np.mean(np.abs(np.diff(ch))) if len(ch) > 1 else 0.0
    slope = np.mean(np.diff(ch)) if len(ch) > 1 else 0.0
        
    # Entropie de permutation (version simplifiée pour la vitesse)
    m, delay = 3, 1
    n = len(ch)
    if n >= m * delay:
        try:
            patterns = np.lib.stride_tricks.sliding_window_view(ch, window_shape=m)[::delay]
            if patterns.size > 0:
                sorted_patterns = np.argsort(patterns, axis=1)
                _, counts = np.unique(sorted_patterns, axis=0, return_counts=True)
                probs = counts / np.sum(counts)
                pe = -np.sum(probs * np.log(probs + 1e-12))
                features.extend([std_dev, mad, slope, pe / np.log(np.math.factorial(m))])
            else:
                features.extend([std_dev, mad, slope, 0.0])
        except:
            features.extend([std_dev, mad, slope, 0.0])
    else:
        features.extend([std_dev, mad, slope, 0.0])
            
    return features
    
# Traitement parallèle optimisé avec batch processing
X = []
if len(data) > 0:
    # Préparer les tâches par lots
    batch_size = 1000  # Taille de lot optimale pour équilibrer parallélisme et surcharge
    tasks = []
    batch = []
        
    # Créer des lots de tâches
    for i, epoch in enumerate(data):
        for j, ch in enumerate(epoch):
            batch.append((ch, sf, bands, freqs))
            if len(batch) >= batch_size:
                tasks.append(batch)
                batch = []
    if batch:  # Ajouter le dernier lot incomplet
        tasks.append(batch)
        
    # Exécuter en parallèle par lots
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        for batch in tqdm(tasks, desc="Traitement des lots"):
            try:
                # Traiter chaque lot en parallèle
                batch_results = list(executor.map(
                    lambda x: process_channel(*x),
                    batch,
                    chunksize=max(1, len(batch) // (num_workers * 2))
                ))
                results.extend(batch_results)
            except Exception as e:
                print(f"Erreur lors du traitement d'un lot: {str(e)}")
                # En cas d'erreur, traiter les éléments du lot séquentiellement
                for task in batch:
                    try:
                        results.append(process_channel(*task))
                    except:
                        results.append([0.0] * (len(bands) + 7))  # Valeurs par défaut en cas d'échec
            
            # Reconstruire la structure des époques
            n_channels = len(data[0]) if len(data) > 0 else 0
            if n_channels > 0:
                X = [results[i:i + n_channels] for i in range(0, len(results), n_channels)]
                X = np.array(X, dtype=np.float32).reshape(len(data), -1)
            else:
                X = np.array([], dtype=np.float32)

    # Conversion finale
    X = np.ascontiguousarray(X, dtype=np.float32)
    print(f"Traitement terminé. Données finales: {X.shape}")

    # Réduction de dimension
    # Vérifiez que X n'est pas vide avant d'appliquer UMAP
    if X.shape[0] > 0:
        # Adjust n_neighbors for UMAP if the number of samples is small
        n_neighbors_umap = min(15, X.shape[0]-1) if X.shape[0] > 1 else 1
        # Ensure n_neighbors is at least 1 if there's more than 1 sample
        n_neighbors_umap = max(1, n_neighbors_umap) if X.shape[0] > 1 else 1

        X_embedded = umap.UMAP(n_neighbors=n_neighbors_umap, min_dist=0.1).fit_transform(X)

        # Clustering avec 5 clusters pour les stades de sommeil (W, N1, N2, N3, REM)
        n_clusters = 5

        # Ensure n_clusters is not greater than the number of samples and is at least 1
        n_clusters_kmeans = min(n_clusters, X_embedded.shape[0])
        n_clusters_kmeans = max(1, n_clusters_kmeans) if X_embedded.shape[0] > 0 else 0 # Ensure at least 1 cluster if data exists, 0 if no data

        if n_clusters_kmeans > 0 and n_clusters_kmeans < n_clusters:
            print(f"Attention : Le nombre d'époques est inférieur à {n_clusters}. Réduction du nombre de clusters KMeans à {n_clusters_kmeans}.")


        # Only train KMeans if there are enough samples for the chosen number of clusters (> 0)
        if X_embedded.shape[0] >= n_clusters_kmeans and n_clusters_kmeans > 0:
             # Convertir les données en format GPU (CuPy array)
            X_embedded_gpu = cp.asarray(X_embedded.astype(np.float32))  # float32 pour de meilleures performances

            # Entraîner K-means sur GPU
            kmeans = cuKMeans(n_clusters=n_clusters_kmeans, random_state=42, output_type='numpy')
            kmeans.fit(X_embedded_gpu)
            labels_pred = kmeans.labels_

            # Libérer la mémoire GPU
            del X_embedded_gpu
            cp.get_default_memory_pool().free_all_blocks()

            # Charger et parser l'hypnogramme
            try:
                annotations = mne.read_annotations('SC4001EC-Hypnogram.edf')
                # raw.set_annotations(annotations) # No need to set annotations on raw here

                # Associer chaque epoch à son label d'annotation
                labels_true_str = [] # Store original string labels
                # Ensure the number of epochs matches the number of predictions
                if len(epochs.events) == len(labels_pred):
                    # Get annotation onset times and descriptions
                    annotation_onsets = annotations.onset
                    annotation_durations = annotations.duration
                    annotation_descriptions = annotations.description

                    for tmin in epochs.events[:, 0] / raw.info['sfreq']:
                        label = None
                        # Find the annotation that overlaps with the start of the epoch
                        for onset, duration, description in zip(annotation_onsets,
                                                             annotation_durations,
                                                             annotation_descriptions):
                            if onset <= tmin < onset + duration:
                                label = description
                                break
                        labels_true_str.append(label if label else 'Unknown') # Append 'Unknown' if no annotation found

                    # Convert string labels to numerical labels for confusion matrix
                    unique_true_labels = sorted(list(set(labels_true_str)))
                    # Remove 'Unknown' from unique labels if present and not the only label
                    if 'Unknown' in unique_true_labels and len(unique_true_labels) > 1:
                         unique_true_labels.remove('Unknown')

                    label_mapping = {label: i for i, label in enumerate(unique_true_labels)}
                    # Map 'Unknown' to a new index if it exists and is needed in the mapping
                    if 'Unknown' in set(labels_true_str):
                         # Only add 'Unknown' to mapping if it's present in the actual labels
                         if 'Unknown' not in unique_true_labels: # Prevent double adding
                             label_mapping['Unknown'] = len(unique_true_labels)
                             unique_true_labels.append('Unknown') # Add 'Unknown' to the list of labels for target names

                    labels_true_num = [label_mapping.get(label, -1) for label in labels_true_str] # Use .get() with a default for safety

                    # Filter out epochs with 'Unknown' true labels if they should not be included in evaluation
                    # For confusion matrix and classification report, it's best to have corresponding true and predicted labels.
                    # Decide how to handle 'Unknown' true labels. One option is to exclude them from evaluation.
                    valid_indices = [i for i, label in enumerate(labels_true_str) if label != 'Unknown']
                    if len(valid_indices) > 0:
                        labels_true_filtered = [labels_true_num[i] for i in valid_indices]
                        labels_pred_filtered = [labels_pred[i] for i in valid_indices]

                        # Check if there are enough true labels and predicted labels and if they match in length after filtering
                        if len(labels_true_filtered) > 1 and len(labels_true_filtered) == len(labels_pred_filtered):
                            # Creating the confusion matrix requires numerical labels for y_true
                            # Ensure display labels correspond to the numerical labels present in filtered data
                            # Create display labels based on the unique *filtered* numerical true labels
                            unique_filtered_true_nums = sorted(list(set(labels_true_filtered)))
                            filtered_display_labels = [
                                list(label_mapping.keys())[list(label_mapping.values()).index(num)]
                                for num in unique_filtered_true_nums
                            ]


                            cm = confusion_matrix(labels_true_filtered, labels_pred_filtered)

                            # Use the original string labels for display that correspond to the numerical labels in cm
                            disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                                       display_labels=filtered_display_labels)
                            disp.plot(xticks_rotation=45)
                            plt.title('Matrice de confusion: vrais stades vs clusters (époques non-Unknown)')
                            plt.tight_layout()
                            plt.show()

                            # Afficher le rapport de classification
                            # Classification report also works best with numerical labels and target names
                            print("\nRapport de classification :\n")
                            # Use the filtered string labels for target names in the classification report
                            report_target_names_filtered = filtered_display_labels
                            print(classification_report(labels_true_filtered, labels_pred_filtered, target_names=report_target_names_filtered, zero_division=0))

                        else:
                             print("Pas assez de classes différentes dans les annotations (après avoir filtré les Unknown) ou les nombres d'époques ne correspondent pas pour créer une matrice de confusion.")
                    else:
                        print("Aucune époque avec un vrai stade connu pour générer la matrice de confusion et le rapport de classification.")

                else:
                     print(f"Le nombre d'époques ({len(epochs.events)}) ne correspond pas au nombre de prédictions KMeans ({len(labels_pred)}). Impossible de générer la matrice de confusion.")


            except FileNotFoundError:
                print("Fichier d'hypnogramme non trouvé. La matrice de confusion ne sera pas générée.")
                print("Assurez-vous que le fichier 'SC4001EC-Hypnogram.edf' se trouve dans le même répertoire.")

            # Visualisation des clusters UMAP
            plt.figure(figsize=(12, 8))
            scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1],
                                c=labels_pred,
                                cmap='viridis',
                                alpha=0.6,
                                s=50)
            plt.colorbar(scatter, label='Cluster')
            plt.title('Visualisation UMAP des clusters de stades de sommeil')
            plt.xlabel('UMAP 1')
            plt.ylabel('UMAP 2')

            # Ajouter des étiquettes si les vrais labels sont disponibles
            # Use the original string labels for visualization legend
            if 'labels_true_str' in locals() and len(set(labels_true_str)) > 1:
                # Créer une légende pour les vrais stades
                unique_labels_str = sorted(set(labels_true_str))
                for label_str in unique_labels_str:
                    idx = [i for i, l in enumerate(labels_true_str) if l == label_str]
                    if idx:
                        plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1],
                                   label=f'{label_str} (n={len(idx)})',
                                   alpha=0.3, s=10)
                # Adjust legend location to avoid overlapping with plot
                plt.legend(title='Vrais stades', bbox_to_anchor=(1.05, 1), loc='upper left')

            plt.tight_layout()
            plt.show()

            # Afficher la répartition des clusters
            unique, counts = np.unique(labels_pred, return_counts=True)
            print("\nRépartition des clusters :")
            for cluster, count in zip(unique, counts):
                print(f"Cluster {cluster}: {count} époques ({count/len(labels_pred):.1%})")
        else:
            print(f"Pas assez d'échantillons ({X_embedded.shape[0]}) ou nombre de clusters invalide ({n_clusters_kmeans}) pour exécuter KMeans.")

    else:
        print("Aucune donnée extraite pour la réduction de dimension.")

except FileNotFoundError:
    print("Erreur : Le fichier 'SC4001E0-PSG.edf' n'a pas été trouvé. Veuillez vérifier le chemin du fichier.")
except Exception as e:
    print(f"Une erreur est survenue : {e}")
    raise  # Relancer l'exception pour le débogage