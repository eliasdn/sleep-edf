import os
import sys
import glob
import warnings
import logging
import threading
import numpy as np
import pandas as pd
import mne
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score, 
    calinski_harabasz_score, 
    davies_bouldin_score, 
    confusion_matrix
)
from scipy import stats
import umap.umap_ as umap
from sklearn.cluster import KMeans as skKMeans
import hdbscan
import psutil
import concurrent.futures
from functools import partial
import matplotlib.pyplot as plt
from mne import Epochs, events_from_annotations
from mne.io import read_raw_edf
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sleep_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Contexte pour supprimer temporairement les sorties (thread-safe)
class SuppressOutput:
    def __init__(self):
        self._devnull = None
        self._active = False
        self._original_stdout = None
        self._original_stderr = None
        self._lock = threading.Lock()
        self._refcount = 0
    
    def __enter__(self):
        with self._lock:
            if self._refcount == 0:
                self._original_stdout = sys.stdout
                self._original_stderr = sys.stderr
                self._devnull = open(os.devnull, 'w')
                sys.stdout = self._devnull
                sys.stderr = self._devnull
            
            self._refcount += 1
            self._active = True
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        with self._lock:
            if not self._active:
                return False
            
            self._refcount -= 1
            self._active = False
            
            if self._refcount == 0 and self._original_stdout:
                sys.stdout = self._original_stdout
                sys.stderr = self._original_stderr
                if self._devnull:
                    self._devnull.close()
                    self._devnull = None
        return False

# GPU imports
try:
    import cupy as cp
    from cupyx.scipy import signal as cusignal
    from cuml import UMAP as cuUMAP
    from cuml.cluster import KMeans as cuKMeans
    GPU_AVAILABLE = True
except ImportError:
    print("Warning: GPU libraries not available. Falling back to CPU.")
    import numpy as cp  # Fallback to numpy
    from scipy import signal as cusignal
    from sklearn.cluster import KMeans as cuKMeans
    from umap import UMAP as cuUMAP
    GPU_AVAILABLE = False

# Configuration de l'utilisation du GPU
USE_GPU = GPU_AVAILABLE  # Utiliser le GPU si disponible

# Configuration des chemins
PSG_DIR = "drive/MyDrive/A_collab/sleep_data/physiobank_database_sleep-edfx_sleep-cassette"
HYPNO_DIR = "drive/MyDrive/A_collab/sleep_data/physiobank_database_sleep-edfx_sleep-cassette"
OUTPUT_CSV = "sleep_clustering_results.csv"

# Configuration des bandes de fréquence
bands = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 12),
    'beta': (12, 30),
    'gamma': (30, 40)
}

def process_channel(ch, sf, bands, ch_ref=None, position_norm=None):
    """Extrait les caractéristiques d'un canal EEG avec support GPU via CuPy."""
    # Désactiver temporairement les logs pour les performances
    with SuppressOutput():
        # Validation et conversion des données d'entrée
        try:
            # Convertir en array numpy et forcer le type float32
            ch = np.asarray(ch, dtype=np.float32).flatten()
            if ch.ndim != 1 or len(ch) == 0 or np.all(np.isnan(ch)):
                logger.warning(f"Canal invalide: ndim={ch.ndim}, shape={ch.shape}, nans={np.all(np.isnan(ch))}")
                raise ValueError("Canal invalide")
                
            # Si la référence est fournie, la valider aussi
            if ch_ref is not None:
                ch_ref = np.asarray(ch_ref, dtype=np.float32).flatten()
                if ch_ref.ndim != 1 or len(ch_ref) != len(ch) or np.all(np.isnan(ch_ref)):
                    logger.debug("Référence invalide, utilisation de None")
                    ch_ref = None
                    
        except Exception as e:
            logger.error(f"Erreur de validation des données d'entrée: {e}", exc_info=True)
            expected_size = 6 + len(bands) + 3 + (1 if position_norm is not None else 0)
            return [0.0] * expected_size
    
    # Forcer l'initialisation du contexte CUDA si nécessaire
    if USE_GPU:
        try:
            cp.cuda.Device(0).use()  # Force l'initialisation du device GPU
        except Exception as e:
            print(f"Avertissement: Impossible d'initialiser le GPU: {e}")
            return [0.0] * (6 + len(bands) + 3 + (1 if position_norm is not None else 0))
    
    # Préparer les données avec le bon type et sur le bon device
    if USE_GPU:
        try:
            # Conversion explicite et copie pour assurer la contigüité
            ch_xp = cp.asarray(ch, dtype=cp.float32).copy()
            ch_ref_xp = cp.asarray(ch_ref, dtype=cp.float32).copy() if ch_ref is not None else None
            xp = cp
            signal = cusignal
            # Convertir la fréquence d'échantillonnage en float natif
            sf_xp = float(sf)
        except Exception as e:
            print(f"Erreur lors de la préparation des données GPU: {e}")
            return [0.0] * (6 + len(bands) + 3 + (1 if position_norm is not None else 0))
    else:
        try:
            ch_xp = np.asarray(ch, dtype=np.float32)
            ch_ref_xp = np.asarray(ch_ref, dtype=np.float32) if ch_ref is not None else None
            xp = np
            from scipy import signal
            sf_xp = float(sf)
        except Exception as e:
            print(f"Erreur lors de la préparation des données CPU: {e}")
            return [0.0] * (6 + len(bands) + 3 + (1 if position_norm is not None else 0))
    
    features = []
    
    try:
        # 1. Statistiques de base
        ch_cpu = cp.asnumpy(ch_xp) if USE_GPU else ch_xp
        xp = np  # Forcer l'utilisation de numpy pour les calculs CPU
        features.extend([
            float(xp.mean(ch_cpu)), 
            float(xp.std(ch_cpu)), 
            float(stats.skew(ch_cpu)), 
            float(stats.kurtosis(ch_cpu)),
            float(xp.percentile(ch_cpu, 5)), 
            float(xp.percentile(ch_cpu, 95))
        ])
        
        # 2. Analyse spectrale - Toujours utiliser le CPU pour la stabilité
        try:
            from scipy import signal as sp_signal  # Ajout clé pour éviter l'erreur GPU !
            nperseg = min(256, len(ch_xp))
            # Convertir en numpy si nécessaire (pour GPU) et utiliser scipy.signal.welch
            ch_cpu = cp.asnumpy(ch_xp) if USE_GPU else ch_xp
            freqs_cpu, psd_cpu = sp_signal.welch(ch_cpu, fs=sf_xp, nperseg=int(nperseg))
                
        except Exception as e:
            print(f"Erreur dans l'analyse spectrale: {e}")
            # Retourner un vecteur de zéros de la bonne taille
            expected_size = 6 + len(bands) + 3 + (1 if position_norm is not None else 0)
            return [0.0] * expected_size
            
        # Ajouter les bandes de fréquence
        for band_name, (fmin, fmax) in bands.items():
            band_mask = (freqs_cpu >= fmin) & (freqs_cpu <= fmax)
            if np.any(band_mask):
                band_power = float(np.sum(psd_cpu[band_mask]))
                features.append(band_power)
            else:
                features.append(0.0)
        
        # 3. Cohérence avec un autre canal (si fourni et activé)
        ENABLE_COHERENCE = False  # Désactivé par défaut pour les tests
        if ENABLE_COHERENCE and ch_ref is not None and ch_ref_xp is not None:
            try:
                f, coh = sp_signal.coherence(  # utiliser sp_signal ici aussi
                    ch_cpu,
                    cp.asnumpy(ch_ref_xp) if USE_GPU else ch_ref_xp,
                    fs=sf, 
                    nperseg=nperseg
                )
                coh_mean = float(np.mean(coh))
                features.append(coh_mean)
            except Exception as e:
                print(f"Erreur de cohérence: {e}")
                features.append(0.0)
        else:
            features.append(0.0)
        
        # 4. Paramètres de Hjorth (désactivés par défaut pour les tests)
        ENABLE_HJORTH = False
        
        if ENABLE_HJORTH:
            try:
                diff1 = xp.diff(ch_xp, 1)
                diff2 = xp.diff(ch_xp, 2)
                
                var_ch = xp.var(ch_xp)
                var_diff1 = xp.var(diff1)
                var_diff2 = xp.var(diff2)
                
                mobility = float(xp.sqrt(var_diff1 / var_ch))
                complexity = float(xp.sqrt((var_diff2 * var_ch) / (var_diff1 ** 2)))
                
                features.extend([mobility, complexity])
            except Exception as e:
                print(f"Erreur calcul Hjorth: {e}")
                features.extend([0.0, 0.0])
        else:
            features.extend([0.0, 0.0])
        
        # 5. Position relative (si fournie)
        if position_norm is not None:
            features.append(float(position_norm))
            
        # Nettoyer la mémoire GPU si nécessaire
        if USE_GPU:
            cp.get_default_memory_pool().free_all_blocks()
            
    except Exception as e:
        print(f"Erreur dans process_channel: {e}")
        # Retourner un vecteur de zéros de la bonne taille en cas d'erreur
        expected_size = 6 + len(bands) + 3  # stats + bandes + cohérence + hjorth
        if position_norm is not None:
            expected_size += 1
        features = [0.0] * expected_size
    
    return features

def load_hypnogram(hypnogram_path):
    try:
        raw_annot = mne.io.read_raw_edf(hypnogram_path, preload=True, verbose=False)
        annotations = raw_annot.annotations

        stage_mapping = {
            'Sleep stage W': 0,
            'Sleep stage 1': 1,
            'Sleep stage 2': 2,
            'Sleep stage 3': 3,
            'Sleep stage 4': 3,
            'Sleep stage R': 4,
            'Sleep stage ?': -1,
            'Movement time': -1
        }

        mapped_stages = []
        for desc in annotations.description:
            mapped_stages.append(stage_mapping.get(desc, -1))

        if not mapped_stages:
            raise ValueError("Aucune annotation de stade de sommeil trouvée")

        return np.array(mapped_stages)

    except Exception as e:
        print(f"Erreur lors du chargement de l'hypnogramme {os.path.basename(hypnogram_path)}: {str(e)}")
        return None

def find_hypnogram(psg_path):
    base_prefix = os.path.basename(psg_path)[:7]  # 'SC4001E'
    hypnogram_pattern = base_prefix + '*-Hypnogram.edf'
    folder = os.path.dirname(psg_path)
    matches = glob.glob(os.path.join(folder, hypnogram_pattern))
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        print(f"⚠️ Plusieurs hypnogrammes trouvés pour {psg_path} → {matches}")
        return matches[0]
    else:
        print(f"❌ Aucun hypnogramme trouvé pour {psg_path}")
        return None


def load_edf_with_annotations(psg_path, hypnogram_path, epoch_length=30.0):
    import mne
    import numpy as np
    
    # Charger le PSG
    import io
    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
    sfreq = raw.info['sfreq']
    n_samples = raw.n_times
    n_secs = n_samples / sfreq
    n_epochs = int(np.floor(n_secs / epoch_length))

    # Charger les annotations de l'hypnogramme avec mne.read_annotations
    annotations = mne.read_annotations(hypnogram_path)
    
    # Appliquer les annotations au signal brut pour assurer l'alignement temporel
    raw.set_annotations(annotations)
    
    # Créer les epochs fixes AVANT de récupérer les événements
    # pour s'assurer que les temps sont correctement alignés
    epochs = mne.make_fixed_length_epochs(raw, duration=epoch_length, preload=True)
    
    # Récupérer les événements à partir des annotations
    events, event_id = mne.events_from_annotations(raw)
    
    # Créer un mapping inverse ID -> description
    id_to_description = {v: k for k, v in event_id.items()}
    
    # Mapping des descriptions vers labels numériques
    stage_mapping = {
        'Sleep stage W': 0,
        'Sleep stage 1': 1,
        'Sleep stage 2': 2,
        'Sleep stage 3': 3,
        'Sleep stage 4': 3,  # Sleep stage 3 et 4 fusionnés (convention AASM)
        'Sleep stage R': 4,
        'Sleep stage ?': -1,
        'Movement time': -1,
        'Sleep stage S': 3,  # Certains datasets utilisent 'S' pour le sommeil profond
        'Sleep stage 4': 3   # Redondant mais plus sûr
    }
    
    # Initialiser les labels à -1
    true_labels = np.full(len(epochs), fill_value=-1, dtype=int)
    
    # Remplir true_labels selon les événements
    for event in events:
        onset = event[0] / sfreq  # Convertir l'échantillon en secondes
        desc_id = event[2]  # ID numérique de l'événement
        desc_str = id_to_description.get(desc_id, '')  # Description textuelle
        
        # Obtenir le label correspondant à la description
        stage_label = stage_mapping.get(desc_str, -1)
        
        # Si on n'a pas trouvé de correspondance directe, essayer une correspondance partielle
        if stage_label == -1 and desc_str:
            for key, value in stage_mapping.items():
                if key in desc_str:
                    stage_label = value
                    break
        
        if stage_label != -1:
            start_epoch = int(onset // epoch_length)
            # Marquer cette époque avec le label correspondant
            if start_epoch < len(true_labels):
                true_labels[start_epoch] = stage_label
    
    # Lisser les labels pour remplir les époques sans annotation
    # en propageant le dernier label valide
    last_valid = -1
    for i in range(len(true_labels)):
        if true_labels[i] != -1:
            last_valid = true_labels[i]
        elif last_valid != -1:
            true_labels[i] = last_valid
    
    # Créer les epochs fixes
    epochs = mne.make_fixed_length_epochs(raw, duration=epoch_length, preload=True)
    
    # Vérification de cohérence
    n_epochs_final = len(epochs)
    if n_epochs_final != len(true_labels):
        print(f"⚠️ Mismatch epochs ({n_epochs_final}) vs labels ({len(true_labels)}), ajustement...")
        min_len = min(n_epochs_final, len(true_labels))
        true_labels = true_labels[:min_len]
        epochs = epochs[:min_len]
    
    # Récupérer les informations avant toute opération qui pourrait fermer le fichier
    n_epochs = len(epochs)
    n_channels = len(epochs.ch_names)
    
    # Créer une copie des epochs pour les désolidariser du fichier brut
    epochs_copy = epochs.copy()
    
    # Libérer la mémoire du raw et des epochs d'origine
    del raw
    del epochs
    
    print(f"✅ Données chargées: {n_epochs} époques, {n_channels} canaux")
    return epochs_copy, true_labels


def process_single_psg_file(psg_path):
    """Traite un seul fichier PSG et retourne un DataFrame avec les résultats."""
    psg_filename = os.path.basename(psg_path)
    subject_id = psg_filename.split('-')[0]
    logger.info(f"Début du traitement du fichier: {psg_filename}")
    
    try:
        # 1. Trouver le fichier hypnogramme correspondant
        hypno_path = find_hypnogram(psg_path)
        if not hypno_path:
            logger.warning(f"Aucun hypnogramme trouvé pour {psg_filename}")
            return pd.DataFrame()
            
        logger.info(f"Fichier hypnogramme trouvé: {os.path.basename(hypno_path)}")
        
        # 2. Charger les données EDF brutes et les annotations
        try:
            with SuppressOutput():
                # Charger les données EDF et les annotations en une seule fois
                epochs, true_labels = load_edf_with_annotations(psg_path, hypno_path)
                
                # Vérifier que le chargement s'est bien passé
                if epochs is None or true_labels is None or len(epochs) == 0:
                    logger.error("Échec du chargement des données EDF et des annotations")
                    return pd.DataFrame()
                    
                logger.info(f"Données EDF et annotations chargées: {len(epochs)} époques")
                
                # Extraire les données brutes des époques
                data = epochs.get_data()
                sfreq = epochs.info['sfreq']
                n_epochs, n_channels, n_times = data.shape
                
                # Filtrer les canaux pour ne garder que les EEG
                eeg_indices = [i for i, ch_name in enumerate(epochs.ch_names) if ch_name.startswith('EEG ')]
                if not eeg_indices:
                    logger.error("Aucun canal EEG trouvé dans les données chargées")
                    return pd.DataFrame()
                    
                data = data[:, eeg_indices, :]
                n_channels = len(eeg_indices)
                
                # Appliquer un filtre passe-bande
                epochs.filter(0.5, 30., fir_design='firwin', verbose=False)
                logger.info("Filtrage appliqué avec succès")
                
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données EDF et annotations: {e}", exc_info=True)
            return pd.DataFrame()
        
        # 3. Vérifier que nous avons des données valides
        if n_epochs == 0 or n_channels == 0 or n_times == 0:
            logger.error("Données d'entrée invalides (époques, canaux ou temps manquants)")
            return pd.DataFrame()
            
        logger.info(f"Données prêtes pour l'extraction des caractéristiques: {n_epochs} époques x {n_channels} canaux x {n_times} points")
        logger.info(f"Nombre de vrais labels chargés: {len(true_labels)}")
        
        # 4. Vérifier la cohérence entre les données et les labels
        if n_epochs != len(true_labels):
            logger.warning(f"Incohérence détectée: {n_epochs} époques mais {len(true_labels)} labels. Ajustement...")
            min_len = min(n_epochs, len(true_labels))
            data = data[:min_len]
            true_labels = true_labels[:min_len]
            n_epochs = min_len
            logger.info(f"Données et labels ajustés à {n_epochs} époques")
        
        # 5. Préparer les arguments pour le traitement parallèle
        args_list = []
        for i in range(n_epochs):
            ch_ref = data[i, 0] if n_channels > 0 else None
            position_norm = i / n_epochs
            
            for j in range(n_channels):
                # Vérifier que les données ne contiennent pas de NaN ou Inf
                channel_data = data[i, j]
                if np.any(np.isnan(channel_data)) or np.any(np.isinf(channel_data)):
                    logger.warning(f"Données invalides détectées dans l'époque {i}, canal {j}. Remplacement par des zéros.")
                    channel_data = np.nan_to_num(channel_data, nan=0.0, posinf=0.0, neginf=0.0)
                
                args_list.append((channel_data, sfreq, bands, ch_ref, position_norm))
        
        # 7. Traitement des canaux (séquentiel pour GPU, parallèle pour CPU)
        features_list = []
        
        try:
            if USE_GPU:
                # Exécution séquentielle pour le GPU (plus stable avec CUDA)
                logger.info("  ⚠️  Mode GPU activé - Traitement séquentiel pour stabilité")
                for args in tqdm(args_list, 
                              desc="  - Extraction des features (GPU)",
                              leave=False,
                              mininterval=5.0):
                    try:
                        features_list.append(process_channel(*args))
                    except Exception as e:
                        logger.warning(f"\n⚠️ Erreur lors du traitement d'un canal: {e}")
                        features_list.append([0.0] * 20)  # Taille par défaut
            else:
                # Exécution parallèle pour le CPU
                num_workers = min(psutil.cpu_count(logical=False), psutil.cpu_count(logical=False)) * 2
                with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = [executor.submit(process_channel, *args) for args in args_list]
                    
                    for future in tqdm(concurrent.futures.as_completed(futures), 
                                     total=len(futures),
                                     desc="  - Extraction des features (CPU)",
                                     leave=False,
                                     mininterval=5.0):
                        try:
                            features_list.append(future.result())
                        except Exception as e:
                            logger.warning(f"\n⚠️ Erreur lors du traitement d'un canal: {e}")
                            features_list.append([0.0] * 20)  # Taille par défaut
                
            if not features_list:
                logger.error("  ❌ Aucune feature extraite")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"  ❌ Erreur lors de l'extraction des features: {str(e)}")
            return pd.DataFrame()
            
        # 8. Préparation des données pour le clustering
        try:
            logger.info("  - Préparation des données pour le clustering...")
            X = np.array(features_list)
            
            # Vérifier les NaN/Inf
            if np.isnan(X).any() or np.isinf(X).any():
                logger.info("  - Remplacement des valeurs NaN/Inf...")
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Normalisation
            logger.info("  - Normalisation des données...")
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Remodeler pour avoir (n_epochs, n_channels * n_features)
            X_reshaped = X_scaled.reshape(n_epochs, -1)
            
            # Vérifier la dimension finale
            if X_reshaped.shape[1] == 0:
                logger.error("  ❌ Aucune feature valide après préparation")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"  ❌ Erreur lors de la préparation des données: {str(e)}")
            return pd.DataFrame()
        
        # 9. Benchmarking des méthodes de réduction de dimension et de clustering
        logger.info("  - Démarrage du benchmark...")
        results = []  # Initialisation de la liste des résultats
        
        # Configurations à tester
        reducers = [
            ('UMAP', umap.UMAP(n_components=2, random_state=42, n_jobs=-1) if not USE_GPU 
              else cuUMAP(n_components=2, random_state=42, output_type='numpy')),
            ('PCA', PCA(n_components=2, random_state=42)),
            ('None', None)
        ]
        
        clusterers = [
            ('KMeans', lambda: cuKMeans(n_clusters=5, random_state=42, output_type='numpy') if USE_GPU 
                          else skKMeans(n_clusters=5, random_state=42, n_init=10)),
            ('HDBSCAN', lambda: hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5, gen_min_span_tree=True))
        ]
        
        # Boucle sur les configurations
        for reducer_name, reducer in reducers:
            try:
                logger.info(f"\n  - Test de {reducer_name}...")
                
                # Appliquer la réduction de dimension
                if reducer_name == 'None':
                    X_transformed = X_reshaped
                else:
                    try:
                        if USE_GPU and reducer_name == 'UMAP':
                            X_gpu = cp.asarray(X_reshaped.astype(np.float32))
                            X_transformed = reducer.fit_transform(X_gpu)
                            del X_gpu
                            cp.get_default_memory_pool().free_all_blocks()
                        else:
                            X_transformed = reducer.fit_transform(X_reshaped)
                    except Exception as e:
                        logger.error(f"    ❌ Erreur avec {reducer_name}: {str(e)}")
                        continue
                
                nb_clusters_with_valid_labels = 0
                nb_clusters_without_valid_labels = 0
                # Boucle sur les méthodes de clustering
                for clusterer_name, clusterer_factory in clusterers:
                    try:
                        logger.info(f"    - Test de {clusterer_name}...")
                        
                        # Entraîner le modèle de clustering
                        clusterer = clusterer_factory()
                        
                        if USE_GPU and clusterer_name == 'KMeans':
                            X_cluster = cp.asarray(X_transformed.astype(np.float32))
                            clusterer.fit(X_cluster)
                            labels = clusterer.labels_
                            del X_cluster
                            cp.get_default_memory_pool().free_all_blocks()
                        else:
                            if hasattr(clusterer, 'fit_predict'):
                                labels = clusterer.fit_predict(X_transformed)
                            else:
                                clusterer.fit(X_transformed)
                                labels = clusterer.labels_
                        
                        # Calculer les métriques
                        metrics = {
                            'psg_file': psg_filename,
                            'reducer': reducer_name,
                            'clusterer': clusterer_name,
                            'n_clusters': len(np.unique(labels[labels >= 0])),
                            'n_noise': np.sum(labels == -1) if clusterer_name == 'HDBSCAN' else 0
                        }
                        
                        # Métriques de clustering
                        if len(np.unique(labels[labels >= 0])) > 1:
                            try:
                                metrics['silhouette'] = silhouette_score(X_transformed, labels)
                                metrics['calinski_harabasz'] = calinski_harabasz_score(X_transformed, labels)
                                metrics['davies_bouldin'] = davies_bouldin_score(X_transformed, labels)
                            except Exception as e:
                                logger.error(f"Erreur métriques: {str(e)}")
                        
                        # Métriques par rapport aux annotations
                        if true_labels is not None and len(true_labels) == len(labels):
                            try:
                                # Créer un mappage cluster -> label majoritaire (avec logs + robustesse)
                                cluster_to_label = {}
                                for cluster in np.unique(labels[labels >= 0]):
                                    mask = (labels == cluster)
                                    true_in_cluster = true_labels[mask]

                                    # Filtrer les labels valides
                                    valid_true_in_cluster = true_in_cluster[true_in_cluster >= 0]
                                    total_in_cluster = len(true_in_cluster)
                                    valid_count = len(valid_true_in_cluster)

                                    if valid_count > 0:
                                        try:
                                            # Calcul du label majoritaire
                                            if valid_true_in_cluster.size > 0:
                                                bincount = np.bincount(valid_true_in_cluster)
                                                majority_label = np.argmax(bincount)
                                                cluster_to_label[cluster] = majority_label

                                                # Logging détaillé
                                                label_counts = np.bincount(valid_true_in_cluster)
                                                logger.debug(f"Cluster {cluster}: {valid_count}/{total_in_cluster} labels valides")
                                                logger.debug(f"  Répartition: {dict(zip(np.nonzero(label_counts)[0], label_counts[label_counts > 0]))}")
                                                logger.debug(f"  Label majoritaire: {majority_label} ({(np.max(bincount) / valid_count)*100:.1f}%)")
                                        except Exception as e:
                                            logger.warning(f"Erreur lors du calcul du label majoritaire pour le cluster {cluster}: {e}")
                                            cluster_to_label[cluster] = -1
                                    else:
                                        cluster_to_label[cluster] = -1
                                        
                                # Prédire les labels pour tous les points
                                predicted_labels = np.full_like(labels, -1)
                                for cluster, label in cluster_to_label.items():
                                    predicted_labels[labels == cluster] = label
                                
                                # Calculer la pureté (uniquement sur les points avec des vrais labels)
                                valid_mask = (true_labels >= 0) & (predicted_labels >= 0)
                                if np.any(valid_mask):
                                    correct = np.sum(true_labels[valid_mask] == predicted_labels[valid_mask])
                                    total = np.sum(valid_mask)
                                    metrics['purity'] = correct / total
                                    metrics['n_valid_pairs'] = total
                                    
                                    # Calculer la matrice de confusion
                                    conf_matrix = confusion_matrix(
                                        true_labels[valid_mask], 
                                        predicted_labels[valid_mask],
                                        labels=np.unique(true_labels[valid_mask])
                                    )
                                    metrics['confusion_matrix'] = conf_matrix
                                    
                                    # Log des informations de pureté
                                    logger.info(f"Purity: {metrics['purity']:.3f} ({correct}/{total} points corrects)")
                                    logger.debug("Matrice de confusion:\n%s", conf_matrix)
                                    
                            except Exception as e:
                                logger.error(f"Erreur lors du calcul de la pureté: {e}", exc_info=True)
                        
                        results.append(metrics)
                        logger.info(f"{clusterer_name} terminé avec {metrics['n_clusters']} clusters")
                        
                    except Exception as e:
                        logger.error(f"Erreur avec {clusterer_name}: {e}", exc_info=True)
                        continue
                        
            except Exception as e:
                logger.error(f"Erreur lors du benchmark: {e}", exc_info=True)
                return pd.DataFrame()
        
        # Créer un DataFrame avec les résultats
        if not results:
            logger.error("Aucun résultat à enregistrer")
            return pd.DataFrame()
            
        results_df = pd.DataFrame(results)
        results_df['subject_id'] = subject_id
        results_df['psg_file'] = psg_filename
        
        # Enregistrer les résultats
        results_df.to_csv(OUTPUT_CSV, mode='a', header=not os.path.exists(OUTPUT_CSV), index=False)
        logger.info(f"Résultats enregistrés dans {OUTPUT_CSV}")
        
        return results_df
        
    except Exception as e:
        logger.critical(f"ERREUR CRITIQUE lors du traitement de {psg_filename}: {str(e)}", exc_info=True)
        return pd.DataFrame()

def process_channel_wrapper(args):
    """Wrapper pour le traitement parallèle des canaux."""
    return process_channel(*args)

def main():
    # Vérifier et créer le dossier de sortie
    os.makedirs(os.path.dirname(OUTPUT_CSV) or '.', exist_ok=True)
    
    # Lister les fichiers PSG
    psg_files = glob.glob(os.path.join(PSG_DIR, '*-PSG.edf'))
    if not psg_files:
        print(f"Aucun fichier PSG trouvé dans {PSG_DIR}")
        return
        
    print(f"Traitement de {len(psg_files)} fichiers PSG...")
    
    # Limiter le nombre de fichiers pour les tests
    # psg_files = psg_files[:1]  # Décommenter pour tester avec un seul fichier
    
    # Configuration du traitement
    max_psg_workers = min(psutil.cpu_count(logical=False), 4)  # Ajuster selon la mémoire GPU disponible
    results = []
    
    if USE_GPU:
        # Mode hybride CPU + GPU → ThreadPoolExecutor pour paralléliser le préprocessing
        logger.info("⚠️  Mode GPU hybride : ThreadPool CPU pour chargement + process_channel GPU séquentiel")
        
        from concurrent.futures import ThreadPoolExecutor, as_completed

        try:
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                for psg_file in psg_files:
                    futures.append(executor.submit(process_single_psg_file, psg_file))
                
                for future in tqdm(as_completed(futures), total=len(futures),
                               desc="Traitement des fichiers PSG (hybride CPU/GPU)", unit="fichier"):
                    try:
                        result = future.result()
                        if not result.empty:
                            results.append(result)
                            # Sauvegarde incrémentielle
                            pd.concat(results).to_csv(OUTPUT_CSV, index=False)
                            logger.debug("Résultats sauvegardés")
                    except Exception as e:
                        logger.error(f"Erreur lors du traitement d'un fichier PSG: {str(e)}", exc_info=True)
                        continue

        except Exception as e:
            logger.critical("Erreur critique dans le traitement hybride", exc_info=True)
            raise
        
    else:
        # Mode CPU → traitement parallèle avec ThreadPoolExecutor (classique)
        logger.info(f"Lancement du traitement parallèle sur {max_psg_workers} PSG en parallèle (CPU)")
        
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_psg_workers) as executor:
                future_to_psg = {
                    executor.submit(process_single_psg_file, psg_file): psg_file 
                    for psg_file in psg_files
                }
                
                for future in tqdm(
                    concurrent.futures.as_completed(future_to_psg),
                    total=len(psg_files),
                    desc="Traitement des fichiers PSG (CPU)",
                    unit="fichier"
                ):
                    psg_file = future_to_psg[future]
                    try:
                        result = future.result()
                        if not result.empty:
                            results.append(result)
                            # Sauvegarde incrémentielle
                            pd.concat(results).to_csv(OUTPUT_CSV, index=False)
                            logger.debug(f"Résultats sauvegardés pour {os.path.basename(psg_file)}")
                    except Exception as e:
                        logger.error(f"Erreur lors du traitement de {os.path.basename(psg_file)}: {str(e)}",
                                     exc_info=True)
        except Exception as e:
            logger.critical("Erreur critique dans le traitement parallèle (CPU)", exc_info=True)
            raise
            
    if results:
        final_df = pd.concat(results, ignore_index=True)
        final_df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nRésultats sauvegardés dans {os.path.abspath(OUTPUT_CSV)}")
        print(f"Nombre total d'analyses: {len(final_df)}")
        
        ### Ajout → Calcul du score de purity global ###
        print("\n✅ Calcul du score de purity global sur l'ensemble du benchmark...")
        
        # Récupérer toutes les matrices de confusion
        all_cm = []
        total_samples = 0
        correct_samples = 0
        
        for cm_str in final_df['confusion_matrix'].dropna():
            cm = np.array(eval(cm_str))  # Convertir la liste en np.array
            all_cm.append(cm)
            
            # Pour purity: on somme les max par ligne
            correct_samples += np.sum(np.max(cm, axis=1))
            total_samples += np.sum(cm)
        
        # Calcul du purity global
        if total_samples > 0:
            global_purity = correct_samples / total_samples
            print(f"🎯 Score de purity global : {global_purity:.4f}")
        else:
            print("⚠️ Pas de matrice de confusion valide pour calculer la purity globale.")


if __name__ == "__main__":
    main()
