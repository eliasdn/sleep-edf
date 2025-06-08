#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pipeline simplifié pour l'analyse des stades de sommeil à partir de signaux EEG.
"""

import os
import glob
import logging
import warnings
import numpy as np
import pandas as pd
import mne
from tqdm import tqdm
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans
import hdbscan
from scipy import signal, stats
from antropy import app_entropy, sample_entropy, petrosian_fd, katz_fd, detrended_fluctuation

# Configuration
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paramètres
DATA_DIR = "sleep_data/physiobank_database_sleep-edfx_sleep-cassette"
PSG_PATTERN = os.path.join(DATA_DIR, "*-PSG.edf")
HYPNO_PATTERN = os.path.join(DATA_DIR, "*-Hypnogram.edf")
OUTPUT_CSV = "sleep_clustering_results.csv"
CPU_COUNT = max(1, os.cpu_count() - 1)
EEG_BANDS = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 12), 'beta': (12, 30), 'gamma': (30, 40)}

def load_edf_with_annotations(edf_path, hypno_path):
    """Charge un fichier EDF et son hypnogramme."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            
        annotations = mne.read_annotations(hypno_path)
        if not annotations:
            logger.warning(f"Aucune annotation trouvée dans {os.path.basename(hypno_path)}")
            events = mne.make_fixed_length_events(raw, id=1, duration=30, overlap=0)
            epochs = mne.Epochs(raw, events, tmin=0, tmax=30, baseline=None, preload=True, verbose=False)
            return raw, epochs, ['Unknown'] * len(epochs)
            
        raw.set_annotations(annotations)
        events, event_id = mne.events_from_annotations(raw, event_id='auto', chunk_duration=30, verbose=False)
        epochs = mne.Epochs(raw, events, event_id=event_id, tmin=0, tmax=30, baseline=None, preload=True, verbose=False)
        true_labels = [next((k for k, v in event_id.items() if v == e[2]), 'Unknown') for e in events]
        return raw, epochs, true_labels
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement de {os.path.basename(edf_path)}: {str(e)}")
        return None, None, None

def process_channel(ch_data, sfreq, ch_ref=None):
    """Calcule les caractéristiques d'un canal EEG."""
    features = []
    
    # Statistiques de base
    features.extend([np.mean(ch_data), np.std(ch_data), np.min(ch_data), np.max(ch_data), 
                    np.median(ch_data), np.ptp(ch_data), stats.skew(ch_data), stats.kurtosis(ch_data)])
    
    # Entropies
    for func in [lambda x: app_entropy(x, order=2), lambda x: sample_entropy(x, order=2, r=0.2*np.std(x))]:
        try: features.append(func(ch_data))
        except: features.append(0.0)
    
    # Dimensions fractales
    for func in [petrosian_fd, katz_fd, detrended_fluctuation]:
        try: features.append(func(ch_data))
        except: features.append(0.0)
    
    # Analyse spectrale
    freqs, psd = signal.welch(ch_data, fs=sfreq, nperseg=min(256, len(ch_data)))
    for fmin, fmax in EEG_BANDS.values():
        band_mask = (freqs >= fmin) & (freqs <= fmax)
        band_power = np.trapz(psd[band_mask], freqs[band_mask]) if np.any(band_mask) else 0.0
        features.append(band_power)
    
    # Cohérence avec un autre canal (si fourni)
    if ch_ref is not None:
        try:
            f, coh = signal.coherence(ch_data, ch_ref, fs=sfreq, nperseg=min(256, len(ch_data)))
            features.append(np.mean(coh))
        except:
            features.append(0.0)
    else:
        features.append(0.0)
    
    return features

def extract_features(epochs):
    """Extrait les caractéristiques d'un ensemble d'époques EEG."""
    try:
        n_epochs, n_channels = len(epochs), len(epochs.ch_names)
        logger.info(f"Traitement de {n_epochs} époques et {n_channels} canaux...")
        
        # Traitement parallèle
        with ProcessPoolExecutor(max_workers=CPU_COUNT) as executor:
            args = [(epochs[i].squeeze()[ch_idx], epochs.info['sfreq'], 
                    None if ch_idx == 0 else epochs[i].squeeze()[0])
                   for i in range(n_epochs) for ch_idx in range(n_channels)]
            
            results = list(tqdm(executor.map(lambda x: process_channel(*x), args), 
                              total=len(args), desc="Traitement des canaux"))
        
        # Mise en forme des résultats
        features = np.array(results).reshape(n_epochs, -1)
        features = features[:, ~np.isnan(features).any(axis=0)]
        return StandardScaler().fit_transform(features)
        
    except Exception as e:
        logger.error(f"Erreur lors de l'extraction des caractéristiques: {str(e)}")
        return None
    
def apply_clustering(features, method='kmeans', n_clusters=5):
    """Applique un algorithme de clustering."""
    try:
        if method == 'kmeans':
            model = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        else:  # hdbscan
            model = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=2, 
                                 cluster_selection_epsilon=0.5, cluster_selection_method='eom')
        return model.fit_predict(StandardScaler().fit_transform(features)), model
def evaluate_clustering(features, labels, true_labels=None):
    """Évalue la qualité du clustering."""
    metrics = {}
    if len(np.unique(labels)) < 2:
        return {'silhouette': -1, 'calinski_harabasz': -1, 'davies_bouldin': float('inf'), 'purity': 0.0}
    
    for name, func in [('silhouette', silhouette_score), 
                       ('calinski_harabasz', calinski_harabasz_score)]:
        try: metrics[name] = func(features, labels)
        except: metrics[name] = -1
    
    try: metrics['davies_bouldin'] = davies_bouldin_score(features, labels)
    except: metrics['davies_bouldin'] = float('inf')
    
    if true_labels is not None and len(true_labels) == len(labels):
        try:
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(true_labels, labels)
            metrics['purity'] = np.sum(np.amax(cm, axis=0)) / np.sum(cm)
        except: metrics['purity'] = 0.0
    
    return metrics

def process_single_psg(psg_path):
    """Traite un seul fichier PSG."""
    try:
        # Trouver et charger l'hypnogramme
        base_name = os.path.basename(psg_path).replace('-PSG.edf', '')
        hypno_matches = glob.glob(psg_path.replace(base_name, f"{base_name[:-1]}*-Hypnogram.edf"))
        if not hypno_matches: return None
        
        # Charger les données
        raw, epochs, true_labels = load_edf_with_annotations(psg_path, hypno_matches[0])
        if raw is None: return None
        
        # Extraire les caractéristiques et effectuer le clustering
        features = extract_features(epochs)
        if features is None: return None
        
        results = []
        for method in ['kmeans', 'hdbscan']:
            labels, _ = apply_clustering(features, method)
            if labels is None: continue
                
            metrics = evaluate_clustering(features, labels, true_labels)
            result = {
                'file': os.path.basename(psg_path),
                'method': method,
                'n_clusters': len(np.unique(labels)),
                **metrics
            }
            results.append(result)
            logger.info(f"{result['file']} - {method}: {metrics['purity']:.2f} purity, {metrics['silhouette']:.2f} silhouette")
        
        return results
        
    except Exception as e:
        logger.error(f"Erreur avec {psg_path}: {str(e)}")
        return None

def main():
    """Fonction principale."""
    psg_files = glob.glob(PSG_PATTERN)
    if not psg_files:
        logger.error("Aucun fichier PSG trouvé")
        return 1
    
    logger.info(f"Traitement de {len(psg_files)} fichiers PSG")
    all_results = []
    
    # Traitement parallèle
    with ProcessPoolExecutor(max_workers=CPU_COUNT) as executor:
        futures = {executor.submit(process_single_psg, psg_file): psg_file for psg_file in psg_files}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), 
                          desc="Traitement des fichiers PSG"):
            try:
                result = future.result()
                if result: all_results.extend(result)
            except Exception as e:
                logger.error(f"Erreur: {str(e)}")
    
    # Sauvegarder les résultats
    if all_results:
        df_results = pd.DataFrame(all_results)
        df_results.to_csv(OUTPUT_CSV, index=False)
        logger.info(f"Résultats sauvegardés dans {OUTPUT_CSV}")
        
        # Afficher un résumé
        summary = df_results.groupby('method').agg({
            'purity': 'mean',
            'silhouette': 'mean',
            'davies_bouldin': 'mean',
            'n_clusters': 'mean'
        })
        logger.info("\nRésumé des performances:\n" + str(summary))
    else:
        logger.warning("Aucun résultat à sauvegarder")
    
    return 0

def process_channel_wrapper(args):
    """Wrapper pour le traitement parallèle des canaux."""
    return process_channel(*args)

if __name__ == "__main__":
    import sys
    sys.exit(main())
