import os
import glob
import numpy as np
import pandas as pd
import mne
import umap.umap_ as umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm
import hdbscan
from antropy import app_entropy, sample_entropy, spectral_entropy, petrosian_fd, katz_fd, detrended_fluctuation
from scipy import stats, signal
from scipy.fft import fft
import antropy as ant
import matplotlib.pyplot as plt
import seaborn as sns
from mne import Epochs, events_from_annotations
from mne.io import read_raw_edf
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configuration des chemins
PSG_DIR = "sleep_data/physiobank_database_sleep-edfx_sleep-cassette"
HYPNO_DIR = "sleep_data/physiobank_database_sleep-edfx_sleep-cassette"
OUTPUT_CSV = "sleep_clustering_results.csv"

# Configuration des bandes de fréquence
bands = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 12),
    'beta': (12, 30),
    'gamma': (30, 40)
}

# Configuration des chemins
PSG_DIR = "sleep_data/physiobank_database_sleep-edfx_sleep-cassette"
HYPNO_DIR = "sleep_data/physiobank_database_sleep-edfx_sleep-cassette"
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
    """Extrait les caractéristiques d'un canal EEG."""
    features = []
    
    # 1. Entropies
    try:
        pe = ant.perm_entropy(ch, order=3, delay=1, normalize=True)
        features.append(pe)
    except Exception as e:
        features.append(0.0)
        
    try:
        se = sample_entropy(ch, order=2, metric='chebyshev')
        features.append(se)
    except:
        features.append(0.0)
        
    try:
        ae = app_entropy(ch, order=2, metric='chebyshev')
        features.append(ae)
    except:
        try:
            ae = app_entropy(ch)
            features.append(ae)
        except:
            features.append(0.0)
    
    # 2. Fractales
    try:
        pfd = petrosian_fd(ch)
        features.append(pfd)
    except:
        features.append(0.0)
        
    try:
        kfd = katz_fd(ch)
        features.append(kfd)
    except:
        features.append(0.0)
        
    try:
        dfa = detrended_fluctuation(ch)
        features.append(dfa)
    except:
        features.append(0.0)
    
    # 3. Statistiques de base
    features.extend([
        np.mean(ch), np.std(ch), stats.skew(ch), stats.kurtosis(ch),
        np.percentile(ch, 5), np.percentile(ch, 95)
    ])
    
    # 4. Analyse spectrale
    freqs, psd = signal.welch(ch, sf, nperseg=min(256, len(ch)))
    for band_name, (fmin, fmax) in bands.items():
        band_mask = (freqs >= fmin) & (freqs <= fmax)
        if np.any(band_mask):
            band_power = np.sum(psd[band_mask])
            features.append(band_power)
        else:
            features.append(0.0)
    
    # 5. Cohérence avec un autre canal (si fourni)
    if ch_ref is not None:
        try:
            f, coh = signal.coherence(ch, ch_ref, fs=sf, nperseg=min(256, len(ch)))
            features.append(np.mean(coh))
        except:
            features.append(0.0)
    else:
        features.append(0.0)
    
    # 6. Paramètres de Hjorth
    diff1 = np.diff(ch, 1)
    diff2 = np.diff(ch, 2)
    
    # Mobilité
    mobility = np.sqrt(np.var(diff1) / np.var(ch))
    # Complexité
    complexity = np.sqrt(np.var(diff2) * np.var(ch) / np.var(diff1) ** 2)
    
    features.extend([mobility, complexity])
    
    # 7. Position relative dans l'enregistrement (si fournie)
    if position_norm is not None:
        features.append(position_norm)
    
    return features

def load_edf_with_annotations(edf_path, hypnogram_path):
    """Charge un fichier EDF avec ses annotations."""
    try:
        # Charger le fichier EDF
        raw = read_raw_edf(edf_path, preload=True)
        
        # Lire les annotations
        annot = mne.read_annotations(hypnogram_path)
        
        # Créer des événements à partir des annotations
        event_id = {
            'Sleep stage W': 0,
            'Sleep stage 1': 1,
            'Sleep stage 2': 2,
            'Sleep stage 3': 3,
            'Sleep stage 4': 3,  # On combine N3 et N4
            'Sleep stage R': 4,
            'Sleep stage ?': -1,
            'Movement time': -1
        }
        
        # Créer des epochs de 30 secondes
        events, _ = events_from_annotations(raw, event_id=event_id, chunk_duration=30)
        
        # Créer les epochs (rejette les segments avec artefacts)
        epochs = Epochs(raw, events, tmin=0, tmax=30, baseline=None, 
                       preload=True, reject_by_annotation=True)
        
        return raw, epochs, events[:, 2]
        
    except Exception as e:
        print(f"Erreur lors du chargement de {edf_path}: {str(e)}")
        raise

def process_single_psg_file(psg_path):
    """Traite un seul fichier PSG et retourne un DataFrame de résultats."""
    try:
        base_name = os.path.basename(psg_path).replace('-PSG.edf', '')
        hypno_path = psg_path.replace('-PSG.edf', '-Hypnogram.edf')
        
        if not os.path.exists(hypno_path):
            print(f"!!! Hypnogram manquant pour {base_name}, skip.")
            return pd.DataFrame()
            
        print(f"\n=== Traitement {base_name} ===")
        
        # Charger les données
        raw, epochs, true_labels = load_edf_with_annotations(psg_path, hypno_path)
        raw.filter(1., 40.)
        
        # Vérifier la cohérence
        if len(epochs) == 0:
            print(f"Aucune epoch valide pour {base_name}")
            return pd.DataFrame()
            
        # Extraire les données
        data = epochs.get_data()
        sf = raw.info['sfreq']
        
        # Préparer les arguments pour le traitement parallèle
        n_epochs, n_channels, _ = data.shape
        args_list = []
        
        for i in range(n_epochs):
            ch_ref = data[i, 0] if n_channels > 0 else None
            position_norm = i / n_epochs
            for j in range(n_channels):
                args_list.append((data[i, j], sf, bands, ch_ref, position_norm))
        
        # Traitement parallèle des canaux
        with ProcessPoolExecutor(max_workers=4) as executor:
            features_list = list(executor.map(process_channel_wrapper, args_list))
        
        # Mise en forme des features
        X = np.array(features_list).reshape(n_epochs, n_channels, -1)
        X = X.reshape(n_epochs, -1)
        
        # Filtrer les labels invalides
        valid_mask = (true_labels >= 0) & (true_labels < 5)
        X_valid = X[valid_mask]
        true_labels_valid = true_labels[valid_mask]
        
        if len(X_valid) == 0:
            print(f"Aucune donnée valide pour {base_name}")
            return pd.DataFrame()
            
        # Standardisation
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_valid)
        
        # Configurations à tester
        configs = [
            ("UMAP", umap.UMAP(n_components=2, random_state=42), "KMeans", KMeans(n_clusters=5, random_state=42, n_init=10)),
            ("PCA", PCA(n_components=0.95, random_state=42), "KMeans", KMeans(n_clusters=5, random_state=42, n_init=10)),
            (None, None, "KMeans", KMeans(n_clusters=5, random_state=42, n_init=10)),
            ("UMAP", umap.UMAP(n_components=2, random_state=42), "HDBSCAN", hdbscan.HDBSCAN(min_cluster_size=5)),
            ("PCA", PCA(n_components=0.95, random_state=42), "HDBSCAN", hdbscan.HDBSCAN(min_cluster_size=5)),
            (None, None, "HDBSCAN", hdbscan.HDBSCAN(min_cluster_size=5))
        ]
        
        # Évaluer chaque configuration
        results = []
        for reducer_name, reducer, cluster_name, clusterer in configs:
            try:
                # Réduction de dimension
                if reducer is not None:
                    X_red = reducer.fit_transform(X_scaled)
                else:
                    X_red = X_scaled
                
                # Clustering
                labels = clusterer.fit_predict(X_red)
                
                # Calcul des métriques
                if len(np.unique(labels[labels >= 0])) > 1:
                    sil = silhouette_score(X_red, labels) if len(np.unique(labels)) > 1 else np.nan
                    ch = calinski_harabasz_score(X_red, labels) if len(np.unique(labels)) > 1 else np.nan
                    db = davies_bouldin_score(X_red, labels) if len(np.unique(labels)) > 1 else np.nan
                else:
                    sil = ch = db = np.nan
                
                # Calcul de la pureté
                if len(np.unique(true_labels_valid)) > 1 and len(np.unique(labels)) > 1:
                    mask = labels != -1
                    if np.sum(mask) > 0:
                        df = pd.DataFrame({'true': true_labels_valid[mask], 'pred': labels[mask]})
                        cont_mat = pd.crosstab(df['true'], df['pred'])
                        purity = np.sum(np.max(cont_mat.values, axis=0)) / np.sum(cont_mat.values)
                    else:
                        purity = 0.0
                else:
                    purity = 0.0
                
                results.append({
                    'fichier': base_name,
                    'reducer': reducer_name or 'None',
                    'clusterer': cluster_name,
                    'n_clusters': len(np.unique(labels[labels >= 0])),
                    'silhouette': sil,
                    'calinski_harabasz': ch,
                    'davies_bouldin': db,
                    'purity': purity,
                    'n_samples': len(X_valid)
                })
                
            except Exception as e:
                print(f"Erreur avec {reducer_name or 'None'} + {cluster_name}: {str(e)}")
                continue
        
        return pd.DataFrame(results)
        
    except Exception as e:
        print(f"Erreur lors du traitement de {psg_path}: {str(e)}")
        return pd.DataFrame()

def process_channel_wrapper(args):
    """Wrapper pour le traitement parallèle des canaux."""
    return process_channel(*args)

def main():
    # Vérifier et créer le dossier de sortie
    os.makedirs(os.path.dirname(OUTPUT_CSV) or '.', exist_ok=True)
    
    # Lister les fichiers PSG
    psg_files = sorted(glob.glob(os.path.join(PSG_DIR, "*-PSG.edf")))
    print(f"Fichiers PSG trouvés: {len(psg_files)}")
    
    if not psg_files:
        print("Aucun fichier PSG trouvé.")
        return
    
    # Traitement parallèle des fichiers
    results = []
    cpu_count = max(1, multiprocessing.cpu_count() - 1)  # Laisser un coeur libre
    
    print(f"\nDémarrage du traitement parallèle sur {cpu_count} cœurs...")
    
    with ProcessPoolExecutor(max_workers=cpu_count) as executor:
        # Soumettre toutes les tâches
        futures = {executor.submit(process_single_psg_file, psg_file): psg_file 
                  for psg_file in psg_files}
        
        # Suivi de la progression
        for future in tqdm(as_completed(futures), total=len(futures), desc="Traitement des fichiers"):
            psg_file = futures[future]
            try:
                result = future.result()
                if not result.empty:
                    results.append(result)
            except Exception as e:
                print(f"Erreur lors du traitement de {psg_file}: {str(e)}")
    
    # Fusionner et sauvegarder les résultats
    if results:
        final_df = pd.concat(results, ignore_index=True)
        final_df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nRésultats sauvegardés dans {os.path.abspath(OUTPUT_CSV)}")
        print(f"Nombre total d'analyses: {len(final_df)}")
    else:
        print("Aucun résultat valide à sauvegarder.")

if __name__ == "__main__":
    main()
