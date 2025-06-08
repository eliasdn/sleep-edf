import os
import glob
import warnings
import numpy as np
import pandas as pd
import mne
import umap.umap_ as umap
import concurrent.futures
import psutil
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans as skKMeans
import hdbscan
from antropy import app_entropy, sample_entropy, petrosian_fd, katz_fd, detrended_fluctuation
from scipy import stats, signal
import antropy as ant
import matplotlib.pyplot as plt
from mne import Epochs, events_from_annotations
from mne.io import read_raw_edf
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configuration globale
USE_GPU = False  # Mettre à True pour utiliser le GPU si disponible

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
    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
    sfreq = raw.info['sfreq']
    n_samples = raw.n_times
    n_secs = n_samples / sfreq
    n_epochs = int(np.floor(n_secs / epoch_length))

    # Charger les annotations de l'hypnogramme
    raw_annot = mne.io.read_raw_edf(hypnogram_path, preload=True, verbose=False)
    annotations = raw_annot.annotations
    
    # Mapping des descriptions vers labels numériques
    stage_mapping = {
        'Sleep stage W': 0,
        'Sleep stage 1': 1,
        'Sleep stage 2': 2,
        'Sleep stage 3': 3,
        'Sleep stage 4': 3,  # Sleep stage 3 et 4 fusionnés (convention AASM)
        'Sleep stage R': 4,
        'Sleep stage ?': -1,
        'Movement time': -1
    }
    
    # Initialiser les labels à -1
    true_labels = np.full(n_epochs, fill_value=-1, dtype=int)
    
    # Remplir true_labels selon les annotations
    for onset, duration, desc in zip(annotations.onset, annotations.duration, annotations.description):
        start_epoch = int(onset // epoch_length)
        n_epochs_this_stage = int(np.ceil(duration / epoch_length))
        stage_label = stage_mapping.get(desc, -1)
        
        if stage_label != -1:
            end_epoch = start_epoch + n_epochs_this_stage
            # Sécuriser l'indexation pour ne pas dépasser n_epochs
            true_labels[start_epoch:end_epoch] = stage_label
    
    # Créer les epochs fixes
    epochs = mne.make_fixed_length_epochs(raw, duration=epoch_length, preload=True)
    
    # Vérification de cohérence
    n_epochs_final = len(epochs)
    if n_epochs_final != len(true_labels):
        print(f"⚠️ Mismatch epochs ({n_epochs_final}) vs labels ({len(true_labels)}), ajustement...")
        min_len = min(n_epochs_final, len(true_labels))
        true_labels = true_labels[:min_len]
        epochs = epochs[:min_len]
    
    print(f"✅ Données chargées: {len(epochs)} époques, {len(epochs.info['ch_names'])} canaux")
    return epochs, true_labels


def process_single_psg_file(psg_path):
    """Traite un seul fichier PSG et retourne un DataFrame de résultats."""
    results = []
    psg_name = os.path.basename(psg_path)
    
    try:
        print(f"\nTraitement du fichier: {psg_name}")
        
        # 1. Chercher le fichier hypnogram correspondant
        base_name = os.path.basename(psg_path).replace('-PSG.edf', '')
        hypno_dir = os.path.dirname(psg_path)
            
        hypno_path = find_hypnogram(psg_path)
        
        # 2. Charger les données EDF et les annotations
        try:
            epochs, true_labels = load_edf_with_annotations(psg_path, hypno_path)
            raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False) 
            print(f"  - Données chargées: {len(epochs)} époques, {len(epochs.ch_names)} canaux")
        except Exception as e:
            print(f"  ❌ Erreur lors du chargement des données: {str(e)}")
            return pd.DataFrame()
            
        # 3. Vérifier les données
        if epochs is None or len(epochs) == 0:
            print("  ❌ Aucune donnée valide après chargement")
            return pd.DataFrame()
            
        # 3. Prétraitement
        try:
            print("  - Filtrage du signal...")
            raw.filter(1., 40.)
            data = epochs.get_data()
            sf = raw.info['sfreq']
            print(f"  - Signal filtré: {sf} Hz")
        except Exception as e:
            print(f"  ❌ Erreur lors du prétraitement: {str(e)}")
            return pd.DataFrame()
        
        # 5. Traitement des canaux
        n_epochs, n_channels, n_times = data.shape
        print(f"  - Extraction des caractéristiques pour {n_epochs} époques x {n_channels} canaux...")
        
        # 6. Préparer les arguments pour le traitement parallèle
        args_list = []
        for i in range(n_epochs):
            ch_ref = data[i, 0] if n_channels > 0 else None
            position_norm = i / n_epochs
            
            for j in range(n_channels):
                args_list.append((data[i, j], sf, bands, ch_ref, position_norm))
        
        # 7. Traitement parallèle des canaux
        features_list = []
        num_workers = min(4, psutil.cpu_count(logical=False))
        
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                features_list = list(tqdm(
                    executor.map(process_channel_wrapper, args_list),
                    total=len(args_list),
                    desc="  - Extraction des features",
                    leave=False,
                    mininterval=5.0  # Mettre à jour la barre de progression toutes les 5 secondes
                ))
            
            if not features_list:
                print("  ❌ Aucune feature extraite")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"  ❌ Erreur lors de l'extraction des features: {str(e)}")
            return pd.DataFrame()
            
        # 8. Préparation des données pour le clustering
        try:
            print("  - Préparation des données pour le clustering...")
            X = np.array(features_list)
            
            # Vérifier les NaN/Inf
            if np.isnan(X).any() or np.isinf(X).any():
                print("  - Remplacement des valeurs NaN/Inf...")
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Normalisation
            print("  - Normalisation des données...")
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Remodeler pour avoir (n_epochs, n_channels * n_features)
            X_reshaped = X_scaled.reshape(n_epochs, -1)
            
            # Vérifier la dimension finale
            if X_reshaped.shape[1] == 0:
                print("  ❌ Aucune feature valide après préparation")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"  ❌ Erreur lors de la préparation des données: {str(e)}")
            return pd.DataFrame()
        
        # 9. Benchmarking des méthodes de réduction de dimension et de clustering
        print("  - Démarrage du benchmark...")
        
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
                print(f"\n  - Test de {reducer_name}...")
                
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
                        print(f"    ❌ Erreur avec {reducer_name}: {str(e)}")
                        continue
                
                nb_clusters_with_valid_labels = 0
                nb_clusters_without_valid_labels = 0
                # Boucle sur les méthodes de clustering
                for clusterer_name, clusterer_factory in clusterers:
                    try:
                        print(f"    - Test de {clusterer_name}...", end=' ')
                        
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
                            'psg_file': psg_name,
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
                                print(f"⚠️ Erreur métriques: {str(e)}")
                        
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
                                            else:
                                                cluster_to_label[cluster] = -1


                                            # Logging détaillé
                                            label_counts = np.bincount(valid_true_in_cluster)
                                            print(f"    📊 Cluster {cluster}: {total_in_cluster} samples (valid: {valid_count}), "
                                                f"majority label = {majority_label} ({label_counts[majority_label]} samples)")
                                            nb_clusters_with_valid_labels += 1

                                        except Exception as e:
                                            print(f"⚠️ Erreur lors du calcul du label majoritaire pour le cluster {cluster}: {e}")
                                            cluster_to_label[cluster] = -1
                                    else:
                                        print(f"ℹ️ Cluster {cluster}: {total_in_cluster} samples (aucun label valide — ignoré dans l'évaluation)")
                                        cluster_to_label[cluster] = -1
                                        nb_clusters_without_valid_labels += 1


                                # Appliquer le mapping pour obtenir les labels prédits
                                predicted_labels = np.array([cluster_to_label.get(cluster, -1) for cluster in labels])
                                 
                                # Calculer la pureté
                                valid_mask = (predicted_labels >= 0) & (true_labels >= 0)
                                if np.any(valid_mask):
                                    purity = np.mean(predicted_labels[valid_mask] == true_labels[valid_mask])
                                    metrics['purity'] = purity
                                    
                                    # Matrice de confusion
                                    cm = confusion_matrix(true_labels[valid_mask], predicted_labels[valid_mask])
                                    metrics['confusion_matrix'] = str(cm.tolist())
                                    
                            except Exception as e:
                                print(f"⚠️ Erreur calcul métriques annotations: {str(e)}")
                        
                        results.append(metrics)
                        print("✓")
                        
                    except Exception as e:
                        print(f"❌ Erreur avec {clusterer_name}: {str(e)}")
                        continue
                        
            except Exception as e:
                print(f"  ❌ Erreur majeure avec {reducer_name}: {str(e)}")
                continue
        
        return pd.DataFrame(results)
        
    except Exception as e:
        print(f"\n❌ ERREUR CRITIQUE lors du traitement de {psg_name}: {str(e)}")
        import traceback
        traceback.print_exc()
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
    cpu_count = max(1, multiprocessing.cpu_count())  # Laisser un coeur libre
    
    print(f"\nDémarrage du traitement parallèle sur {cpu_count} cœurs...")
    
    with ProcessPoolExecutor(max_workers=cpu_count) as executor:
        # Soumettre toutes les tâches
        futures = {executor.submit(process_single_psg_file, psg_file): psg_file 
                  for psg_file in psg_files}
        
        # Suivi de la progression
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), 
                          desc="Traitement des fichiers"):
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
