# main_optimized.py

# Configuration globale
USE_GPU = True  # Mettre False pour repasser en CPU

import mne
import numpy as np
import cupy as cp
import cupyx.scipy.signal
from cuml.cluster import KMeans as cuKMeans
from cuml.manifold import UMAP as cuUMAP
import antropy as ant
import umap
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from scipy.stats import entropy
from scipy.signal import welch, get_window
import concurrent.futures
import psutil
from tqdm import tqdm

# Charger les fichiers EEG et Hypnogramme
try:
    raw = mne.io.read_raw_edf('SC4001E0-PSG.edf', preload=True)
    raw.filter(1., 40.)

    epochs = mne.make_fixed_length_epochs(raw, duration=5.0, preload=True)
    data = epochs.get_data()

    sf = raw.info['sfreq']
    bands = np.array([(0.5,4), (4,8), (8,12), (12,30)], dtype=np.float32)

    def calculate_permutation_entropy(epoch, m=3, delay=1):
        try:
            pe = ant.perm_entropy(epoch, order=m, delay=delay, normalize=True)
            return pe
        except:
            return 0.0

    def process_channel(ch, sf, bands):
        features = []

        # Welch GPU ou CPU
        if USE_GPU:
            try:
                ch_gpu = cp.asarray(ch.astype(np.float32))
                _, psd_gpu = cupyx.scipy.signal.welch(ch_gpu, sf, nperseg=min(256, len(ch)))
                psd = cp.asnumpy(psd_gpu)
            except:
                window = get_window('hann', min(256, len(ch)))
                _, psd = welch(ch, sf, window=window, nperseg=min(256, len(ch)), return_onesided=True)
        else:
            window = get_window('hann', min(256, len(ch)))
            _, psd = welch(ch, sf, window=window, nperseg=min(256, len(ch)), return_onesided=True)

        # Entropie permutation
        pe = calculate_permutation_entropy(ch, m=3, delay=1)
        features.append(pe)

        # Entropie spectrale
        psd_norm = psd / (np.sum(psd) + 1e-12)
        features.append(entropy(psd_norm + 1e-12))

        # Bandpower
        freqs = np.fft.rfftfreq(min(256, len(ch)), 1.0/sf)
        for fmin, fmax in bands:
            idx_band = (freqs >= fmin) & (freqs <= fmax)
            features.append(np.mean(psd[idx_band]) if np.any(idx_band) else 0.0)

        return features

    num_cores = psutil.cpu_count(logical=False)
    num_workers = min(4, num_cores)

    print(f"Traitement de {len(data)} époques avec {num_workers} workers...")

    X = []
    batch_size = 1000
    tasks = []
    batch = []

    for i, epoch in enumerate(data):
        for j, ch in enumerate(epoch):
            batch.append((ch, sf, bands))
            if len(batch) >= batch_size:
                tasks.append(batch)
                batch = []
    if batch:
        tasks.append(batch)

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for batch in tqdm(tasks, desc="Traitement des lots"):
            try:
                batch_results = list(executor.map(
                    lambda x: process_channel(*x),
                    batch,
                    chunksize=max(1, len(batch) // (num_workers * 2))
                ))
                results.extend(batch_results)
            except Exception as e:
                print(f"Erreur lors du traitement d'un lot: {str(e)}")
                for task in batch:
                    try:
                        results.append(process_channel(*task))
                    except:
                        results.append([0.0] * (len(bands) + 2))

    n_channels = len(data[0]) if len(data) > 0 else 0
    if n_channels > 0:
        X = [results[i:i + n_channels] for i in range(0, len(results), n_channels)]
        X = np.array(X, dtype=np.float32).reshape(len(data), -1)
    else:
        X = np.array([], dtype=np.float32)

    X = np.ascontiguousarray(X, dtype=np.float32)
    print(f"Traitement terminé. Données finales: {X.shape}")

    if X.shape[0] > 0:
        n_neighbors_umap = min(15, X.shape[0]-1) if X.shape[0] > 1 else 1
        n_neighbors_umap = max(1, n_neighbors_umap) if X.shape[0] > 1 else 1

        if USE_GPU:
            print("UMAP GPU activé (cuML)")
            X_gpu = cp.asarray(X.astype(np.float32))
            X_embedded = cuUMAP(n_neighbors=n_neighbors_umap, min_dist=0.1).fit_transform(X_gpu)
            X_embedded = cp.asnumpy(X_embedded)
        else:
            print("UMAP CPU activé (UMAP sklearn)")
            X_embedded = umap.UMAP(n_neighbors=n_neighbors_umap, min_dist=0.1).fit_transform(X)

        n_clusters = 5
        n_clusters_kmeans = min(n_clusters, X_embedded.shape[0])
        n_clusters_kmeans = max(1, n_clusters_kmeans)

        if n_clusters_kmeans > 0:
            X_embedded_gpu = cp.asarray(X_embedded.astype(np.float32))
            kmeans = cuKMeans(n_clusters=n_clusters_kmeans, random_state=42, output_type='numpy')
            kmeans.fit(X_embedded_gpu)
            labels_pred = kmeans.labels_
            del X_embedded_gpu
            cp.get_default_memory_pool().free_all_blocks()

            try:
                annotations = mne.read_annotations('SC4001EC-Hypnogram.edf')
                labels_true_str = []
                if len(epochs.events) == len(labels_pred):
                    annotation_onsets = annotations.onset
                    annotation_durations = annotations.duration
                    annotation_descriptions = annotations.description

                    for tmin in epochs.events[:, 0] / raw.info['sfreq']:
                        label = None
                        for onset, duration, description in zip(annotation_onsets, annotation_durations, annotation_descriptions):
                            if onset <= tmin < onset + duration:
                                label = description
                                break
                        labels_true_str.append(label if label else 'Unknown')

                    unique_true_labels = sorted(list(set(labels_true_str)))
                    if 'Unknown' in unique_true_labels and len(unique_true_labels) > 1:
                        unique_true_labels.remove('Unknown')

                    label_mapping = {label: i for i, label in enumerate(unique_true_labels)}
                    if 'Unknown' in set(labels_true_str):
                        if 'Unknown' not in unique_true_labels:
                            label_mapping['Unknown'] = len(unique_true_labels)
                            unique_true_labels.append('Unknown')

                    labels_true_num = [label_mapping.get(label, -1) for label in labels_true_str]
                    valid_indices = [i for i, label in enumerate(labels_true_str) if label != 'Unknown']
                    if len(valid_indices) > 0:
                        labels_true_filtered = [labels_true_num[i] for i in valid_indices]
                        labels_pred_filtered = [labels_pred[i] for i in valid_indices]

                        if len(labels_true_filtered) > 1 and len(labels_true_filtered) == len(labels_pred_filtered):
                            unique_filtered_true_nums = sorted(list(set(labels_true_filtered)))
                            filtered_display_labels = [
                                list(label_mapping.keys())[list(label_mapping.values()).index(num)]
                                for num in unique_filtered_true_nums
                            ]

                            cm = confusion_matrix(labels_true_filtered, labels_pred_filtered)
                            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=filtered_display_labels)
                            disp.plot(xticks_rotation=45)
                            plt.title('Matrice de confusion: vrais stades vs clusters (GPU)')
                            plt.tight_layout()
                            plt.show()

                            print("\nRapport de classification :\n")
                            print(classification_report(labels_true_filtered, labels_pred_filtered, target_names=filtered_display_labels, zero_division=0))

                        else:
                            print("Pas assez de classes différentes dans les annotations ou mismatch.")
                    else:
                        print("Aucune époque avec un vrai stade connu pour la matrice de confusion.")
                else:
                    print(f"Mismatch epochs ({len(epochs.events)}) vs predictions ({len(labels_pred)}).")
            except FileNotFoundError:
                print("Fichier d'hypnogramme non trouvé.")

            plt.figure(figsize=(12, 8))
            scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels_pred, cmap='viridis', alpha=0.6, s=50)
            plt.colorbar(scatter, label='Cluster')
            plt.title('Visualisation UMAP des clusters de stades de sommeil (GPU)')
            plt.xlabel('UMAP 1')
            plt.ylabel('UMAP 2')
            plt.tight_layout()
            plt.show()

            unique, counts = np.unique(labels_pred, return_counts=True)
            print("\nRépartition des clusters :")
            for cluster, count in zip(unique, counts):
                print(f"Cluster {cluster}: {count} époques ({count/len(labels_pred):.1%})")

        else:
            print("Pas assez d'échantillons pour KMeans.")

    else:
        print("Aucune donnée extraite pour la réduction de dimension.")

except FileNotFoundError:
    print("Erreur : Fichier EEG non trouvé.")
except Exception as e:
    print(f"Une erreur est survenue : {e}")
    raise
