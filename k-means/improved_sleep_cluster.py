# main_optimized_v3.py (ameliore avec plus de features, UMAP opti, KMeans conserve)

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
from scipy.stats import entropy, skew, kurtosis
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
    bands = np.array([(0.5,4), (4,8), (8,12), (12,15), (12,30), (30,50)], dtype=np.float32)

    def calculate_permutation_entropy(epoch, m=3, delay=1):
        try:
            pe = ant.perm_entropy(epoch, order=m, delay=delay, normalize=True)
            return pe
        except:
            return 0.0

    def hjorth_parameters(signal):
        first_deriv = np.diff(signal)
        second_deriv = np.diff(first_deriv)

        var_zero = np.var(signal)
        var_d1 = np.var(first_deriv)
        var_d2 = np.var(second_deriv)

        activity = var_zero
        mobility = np.sqrt(var_d1 / var_zero) if var_zero != 0 else 0.0
        complexity = np.sqrt(var_d2 / var_d1) / mobility if var_d1 != 0 and mobility != 0 else 0.0

        return activity, mobility, complexity

    def compute_coherence(ch1, ch2, sf, band):
        from scipy.signal import coherence
        f, Cxy = coherence(ch1, ch2, sf, nperseg=256)
        idx_band = (f >= band[0]) & (f <= band[1])
        return np.mean(Cxy[idx_band]) if np.any(idx_band) else 0.0

    def process_channel(ch, sf, bands, ch_other=None, position_norm=None):
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
        try:
            pe = ant.perm_entropy(ch, order=3, delay=1, normalize=True)
        except:
            pe = 0.0
        features.append(pe)

        # Entropie spectrale
        psd_norm = psd / (np.sum(psd) + 1e-12)
        features.append(entropy(psd_norm + 1e-12))

        # Bandpower
        freqs = np.fft.rfftfreq(min(256, len(ch)), 1.0/sf)
        for fmin, fmax in bands:
            idx_band = (freqs >= fmin) & (freqs <= fmax)
            features.append(np.mean(psd[idx_band]) if np.any(idx_band) else 0.0)

        # Statistiques temporelles
        features.append(np.mean(ch))
        features.append(np.std(ch))
        features.append(skew(ch))
        features.append(kurtosis(ch))

        # Higuchi fractal dimension optimisée (kmax réduit)
        try:
            hfd = ant.higuchi_fd(ch, kmax=6)
            features.append(hfd)
        except:
            features.append(0.0)

        # Detrended fluctuation analysis (DFA)
        try:
            dfa = ant.detrended_fluctuation(ch)
            features.append(dfa)
        except:
            features.append(0.0)

        # Approximate Entropy optimisée
        try:
            apen = ant.app_entropy(ch, m=2, r=0.2 * np.std(ch))
            features.append(apen)
        except:
            features.append(0.0)

        # Sample Entropy optimisée
        try:
            sampen = ant.sample_entropy(ch, order=2)
            features.append(sampen)
        except:
            features.append(0.0)

        # Hjorth parameters
        try:
            activity, mobility, complexity = hjorth_parameters(ch)
            features.extend([activity, mobility, complexity])
        except:
            features.extend([0.0, 0.0, 0.0])

        # Coherence avec ch_other (si fourni)
        if ch_other is not None:
            for fmin, fmax in bands:
                try:
                    coh = compute_coherence(ch, ch_other, sf, (fmin, fmax))
                    features.append(coh)
                except:
                    features.append(0.0)

        # Ajout position normalisée dans la nuit (si fournie)
        if position_norm is not None:
            features.append(position_norm)
        else:
            features.append(0.0)

        return features

    def process_channel_wrapper(args):
        if USE_GPU:
            return process_channel_gpu(*args)
        else:
            return process_channel(*args)

    def process_channel_gpu(ch, sf, bands, ch_other=None, position_norm=None):
        features = []

        # Conversion du signal en GPU array
        ch_gpu = cp.asarray(ch.astype(np.float32))

        # Welch GPU avec cupyx
        try:
            window = cp.asarray(get_window('hann', min(256, len(ch))))
            _, psd_gpu = cupyx.scipy.signal.welch(ch_gpu, sf, window=window, nperseg=min(256, len(ch)))
        except:
            # Fallback sans fenêtre si problème
            _, psd_gpu = cupyx.scipy.signal.welch(ch_gpu, sf, nperseg=min(256, len(ch)))

        # Entropie permutation (CPU)
        try:
            pe = ant.perm_entropy(ch, order=3, delay=1, normalize=True)
        except:
            pe = 0.0
        features.append(pe)

        # Entropie spectrale (GPU + conversion CPU pour entropy)
        psd_norm_gpu = psd_gpu / (cp.sum(psd_gpu) + 1e-12)
        psd_norm_cpu = cp.asnumpy(psd_norm_gpu)
        features.append(entropy(psd_norm_cpu + 1e-12))

        # Bandpower (GPU)
        freqs_gpu = cp.fft.rfftfreq(min(256, len(ch)), 1.0/sf)
        for fmin, fmax in bands:
            idx_band = (freqs_gpu >= fmin) & (freqs_gpu <= fmax)
            features.append(cp.mean(psd_gpu[idx_band]).get() if cp.any(idx_band) else 0.0)

        # Statistiques temporelles (GPU → cupy puis get vers CPU)
        features.append(cp.mean(ch_gpu).get())
        features.append(cp.std(ch_gpu).get())
        features.append(skew(cp.asnumpy(ch_gpu)))
        features.append(kurtosis(cp.asnumpy(ch_gpu)))

        # Higuchi fractal dimension (CPU)
        try:
            hfd = ant.higuchi_fd(ch, kmax=6)
            features.append(hfd)
        except:
            features.append(0.0)

        # Detrended fluctuation analysis (DFA) (CPU)
        try:
            dfa = ant.detrended_fluctuation(ch)
            features.append(dfa)
        except:
            features.append(0.0)

        # Approximate Entropy (CPU)
        try:
            apen = ant.app_entropy(ch, m=2, r=0.2 * np.std(ch))
            features.append(apen)
        except:
            features.append(0.0)

        # Sample Entropy (CPU)
        try:
            sampen = ant.sample_entropy(ch, order=2)
            features.append(sampen)
        except:
            features.append(0.0)

        # Hjorth parameters (GPU + CPU pour la formule finale)
        first_deriv = cp.diff(ch_gpu)
        second_deriv = cp.diff(first_deriv)

        var_zero = cp.var(ch_gpu).get()
        var_d1 = cp.var(first_deriv).get()
        var_d2 = cp.var(second_deriv).get()

        activity = var_zero
        mobility = np.sqrt(var_d1 / var_zero) if var_zero != 0 else 0.0
        complexity = np.sqrt(var_d2 / var_d1) / mobility if var_d1 != 0 and mobility != 0 else 0.0

        features.extend([activity, mobility, complexity])

        # Coherence avec ch_other (GPU avec cupyx)
        if ch_other is not None:
            ch_other_gpu = cp.asarray(ch_other.astype(np.float32))
            for fmin, fmax in bands:
                try:
                    f, Cxy = cupyx.scipy.signal.coherence(ch_gpu, ch_other_gpu, fs=sf, nperseg=256)
                    idx_band = (f >= fmin) & (f <= fmax)
                    features.append(cp.mean(Cxy[idx_band]).get() if cp.any(idx_band) else 0.0)
                except Exception as e:
                    print(f"Erreur calcul cohérence: {str(e)}")
                    features.append(0.0)

        # Ajout position normalisée dans la nuit
        if position_norm is not None:
            features.append(position_norm)
        else:
            features.append(0.0)

        return features

    # Extraction des features pour chaque canal
    n_channels = data.shape[1]
    features_list = []
    for ch in range(n_channels):
        features = process_channel_wrapper((data[:, ch, :], sf, bands))
        features_list.append(features)

    # Réduction de dimension avec UMAP
    if USE_GPU:
        reducer = cuUMAP(n_components=2, random_state=42, output_type='numpy')
        X_embedded = reducer.fit_transform(features_list)
    else:
        reducer = umap.UMAP(n_components=2, random_state=42)
        X_embedded = reducer.fit_transform(features_list)

    # Clustering avec KMeans
    n_clusters_kmeans = 5
    if USE_GPU:
        print("KMeans GPU optimisé activé")
        X_embedded_gpu = cp.asarray(X_embedded.astype(np.float32))
        kmeans = cuKMeans(n_clusters=n_clusters_kmeans, random_state=42, output_type='numpy')
        kmeans.fit(X_embedded_gpu)
        labels_pred = kmeans.labels_
        del X_embedded_gpu
        cp.get_default_memory_pool().free_all_blocks()
    else:
        print("KMeans CPU (sklearn) activé")
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters_kmeans, random_state=42)
        labels_pred = kmeans.fit_predict(X_embedded)

    # Affichage des résultats
    if len(labels_pred) > 1:
        try:
            annotations = mne.read_annotations('SC4001EC-Hypno.edf')
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
                    plt.title('Matrice de confusion: vrais stades vs clusters (epoques non-Unknown)')
                    plt.tight_layout()
                    plt.show()

                    print("Rapport de classification :")
                    print(classification_report(labels_true_filtered, labels_pred_filtered, target_names=filtered_display_labels, zero_division=0))
                else:
                    print("Pas assez de classes différentes dans les annotations ou mismatch.")
            print("\nRépartition des clusters :")
            for cluster, count in zip(unique, counts):
                print(f"Cluster {cluster}: {count} époques ({count/len(labels_pred):.1%})")

            # Calcul des métriques d'évaluation si les vrais labels sont disponibles
            if 'labels_true_filtered' in locals() and len(labels_true_filtered) > 1:
                try:
                    from sklearn.metrics import silhouette_score
                    
                    # Fonction pour calculer la pureté
                    def purity_score(y_true, y_pred):
                        contingency_matrix = confusion_matrix(y_true, y_pred)
                        return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
                    
                    # Silhouette score sur X_embedded
                    sil_score = silhouette_score(X_embedded[valid_indices], labels_pred_filtered)
                    print(f"\nSilhouette Score : {sil_score:.3f}")

                    # Purity score
                    pur_score = purity_score(labels_true_filtered, labels_pred_filtered)
                    print(f"Purity Score : {pur_score:.3f}")

                except Exception as e:
                    print(f"Erreur lors du calcul des scores de qualité : {e}")
            else:
                print("\nPas de labels vrais suffisants pour évaluer la qualité des clusters.")

        except FileNotFoundError:
        print("Fichier d'hypnogramme non trouvé.")
    except Exception as e:
        print(f"Une erreur est survenue pendant la lecture des annotations : {e}")
else:
    print("Pas assez d'échantillons pour KMeans.")
