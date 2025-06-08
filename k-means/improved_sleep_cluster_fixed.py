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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans as skKMeans  # Pour le CPU
import hdbscan
from scipy.stats import entropy, skew, kurtosis
from scipy.signal import welch, get_window, coherence
import concurrent.futures
import psutil
from tqdm import tqdm

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
        features.append(pe)
    except Exception as e:
        print(f"[Warning] Erreur calcul entropie permutation: {str(e)[:100]}")
        features.append(0.0)

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
    except Exception as e:
        print(f"[Warning] Erreur calcul dimension fractale de Higuchi: {str(e)[:100]}")
        features.append(0.0)

    # Detrended fluctuation analysis (DFA)
    try:
        dfa = ant.detrended_fluctuation(ch)
        features.append(dfa)
    except Exception as e:
        print(f"[Warning] Erreur calcul DFA: {str(e)[:100]}")
        features.append(0.0)


    # Approximate Entropy optimisée
    try:
        # Vérification de la signature de app_entropy
        # Essayons d'abord sans paramètres
        apen = ant.app_entropy(ch)
        features.append(apen)
    except Exception as e:
        try:
            # Si ça échoue, essayons avec les paramètres par défaut de la documentation
            apen = ant.app_entropy(ch, order=2, metric='chebyshev')
            features.append(apen)
        except Exception as e2:
            print(f"[Warning] Erreur calcul entropie approximative: {str(e2)[:100]}")
            features.append(0.0)


    # Sample Entropy optimisée
    try:
        sampen = ant.sample_entropy(ch, order=2)
        features.append(sampen)
    except Exception as e:
        print(f"[Warning] Erreur calcul entropie d'échantillon: {str(e)[:100]}")
        features.append(0.0)


    # Hjorth parameters
    try:
        activity, mobility, complexity = hjorth_parameters(ch)
        features.extend([activity, mobility, complexity])
    except Exception as e:
        print(f"[Warning] Erreur calcul paramètres de Hjorth: {str(e)[:100]}")
        features.extend([0.0, 0.0, 0.0])

    # Coherence avec ch_other (si fourni)
    if ch_other is not None:
        for fmin, fmax in bands:
            try:
                coh = compute_coherence(ch, ch_other, sf, (fmin, fmax))
                features.append(coh)
            except Exception as e:
                print(f"[Warning] Erreur calcul cohérence ({fmin}-{fmax}Hz): {str(e)[:100]}")
                features.append(0.0)

    # Ajout position normalisée dans la nuit (si fournie)
    if position_norm is not None:
        features.append(position_norm)
    else:
        features.append(0.0)


    return features

def process_channel_wrapper(args):
    return process_channel(*args)

def purity_score(y_true, y_pred):
    contingency_matrix = confusion_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

def load_edf_with_annotations(edf_path, annotation_path):
    """Charge un fichier EDF et ses annotations, et retourne les données et les vrais labels."""
    raw = mne.io.read_raw_edf(edf_path, preload=True)
    
    # Charger les annotations
    annot = mne.read_annotations(annotation_path)
    raw.set_annotations(annot)
    
    # Extraire les événements à partir des annotations
    events, event_id = mne.events_from_annotations(raw)
    
    # Créer des epochs basées sur les annotations (30 secondes par époque)
    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=0, tmax=30, baseline=None, preload=True)
    
    # Mapper les labels d'annotation aux indices de classe (0-4)
    label_mapping = {
        'Sleep stage W': 0,
        'Sleep stage 1': 1,
        'Sleep stage 2': 2,
        'Sleep stage 3': 3,
        'Sleep stage 4': 3,  # Combiner N3 et N4
        'Sleep stage R': 4,   # REM
        'Sleep stage ?': -1,  # Inconnu
        'Movement time': -1   # Ignorer
    }
    
    # Extraire et mapper les labels des annotations
    true_labels = []
    for event in events:
        # Trouver le label correspondant à l'ID de l'événement
        label_name = [k for k, v in event_id.items() if v == event[2]]
        if label_name:
            label_name = label_name[0]
            mapped_label = label_mapping.get(label_name, -1)
            if mapped_label != -1:  # Ignorer les labels non mappés
                true_labels.append(mapped_label)
    
    return raw, epochs, np.array(true_labels)

def main():
    try:
        # Charger les données EEG avec les annotations
        raw, epochs, true_labels = load_edf_with_annotations('SC4001E0-PSG.edf', 'SC4001EC-Hypnogram.edf')
        raw.filter(1., 40.)
        data = epochs.get_data()
        sf = raw.info['sfreq']
        
        # Stocker les vrais labels comme variable globale de la fonction
        main.true_labels = true_labels
        has_annotations = len(true_labels) > 0 and len(true_labels) == len(epochs)
        if has_annotations:
            print(f"\n{len(true_labels)} annotations chargées avec les distributions suivantes :")
            unique, counts = np.unique(true_labels, return_counts=True)
            stage_names = ['W', 'N1', 'N2', 'N3', 'REM']
            for label, count in zip(unique, counts):
                print(f"  - {stage_names[label]}: {count} époques ({count/len(true_labels):.1%})")
        bands = np.array([(0.5,4), (4,8), (8,12), (12,15), (12,30), (30,50)], dtype=np.float32)

        # Traitement des canaux
        n_epochs, n_channels, n_times = data.shape
        print(f"Traitement de {n_epochs} époques avec {n_channels} canaux...")

        # Préparer les données pour le traitement parallèle
        args_list = []
        for i in range(n_epochs):
            # Utiliser le premier canal comme référence pour la cohérence
            ch_ref = data[i, 0] if n_channels > 0 else None
            position_norm = i / n_epochs
            
            for j in range(n_channels):
                args_list.append((data[i, j], sf, bands, ch_ref, position_norm))

        # Traitement parallèle des canaux
        features_list = []
        num_cores = psutil.cpu_count(logical=False)
        num_workers = min(4, num_cores)
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            features_list = list(tqdm(executor.map(process_channel_wrapper, args_list), 
                                  total=len(args_list), 
                                  desc="Traitement des canaux"))

        # Remodeler les caractéristiques
        X = np.array(features_list).reshape(n_epochs, n_channels, -1)
        X = X.reshape(n_epochs, -1)  # Aplatir pour UMAP

        # Réduction de dimension avec UMAP
        print("\nRéduction de dimension avec UMAP...")
        if USE_GPU:
            reducer = cuUMAP(n_components=2, random_state=42, output_type='numpy')
            X_embedded = reducer.fit_transform(X.astype(np.float32))
        else:
            reducer = umap.UMAP(n_components=2, random_state=42)
            X_embedded = reducer.fit_transform(X)

        # Détermination du nombre optimal de clusters
        def find_optimal_clusters(X, max_k=10):
            """Trouve le nombre optimal de clusters en utilisant différentes métriques."""
            from sklearn.cluster import KMeans
            
            range_k = range(2, max_k + 1)
            sil_scores = []
            ch_scores = []
            db_scores = []
            
            print("\nRecherche du nombre optimal de clusters...")
            for k in tqdm(range_k, desc="Test des valeurs de k"):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels_pred = kmeans.fit_predict(X)
                
                # Calculer les métriques
                sil_scores.append(silhouette_score(X, labels_pred))
                ch_scores.append(calinski_harabasz_score(X, labels_pred))
                db_scores.append(davies_bouldin_score(X, labels_pred))
            
            # Tracer les courbes des métriques
            plt.figure(figsize=(15, 5))
            
            # Score de silhouette (plus haut = mieux)
            plt.subplot(1, 3, 1)
            plt.plot(range_k, sil_scores, 'bo-')
            plt.xlabel('Nombre de clusters (k)')
            plt.ylabel('Score de silhouette')
            plt.title('Score de silhouette par k')
            
            # Score de Calinski-Harabasz (plus haut = mieux)
            plt.subplot(1, 3, 2)
            plt.plot(range_k, ch_scores, 'go-')
            plt.xlabel('Nombre de clusters (k)')
            plt.ylabel('Score Calinski-Harabasz')
            plt.title('Score Calinski-Harabasz par k')
            
            # Score de Davies-Bouldin (plus bas = mieux)
            plt.subplot(1, 3, 3)
            plt.plot(range_k, db_scores, 'ro-')
            plt.xlabel('Nombre de clusters (k)')
            plt.ylabel('Score Davies-Bouldin')
            plt.title('Score Davies-Bouldin par k')
            
            plt.tight_layout()
            plt.show()
            
            # Trouver le k optimal basé sur le coude des courbes
            # Ici on utilise simplement la moyenne des métriques normalisées
            sil_norm = (sil_scores - np.min(sil_scores)) / (np.max(sil_scores) - np.min(sil_scores) + 1e-10)
            ch_norm = (ch_scores - np.min(ch_scores)) / (np.max(ch_scores) - np.min(ch_scores) + 1e-10)
            db_norm = 1 - ((db_scores - np.min(db_scores)) / (np.max(db_scores) - np.min(db_scores) + 1e-10))
            
            combined_scores = (sil_norm + ch_norm + db_norm) / 3
            optimal_k = range_k[np.argmax(combined_scores)]
            
            print(f"\nNombre optimal de clusters suggéré : {optimal_k}")
            return optimal_k
        
        # Exécuter la recherche du nombre optimal de clusters
        optimal_k = find_optimal_clusters(X_embedded, max_k=10)
        
        # Utiliser le k optimal pour le clustering final
        n_clusters = optimal_k
        print(f"\nClustering avec KMeans (k={n_clusters})...")
        if USE_GPU:
            print("Utilisation du GPU (cuML)")
            X_embedded_gpu = cp.asarray(X_embedded.astype(np.float32))
            kmeans = cuKMeans(n_clusters=n_clusters, random_state=42, output_type='numpy')
            kmeans.fit(X_embedded_gpu)
            labels_pred = kmeans.labels_
            del X_embedded_gpu
            cp.get_default_memory_pool().free_all_blocks()
        else:
            print("Utilisation du CPU (scikit-learn)")
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels_pred = kmeans.fit_predict(X_embedded)

        # Visualisation
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels_pred, 
                            cmap='viridis', alpha=0.6, s=50)
        plt.colorbar(scatter, label='Cluster')
        plt.title(f'Visualisation UMAP des clusters de stades de sommeil ({"GPU" if USE_GPU else "CPU"})')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.tight_layout()
        plt.show()

        # Calcul et affichage des métriques de qualité des clusters
        print("\nMétriques de qualité des clusters :")
        print(f"  - Score de silhouette : {silhouette_score(X_embedded, labels_pred):.4f}")
        print(f"  - Score de Calinski-Harabasz : {calinski_harabasz_score(X_embedded, labels_pred):.4f}")
        print(f"  - Score de Davies-Bouldin : {davies_bouldin_score(X_embedded, labels_pred):.4f}")
        
        # Calcul et affichage de la pureté des clusters si des annotations sont disponibles
        if has_annotations and len(true_labels) == len(labels_pred):
                # Mapper les labels des clusters aux labels d'annotation
                cluster_to_label = {}
                for cluster in np.unique(labels_pred):
                    # Trouver le label d'annotation le plus courant dans ce cluster
                    mask = (labels_pred == cluster)
                    true_labels_in_cluster = true_labels[mask]
                    if len(true_labels_in_cluster) > 0:  # Éviter les clusters vides
                        cluster_to_label[cluster] = np.argmax(np.bincount(true_labels_in_cluster))
                
                # Prédire les labels basés sur la correspondance cluster-annotation
                predicted_labels = np.array([cluster_to_label[cluster] for cluster in labels_pred])
                
                # Calculer la pureté
                purity = np.sum(predicted_labels == true_labels) / len(true_labels)
                print(f"\nMétriques par rapport aux annotations :")
                print(f"  - Pureté des clusters : {purity:.4f}")
                
                # Afficher la matrice de confusion
                cm = confusion_matrix(true_labels, predicted_labels)
                print("\nMatrice de confusion :")
                print(cm)
                
                # Afficher le rapport de classification
                print("\nRapport de classification :")
                print(classification_report(true_labels, predicted_labels, 
                                         target_names=['W', 'N1', 'N2', 'N3', 'REM']))
        
        # Affichage de la répartition des clusters
        unique, counts = np.unique(labels_pred, return_counts=True)
        print("\nRépartition des clusters :")
        for cluster, count in zip(unique, counts):
            print(f"  - Cluster {cluster}: {count} époques ({count/len(labels_pred):.1%})")

    except FileNotFoundError:
        print("Erreur : Fichier EEG non trouvé.")
    except Exception as e:
        print(f"Une erreur est survenue : {e}")
        raise

def benchmark_pipeline(X, X_embedded, true_labels, reducer_name='UMAP', clustering_name='KMeans', n_clusters=5, min_cluster_size=10):
    """
    X : features complètes
    X_embedded : projection 2D (UMAP, PCA, etc.)
    true_labels : labels de l'hypnogramme
    reducer_name : 'UMAP', 'PCA', 'None'
    clustering_name : 'KMeans', 'HDBSCAN'
    n_clusters : pour KMeans
    min_cluster_size : pour HDBSCAN
    """
    print(f"\n=== Pipeline: {reducer_name} + {clustering_name} ===")

    # Clustering
    if clustering_name == 'KMeans':
        if USE_GPU:
            kmeans = cuKMeans(n_clusters=n_clusters, random_state=42, output_type='numpy')
            X_embedded_gpu = cp.asarray(X_embedded.astype(np.float32))
            kmeans.fit(X_embedded_gpu)
            labels_pred = kmeans.labels_
            del X_embedded_gpu
            cp.get_default_memory_pool().free_all_blocks()
        else:
            kmeans = skKMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels_pred = kmeans.fit_predict(X_embedded)
    elif clustering_name == 'HDBSCAN':
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=5, gen_min_span_tree=True)
        labels_pred = clusterer.fit_predict(X_embedded)
    else:
        raise ValueError("clustering_name doit être 'KMeans' ou 'HDBSCAN'.")

    # Evaluation clustering
    mask_valid = labels_pred >= 0  # utile pour HDBSCAN
    n_clusters_found = len(np.unique(labels_pred[mask_valid]))
    print(f"\nNombre de clusters trouvés (excluant bruit): {n_clusters_found}")

    if np.sum(mask_valid) > 0:
        try:
            print(f"  - Score de silhouette: {silhouette_score(X_embedded[mask_valid], labels_pred[mask_valid]):.4f}")
            print(f"  - Score de Calinski-Harabasz: {calinski_harabasz_score(X_embedded[mask_valid], labels_pred[mask_valid]):.4f}")
            print(f"  - Score de Davies-Bouldin: {davies_bouldin_score(X_embedded[mask_valid], labels_pred[mask_valid]):.4f}")
        except Exception as e:
            print(f"  - Erreur dans le calcul des métriques: {str(e)[:100]}")

    # Visualisation
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels_pred, cmap='tab20', s=50, alpha=0.7)
    plt.colorbar(scatter, label='Cluster')
    plt.title(f'Visualisation {reducer_name} + {clustering_name}')
    plt.xlabel(f'{reducer_name} 1')
    plt.ylabel(f'{reducer_name} 2')
    plt.tight_layout()
    plt.show()

    # Evaluation vs annotations
    if true_labels is not None and len(true_labels) == len(labels_pred) and n_clusters_found > 0:
        print("\nMétriques par rapport aux annotations :")
        
        # Créer un mappage cluster -> label majoritaire
        cluster_to_label = {}
        for cluster in np.unique(labels_pred[mask_valid]):
            mask = (labels_pred == cluster)
            true_labels_in_cluster = true_labels[mask]
            if len(true_labels_in_cluster) > 0:
                cluster_to_label[cluster] = np.argmax(np.bincount(true_labels_in_cluster))

        # Prédire les labels basés sur la correspondance cluster-annotation
        predicted_labels = np.array([cluster_to_label.get(cluster, -1) for cluster in labels_pred])
        
        # Calculer et afficher la pureté
        purity = np.sum(predicted_labels == true_labels) / len(true_labels)
        print(f"  - Pureté des clusters: {purity:.4f}")

        # Matrice de confusion
        cm = confusion_matrix(true_labels, predicted_labels)
        print("\nMatrice de confusion :")
        print(cm)

        # Rapport de classification détaillé
        print("\nRapport de classification :")
        # Déterminer les classes uniques présentes
        unique_labels = np.unique(np.concatenate((true_labels, predicted_labels)))
        # Créer les noms de classes dynamiquement
        stage_names = ['W', 'N1', 'N2', 'N3', 'REM']
        # Ajouter des noms pour les classes supplémentaires si nécessaire
        if len(unique_labels) > len(stage_names):
            for i in range(len(stage_names), max(unique_labels) + 1):
                stage_names.append(f'Class_{i}')
        print(classification_report(true_labels, predicted_labels, 
                                 labels=range(len(stage_names)),
                                 target_names=stage_names,
                                 zero_division=0))
    
    # Répartition des clusters
    unique, counts = np.unique(labels_pred, return_counts=True)
    print("\nRépartition des clusters :")
    for cluster, count in zip(unique, counts):
        if cluster == -1:
            print(f"  - Bruit: {count} époques ({count/len(labels_pred):.1%})")
        else:
            print(f"  - Cluster {cluster}: {count} époques ({count/len(labels_pred):.1%})")
    
    return labels_pred

def main():
    try:
        # Charger les données EEG avec les annotations
        raw, epochs, true_labels = load_edf_with_annotations('SC4001E0-PSG.edf', 'SC4001EC-Hypno.edf')
        raw.filter(1., 40.)
        data = epochs.get_data()
        sf = raw.info['sfreq']
        
        # Stocker les vrais labels comme variable globale de la fonction
        main.true_labels = true_labels
        has_annotations = len(true_labels) > 0 and len(true_labels) == len(epochs)
        if has_annotations:
            print(f"\n{len(true_labels)} annotations chargées avec les distributions suivantes :")
            unique, counts = np.unique(true_labels, return_counts=True)
            stage_names = ['W', 'N1', 'N2', 'N3', 'REM']
            for label, count in zip(unique, counts):
                print(f"  - {stage_names[label]}: {count} époques ({count/len(true_labels):.1%})")
        
        bands = np.array([(0.5,4), (4,8), (8,12), (12,15), (12,30), (30,50)], dtype=np.float32)

        # Traitement des canaux
        n_epochs, n_channels, n_times = data.shape
        print(f"\nTraitement de {n_epochs} époques avec {n_channels} canaux...")

        # Préparer les données pour le traitement parallèle
        args_list = []
        for i in range(n_epochs):
            # Utiliser le premier canal comme référence pour la cohérence
            ch_ref = data[i, 0] if n_channels > 0 else None
            position_norm = i / n_epochs
            
            for j in range(n_channels):
                args_list.append((data[i, j], sf, bands, ch_ref, position_norm))

        # Traitement parallèle des canaux
        features_list = []
        num_cores = psutil.cpu_count(logical=False)
        num_workers = min(4, num_cores)
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            features_list = list(tqdm(executor.map(process_channel_wrapper, args_list), 
                                  total=len(args_list), 
                                  desc="Traitement des canaux"))

        # Remodeler les caractéristiques
        X = np.array(features_list).reshape(n_epochs, n_channels, -1)
        X = X.reshape(n_epochs, -1)  # Aplatir pour UMAP

        # Réduction de dimension avec UMAP
        print("\nRéduction de dimension avec UMAP...")
        if USE_GPU:
            reducer = cuUMAP(n_components=2, random_state=42, output_type='numpy')
            X_umap = reducer.fit_transform(X.astype(np.float32))
        else:
            reducer = umap.UMAP(n_components=2, random_state=42)
            X_umap = reducer.fit_transform(X)
        
        # Réduction de dimension avec PCA pour comparaison
        print("Réduction de dimension avec PCA...")
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X)

        # === Benchmarks ===
        # 1. UMAP + KMeans
        print("\n" + "="*50)
        print("1. UMAP + KMeans (baseline)")
        print("="*50)
        benchmark_pipeline(X, X_umap, true_labels if has_annotations else None, 
                          reducer_name='UMAP', clustering_name='KMeans', n_clusters=5)
        
        # 2. UMAP + HDBSCAN
        print("\n" + "="*50)
        print("2. UMAP + HDBSCAN")
        print("="*50)
        benchmark_pipeline(X, X_umap, true_labels if has_annotations else None,
                          reducer_name='UMAP', clustering_name='HDBSCAN', min_cluster_size=10)
        
        # 3. PCA + KMeans
        print("\n" + "="*50)
        print("3. PCA + KMeans")
        print("="*50)
        benchmark_pipeline(X, X_pca, true_labels if has_annotations else None,
                          reducer_name='PCA', clustering_name='KMeans', n_clusters=5)
        
        # 4. Sans réduction de dimension (KMeans direct sur les features)
        if X.shape[1] > 2:  # Éviter si déjà en 2D
            print("\n" + "="*50)
            print("4. Sans réduction de dimension (KMeans direct sur les features)")
            print("="*50)
            # On utilise PCA pour réduire à 2D juste pour la visualisation
            benchmark_pipeline(X, X_pca, true_labels if has_annotations else None,
                             reducer_name='PCA (visu only)', clustering_name='KMeans', n_clusters=5)

    except FileNotFoundError:
        print("Erreur : Fichier EEG non trouvé.")
    except Exception as e:
        print(f"Une erreur est survenue : {e}")
        raise

if __name__ == "__main__":
    main()
