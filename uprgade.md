
🚀 Améliorations possibles
Si tu veux aller plus loin :
1️⃣ Ajoute d'autres features EEG :

Ratio alpha/theta

Puissance gamma

Complexité (entropy)

Dynamique temporelle (variation entre fenêtres)

2️⃣ Utilise HDBSCAN :

clustering plus flexible que k-means (capte mieux les transitions entre états).

3️⃣ Multi-sujets :

Applique ton pipeline sur plusieurs PSG → robustesse.

4️⃣ Autoencodeur + UMAP :

Encode les fenêtres EEG avec un autoencodeur, puis fais UMAP + clustering → souvent plus performant.

