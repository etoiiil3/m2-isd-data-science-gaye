# Notes Grace — Day 1

## Fonction codée : `recognise_isolated` (speech/gmm_hmm.py)

**Ce qu'elle fait :** reconnaît un mot isolé  à partir de son enregistrement.

**Principe :**
- On a un modèle GMM-HMM par mot, déjà entraîné (`wm.models[mot]`).
- Le son à reconnaître est donné sous forme de MFCC (`feats`) : un tableau (T, 39), une ligne toutes les 10 ms.
- On fait « écouter » ce même son à chaque modèle : `wm.models[w].score(feats)` renvoie la
  log-vraisemblance log P(son | mot w), calculée par l'algorithme forward.
- On renvoie le mot dont le modèle donne le score le plus élevé :
  mot reconnu = argmax_w log P(X | w)

**Pourquoi le log :** le produit de centaines de petites probabilités tombe à 0 en machine ;
le log transforme les produits en sommes et ne change pas l'ordre (le max reste le max).

**Test :** `pytest tests/public/test_d1_classical.py::test_isolated` → passe ✅