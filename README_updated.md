# 🗺️ Projet : Modèle invariant pour l’estimation de la longueur optimale d’un TSP

> **TL;DR** : Ce dépôt contient tout le nécessaire pour entraîner, tester et analyser un modèle apprenant à prédire la distance optimale (ou son score) d’instances TSP de 30 à 50 nœuds, tout en restant invariant aux transformations euclidiennes (rotation, translation, permutation de sommets…).

---

## 1. Contexte & objectifs

Les algorithmes exacts pour le *Traveling Salesman Problem* (TSP) deviennent vite inabordables sur de grandes instances. L’objectif est donc de **regresser** la longueur optimale d’un tour via un réseau de neurones, tout en apprenant des invariances qui garantissent qu’une instance tournée, translatée ou permutée produira la même prédiction.

- **Dataset d’entraînement** : `tsp_30_50.json` (≈ 80 k instances).
- **Dataset de test**        : `tsp_test_20k_30_50.json` (20 k instances, jamais vues).

---

## 2. Structure attendue des datasets

Chaque fichier JSON est **une liste** d’entrées :

```json
{
  "id": 0,
  "n_points": 34,
  "points": [[x1, y1], [x2, y2], …],
  "opt_tour": [0, 24, 11, …, 0],
  "score": 5.1610150595416115   // parfois nommé "opt_dist" ou "opt_dist_true"
}
```

> **⚠️ Attention :** Si votre dump emploie `"opt_dist"` (ou `"opt_dist_true"`) à la place de `"score"`, adaptez les scripts en conséquence ; les exemples ci‑dessous le gèrent par défaut.

---

## 3. Workflow rapide (datasets déjà générés)

> Très bonne question ! Voici les grandes étapes **à partir du moment où vous disposez déjà** des fichiers `/home/dcbrain/Bureau/ml_rd/data/tsp_dataset/tsp_30_50.json` et `…/tsp_test_20k_30_50.json`.

### 3.1 Préparation des données d’entraînement
- Vérifiez la présence des clés `points`, `opt_tour`, `score`/`opt_dist`.
- Mettez à jour `load_dataset()` si besoin pour mapper `opt_dist` → `score`.

### 3.2 Entraînement du modèle
```bash
python src/train_invariance.py     --dataset /home/dcbrain/Bureau/ml_rd/data/tsp_dataset/tsp_30_50.json     --batch_size 256     --epochs 50     --learning_rate 1e-3     --output_dir outputs/train
```
Le script :
- normalise les longueurs via **StandardScaler** ;
- applique des transformations (rotations, translations, permutations) *on‑the‑fly* ;
- sauvegarde :
  - `tsp_invariant_model.pt` ;
  - `scaler.pkl` ;
  - `metrics.json`, `training_history.png`.

### 3.3 Évaluation sur le dataset de test
```bash
python src/evaluate_model.py     --model outputs/train/tsp_invariant_model.pt     --scaler outputs/train/scaler.pkl     --dataset /home/dcbrain/Bureau/ml_rd/data/tsp_dataset/tsp_test_20k_30_50.json     --output_dir outputs/eval
```
Le script calcule la **MSE**, les différences absolues/relatives et génère des histogrammes (`invariance_metrics_eval.png`).

### 3.4 Analyse & interprétation
- Inspectez `outputs/eval/metrics.json` : *mse*, *mean_abs_diff*, *mean_rel_diff*, etc. doivent être faibles (≈ 0).
- (Optionnel) comparez avec d’autres architectures ou tailles de graphes.

### 3.5 Résumé visuel
```
[DATASETS PRÊTS]
      |
      v
[ENTRAÎNEMENT]
  - train_invariance.py
  - Sauvegarde modèle + scaler
      |
      v
[ÉVALUATION]
  - evaluate_model.py
  - Sur le dataset de test
      |
      v
[ANALYSE]
  - Interprétation des métriques
```

👉 **Besoin d’automatiser toute la chaîne ?** Dites‑le et je vous fournis un `Makefile` ou un *workflow* CI/CD.

---

## 4. Installation rapide

```bash
# 1 – Environnement Python ≥ 3.10
python -m venv .venv && source .venv/bin/activate

# 2 – Dépendances
pip install -r requirements.txt
```

### 4.1 Dépendances principales

| Paquet          | Version testée |
|-----------------|----------------|
| PyTorch         | 2.3.0          |
| scikit‑learn    | 1.5.0          |
| numpy           | 1.28.0         |
| matplotlib      | 3.9.0          |
| tqdm            | 4.66           |
| python‑dotenv   | 1.0            |

---

## 5. Arborescence recommandée

```
.
├── data/
│   └── tsp_dataset/
│       ├── tsp_30_50.json
│       └── tsp_test_20k_30_50.json
├── src/
│   ├── train_invariance.py
│   ├── evaluate_model.py
│   ├── transformations.py
│   ├── models.py
│   └── utils.py
├── outputs/
│   ├── train/
│   └── eval/
└── README.md
```

---

## 6. Reproductibilité

- Tous les scripts acceptent `--seed` (défaut : `42`).
- Les hachages Git et `torch.__config__.show()` sont loggés pour garantir l’auditabilité.

---

## 7. Instructions « Codex » (OpenAI) ✨

Pour générer ou compléter du code avec ChatGPT/Codex :
1. **Contextualisez** votre prompt (signature de fonction, contraintes temporelles…).
2. **Spécifiez** les limites : complexité, device, style PEP‑8.
3. **Validez** via `pytest` ou un petit notebook avant de pousser.

> Exemple :
> « Écris une fonction PyTorch `compute_tour_length(points, tour)` qui calcule la distance totale. Utilise seulement `torch` et vectorise au maximum. »

---

## 8. Roadmap (suggestion)

- [ ] Curriculum d’apprentissage pour des graphes > 50 nœuds.
- [ ] Exploration d’un *GNN* 
- [ ] Benchmark vs. heuristiques (LKH, Concorde).

---

## 9. Licence & citation

Sous licence **MIT**. Merci de citer :

```text
@misc{tsp_invariant_2025,
  author       = {Votre Nom},
  title        = {A learning‑based invariant estimator for TSP tour length},
  year         = 2025,
  howpublished = {GitHub},
  url          = {https://github.com/username/tsp-invariant-model}
}
```

---

### Bonnes expérimentations ! 🧩
