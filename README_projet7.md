# README — Projet 7 : Prévision de la demande de vélos partagés

## Contexte

**Dataset** : Bike Sharing Dataset — UCI Machine Learning Repository  
**Problématique** : Comment la météo, la saison et le calendrier influencent-ils la demande de vélos partagés ?  
**Fichier principal** : `projet7_bike_sharing.ipynb`

---

## Structure du notebook

Le notebook contient **18 blocs de code** organisés en 7 sections logiques.

---

## Section 1 — Setup

### Bloc 1 — Importation des bibliothèques

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
```

| Bibliothèque | Rôle |
|---|---|
| `numpy` | Calculs numériques (polyfit, percentile, sqrt…) |
| `pandas` | Manipulation du DataFrame, parsing des dates |
| `matplotlib` | Graphiques de base (subplots, bar, scatter, fill_between) |
| `seaborn` | Graphiques statistiques (boxplot, heatmap, stripplot) |
| `sklearn.linear_model` | Régression Linéaire et Ridge |
| `sklearn.ensemble` | Random Forest et Gradient Boosting |
| `sklearn.metrics` | MAE, RMSE, R² |
| `sklearn.preprocessing.StandardScaler` | Normalisation des features pour les modèles linéaires |

---

### Bloc 2 — Téléchargement et chargement des données

**Fonctions utilisées :**

- `urllib.request.urlretrieve(url, path)` : télécharge le fichier ZIP depuis l'URL UCI sans librairie externe.
- `zipfile.ZipFile(path, 'r').extractall(dir)` : décompresse l'archive dans le dossier `bike_data/`.
- `pd.read_csv(path)` : charge `day.csv` et `hour.csv` en DataFrames pandas.

**Ce bloc produit :** `day_df` (731 lignes × 16 colonnes) et `hour_df` (17 379 lignes × 17 colonnes).  
**Note :** le bloc vérifie l'existence du ZIP avant de re-télécharger (idempotent).

---

## Section 2 — Compréhension des données

### Bloc 3 — Exploration initiale

**Fonctions utilisées :**

- `df.shape` : dimensions du DataFrame.
- `df.head(n)` : affiche les n premières lignes.
- `df.dtypes` : types de chaque colonne.

**Variables clés du dataset :**

| Variable | Description |
|---|---|
| `cnt` | **Cible** : total de locations (casual + registered) |
| `temp` | Température normalisée [0, 1] → ×41 pour avoir les °C |
| `hum` | Humidité normalisée [0, 1] |
| `windspeed` | Vitesse du vent normalisée [0, 1] |
| `season` | 1=Printemps, 2=Été, 3=Automne, 4=Hiver |
| `weekday` | 0=Lundi … 6=Dimanche |
| `workingday` | 1 si jour ouvré, 0 sinon |
| `holiday` | 1 si jour férié |
| `weathersit` | 1=Clair, 2=Nuageux, 3=Pluie légère, 4=Pluie forte |
| `casual` | Locations utilisateurs non-inscrits |
| `registered` | Locations utilisateurs inscrits |

---

### Bloc 4 — Statistiques descriptives

**Fonctions utilisées :**

- `df.describe()` : statistiques classiques (count, mean, std, min, quartiles, max).
- `df.isnull().sum()` : nombre de valeurs manquantes par colonne.
- `df.duplicated().sum()` : détection des doublons.

**Résultat attendu :** 0 valeur manquante, 0 doublon — le dataset est propre.

---

## Section 3 — Prétraitement

### Bloc 5 — Conversion de `dteday` en datetime

**Action 3 du cahier des charges.**

```python
day_df['dteday'] = pd.to_datetime(day_df['dteday'])
```

**Pourquoi :** La colonne `dteday` est initialement une chaîne de caractères (`object`). La conversion en `datetime64[ns]` permet :
- d'extraire des composantes (`.dt.month`, `.dt.isocalendar().week`),
- de trier et filtrer chronologiquement (pour le split temporel),
- de l'utiliser comme axe X dans les séries temporelles.

---

### Bloc 6 — Vérification `casual + registered = cnt`

**Action 6 du cahier des charges.**

```python
day_df['_check'] = day_df['casual'] + day_df['registered']
is_valid = (day_df['_check'] == day_df['cnt']).all()
```

**Résultat :** `True` pour toutes les lignes.  
**Conséquence importante :** `casual` et `registered` sont des **composantes directes de `cnt`**. Les inclure comme features constituerait une **fuite de données (data leakage)** : le modèle apprendrait à calculer une somme plutôt qu'à prédire la demande réelle. Ces colonnes sont donc **exclues** de la matrice de features.

---

### Bloc 7 — Création des variables calendaires

**Action 5 du cahier des charges.**

```python
day_df['mois']        = day_df['dteday'].dt.month
day_df['semaine']     = day_df['dteday'].dt.isocalendar().week.astype(int)
day_df['est_weekend'] = day_df['weekday'].isin([0, 6]).astype(int)

def get_periode(mois):
    ...
day_df['periode']     = day_df['mois'].apply(get_periode)
day_df['periode_num'] = day_df['periode'].map({'Hiver': 0, 'Printemps': 1, 'Été': 2, 'Automne': 3})
```

| Variable | Type | Valeurs | Intérêt |
|---|---|---|---|
| `mois` | int | 1–12 | Saisonnalité fine (meilleure résolution que `mnth` déjà présent, utilisé ici pour la clarté) |
| `semaine` | int | 1–53 | Capture les tendances hebdomadaires et les vacances scolaires |
| `est_weekend` | int | 0 / 1 | Distingue usage loisirs (WE) et navette (semaine) |
| `periode` | str | Hiver/Printemps/Été/Automne | Regroupement saisonnier calendaire (≠ `season` UCI qui suit un découpage différent) |
| `periode_num` | int | 0–3 | Encodage numérique de `periode` pour les modèles |

**Fonctions :**
- `dt.month`, `dt.isocalendar().week` : extraction de composantes temporelles depuis un datetime pandas.
- `Series.isin([...]).astype(int)` : conversion booléen → entier binaire.
- `Series.apply(func)` : applique une fonction personnalisée ligne par ligne.
- `Series.map(dict)` : encodage par dictionnaire.

---

## Section 4 — Analyse exploratoire (EDA)

### Bloc 8 — Figure 1 : Série temporelle de `cnt`

**Consigne :** *Série temporelle de `cnt`.*

```python
ax.plot(day_df['dteday'], day_df['cnt'], ...)
ax.fill_between(day_df['dteday'], day_df['cnt'], ...)
rolling_mean = day_df.set_index('dteday')['cnt'].rolling(30).mean()
```

**Fonctions :**
- `ax.plot()` : tracé de la série brute quotidienne.
- `ax.fill_between()` : remplissage sous la courbe pour visualiser les volumes.
- `Series.rolling(30).mean()` : moyenne mobile sur 30 jours — lisse le bruit quotidien et révèle la tendance.
- `plt.savefig()` : sauvegarde en PNG haute résolution (150 dpi).

**Observations clés :**
- Tendance haussière nette entre 2011 et 2012 (popularité croissante du service).
- Cycles saisonniers réguliers : pics estivaux, creux hivernaux.

---

### Bloc 9 — Figure 2 : Boxplot par saison

**Consigne :** *Boxplot par saison.*

```python
sns.boxplot(data=day_df, x='season_label', y='cnt', order=order_saisons, ...)
sns.stripplot(data=day_df, x='season_label', y='cnt', ...)
```

**Fonctions :**
- `sns.boxplot()` : affiche médiane, quartiles Q1/Q3, moustaches (±1.5×IQR) et outliers par saison.
- `sns.stripplot()` : superpose les points individuels pour voir la distribution réelle (transparence `alpha=0.15`).
- `Series.map(dict)` : conversion des codes numériques de saison en labels lisibles.

**Observations :** L'été et l'automne ont les médianes les plus élevées ; l'hiver présente la plus grande variabilité basse.

---

### Bloc 10 — Figure 3 : Relation température / demande

**Consigne :** *Relation température/demande.*

```python
day_df['temp_reel'] = day_df['temp'] * 41
z = np.polyfit(day_df['temp_reel'], day_df['cnt'], 2)
p = np.poly1d(z)
```

**Fonctions :**
- `day_df['temp'] * 41` : dénormalisation (le dataset UCI normalise temp entre 0 et 1 ; maximum réel = 41°C).
- `np.polyfit(x, y, deg=2)` : ajustement d'une courbe polynomiale de degré 2 (parabole) aux données.
- `np.poly1d(coefficients)` : crée un objet fonction évaluable pour tracer la courbe.
- `ax.scatter(..., c=day_df['season'], cmap=...)` : coloration des points par saison.
- `sns.heatmap(corr, annot=True)` : carte de chaleur des corrélations entre variables météo et `cnt`.

**Corrélations observées :**
- Température : **corrélation positive forte** (~0.63)
- Humidité : **corrélation négative modérée** (~-0.10)
- Vent : **corrélation négative faible** (~-0.23)

---

### Bloc 11 — Analyse des tendances

**Action 4 du cahier des charges.**

```python
day_df.groupby('mois')['cnt'].mean()
day_df.groupby('season_label')['cnt'].mean()
day_df.groupby('weekday')['cnt'].mean()
day_df.groupby('weather_label')['cnt'].mean()
```

**Fonctions :**
- `groupby(col)[target].mean()` : moyenne de la cible pour chaque modalité d'une variable catégorielle — c'est l'**agrégation temporelle** clé du projet.
- `plt.subplots(2, 2)` : grille 2×2 de sous-graphiques pour comparer les 4 dimensions en une figure.

**Observations :**
- La demande peak en juin-juillet-août (mois 6-8) et chute en décembre-janvier.
- Les jours de pluie forte réduisent la demande d'environ 50% vs temps clair.

---

## Section 5 — Stratégie de split

### Bloc 12 — Split temporel

**Action 7 du cahier des charges.**

#### Stratégie choisie : **Split Temporel**

| | Split Aléatoire | Split Temporel (choisi) |
|---|---|---|
| **Principe** | Mélange toutes les dates | Entraîne sur le passé, teste sur le futur |
| **Problème** | Data leakage temporel | Aucun |
| **Réalisme** | Artificiel | Simule le cas réel de production |
| **Métriques** | Biaisées à la hausse | Représentatives |

**Justification de la date de coupure `2012-07-01` :**
- Données disponibles : 2011-01-01 → 2012-12-31 (2 ans).
- La coupure donne ~80% train / ~20% test, ratio standard en machine learning.
- Le jeu de test couvre l'été 2012 (période de forte demande) et l'hiver 2012-13, testant la capacité du modèle sur des conditions variées.

```python
CUTOFF_DATE = pd.Timestamp('2012-07-01')
train_df = day_df[day_df['dteday'] <  CUTOFF_DATE].copy()
test_df  = day_df[day_df['dteday'] >= CUTOFF_DATE].copy()
```

**Fonctions :**
- `pd.Timestamp()` : création d'un timestamp pandas pour la comparaison avec `dteday`.
- Filtrage booléen `df[condition].copy()` : extraction des sous-ensembles sans modifier l'original.

---

### Bloc 13 — Préparation de la matrice de features

```python
exclude_cols = ['instant', 'dteday', 'cnt', 'casual', 'registered', ...]
feature_cols = [c for c in day_df.columns if c not in exclude_cols]

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)   # fit ET transform sur le train
X_test_sc  = scaler.transform(X_test)        # transform UNIQUEMENT sur le test
```

**Pourquoi exclure ces colonnes :**

| Colonne exclue | Raison |
|---|---|
| `instant` | Identifiant sans information prédictive |
| `dteday` | Date brute (déjà encodée via `yr`, `mnth`, `mois`, `semaine`) |
| `cnt` | Variable cible — ne pas mettre en feature |
| `casual`, `registered` | Composantes directes de `cnt` → data leakage |
| `season_label`, `weather_label`, `periode` | Versions texte déjà encodées numériquement |

**Pourquoi `scaler.fit_transform` sur train et `transform` sur test :**  
Le scaler ne doit "voir" que les données d'entraînement pour calculer moyenne et écart-type. Appliquer `fit` sur le test constituerait un data leakage.

**Note :** La normalisation est appliquée uniquement pour **Linear Regression** et **Ridge**. Random Forest et Gradient Boosting sont **invariants à l'échelle** des features, donc on leur passe `X_train` non normalisé.

---

## Section 6 — Modèles

### Bloc 14 — Entraînement des 4 modèles

**Action 8 du cahier des charges.**

#### Fonction `evaluate_model(name, model, X_tr, X_te, y_tr, y_te)`

Cette fonction factorise l'entraînement et l'évaluation pour les 4 modèles :
- `model.fit(X_tr, y_tr)` : entraîne le modèle sur le jeu d'entraînement.
- `model.predict(X_te)` : génère les prédictions sur le jeu de test.
- `mean_absolute_error(y_te, y_pred)` : MAE — erreur absolue moyenne en nombre de vélos/jour.
- `mean_squared_error(y_te, y_pred)` + `np.sqrt()` : RMSE — pénalise davantage les grandes erreurs.
- `r2_score(y_te, y_pred)` : R² — proportion de variance expliquée (0 = nul, 1 = parfait).

**Retourne** un dictionnaire avec les métriques, les prédictions (`y_pred`) et l'objet modèle.

---

#### Modèle 1 : Régression Linéaire (`LinearRegression`)

```python
lr = LinearRegression()
```

- **Hypothèse** : relation linéaire entre chaque feature et `cnt`.
- **Avantage** : rapide, interprétable (coefficients directement lisibles).
- **Inconvénient** : ne capture pas les non-linéarités (ex. : l'effet de la température plafonne au-dessus de 25°C).
- **Utilise** : features normalisées (`X_train_sc`, `X_test_sc`).

---

#### Modèle 2 : Ridge Regression (`Ridge`)

```python
ridge = Ridge(alpha=1.0)
```

- **Hypothèse** : régression linéaire + régularisation L2 (pénalisation des grands coefficients).
- **Paramètre** : `alpha=1.0` contrôle la force de la régularisation (plus alpha est grand, plus les coefficients sont réduits vers 0).
- **Avantage vs OLS** : réduit le surajustement lorsque les features sont corrélées (multicolinéarité).
- **Différence avec Lasso** : Ridge garde tous les coefficients (jamais exactement 0), contrairement à Lasso qui fait de la sélection de features.

---

#### Modèle 3 : Random Forest Regressor

```python
rf = RandomForestRegressor(n_estimators=200, max_depth=10,
                           min_samples_leaf=2, random_state=42, n_jobs=-1)
```

| Paramètre | Valeur | Signification |
|---|---|---|
| `n_estimators` | 200 | Nombre d'arbres de décision dans la forêt |
| `max_depth` | 10 | Profondeur maximale de chaque arbre (évite le surajustement) |
| `min_samples_leaf` | 2 | Nombre minimum d'échantillons dans une feuille terminale |
| `random_state` | 42 | Reproductibilité des résultats |
| `n_jobs` | -1 | Utilise tous les cœurs CPU disponibles |

**Principe** : Entraîne `n_estimators` arbres sur des sous-échantillons aléatoires (bagging) et des sous-ensembles aléatoires de features (feature randomness). La prédiction finale est la moyenne des prédictions de tous les arbres.  
**Avantage** : capture les non-linéarités et les interactions entre variables sans normalisation.  
**Ne nécessite pas** de normalisation des features.

---

#### Modèle 4 : Gradient Boosting Regressor

```python
gb = GradientBoostingRegressor(n_estimators=300, learning_rate=0.08,
                               max_depth=4, subsample=0.8, random_state=42)
```

| Paramètre | Valeur | Signification |
|---|---|---|
| `n_estimators` | 300 | Nombre d'arbres ajoutés séquentiellement |
| `learning_rate` | 0.08 | Pas d'apprentissage (contribution de chaque arbre) |
| `max_depth` | 4 | Arbres peu profonds (stumps enrichis) |
| `subsample` | 0.8 | Fraction aléatoire des données utilisée à chaque itération (stochastic GB) |

**Principe** : Contrairement au Random Forest (arbres parallèles indépendants), le Gradient Boosting construit les arbres **séquentiellement** : chaque arbre corrige les erreurs du précédent en descendant le gradient de la fonction de perte.  
**Avantage** : généralement le plus performant sur les données tabulaires structurées.  
**Trade-off** : plus lent à entraîner, plus sensible aux hyperparamètres.

---

### Bloc 15 — Comparaison des modèles

```python
metrics_df = pd.DataFrame([...]).sort_values('R2', ascending=False)
```

**Métriques d'évaluation (Action 9) :**

| Métrique | Formule | Interprétation |
|---|---|---|
| **MAE** | `mean(|y - ŷ|)` | Erreur absolue moyenne en vélos/jour — facile à interpréter |
| **RMSE** | `sqrt(mean((y - ŷ)²))` | Pénalise les grandes erreurs (outliers) — en vélos/jour |
| **R²** | `1 - SS_res/SS_tot` | % de variance expliquée (plus proche de 1 = mieux) |

**Visualisation** : Graphique en barres comparant les 3 métriques pour chaque modèle.

---

## Section 7 — Analyse du meilleur modèle

### Bloc 16 — Figure 4 : Résidus du meilleur modèle

**Consigne :** *Résidus du meilleur modèle.*

```python
residuals = y_test.values - y_pred_best
```

**Trois graphiques :**

1. **Résidus vs prédictions** : Si le modèle est bien spécifié, les résidus doivent être répartis aléatoirement autour de 0 sans structure. Un pattern en éventail indiquerait une hétéroscédasticité.

2. **Distribution des résidus** (`ax.hist()`) : Idéalement proche d'une distribution normale centrée en 0. Un décalage indique un biais systématique.

3. **Valeurs réelles vs prédites** (`ax.scatter()`) : Les points doivent se regrouper autour de la diagonale rouge (ligne de prédiction parfaite). Les écarts révèlent les cas difficiles.

---

### Bloc 17 — Importance des features

```python
imp_df = pd.DataFrame({'feature': feature_cols,
                       'importance': best_model.feature_importances_})
```

**Fonctions :**
- `model.feature_importances_` : attribut disponible sur `RandomForestRegressor` et `GradientBoostingRegressor`. Mesure la réduction moyenne de l'impureté (critère MSE) apportée par chaque feature sur l'ensemble des arbres.
- `ax.barh()` : graphique en barres horizontales pour une meilleure lisibilité des noms de features.

**Note :** Pour la régression linéaire, utilisez `lr.coef_` (coefficients standardisés après normalisation).

---

### Bloc 18 — Interprétation finale

**Action 10 du cahier des charges.**

| Facteur | Direction | Explication |
|---|---|---|
| Température (`temp`) | ↑ forte | Relation positive — pic entre 20-25°C puis légère baisse au-delà |
| Année (`yr`) | ↑ | Popularité croissante du service entre 2011 et 2012 |
| Météo claire (`weathersit=1`) | ↑ | Beau temps favorise fortement l'usage |
| Été / Automne | ↑ | Saisons de forte demande (loisirs + navette confortables) |
| Jours ouvrés (`workingday`) | ↑ légère | Les abonnés utilisent le service pour aller au travail |
| Vent fort (`windspeed`) | ↓ | Décourageant physiquement |
| Humidité élevée (`hum`) | ↓ modérée | Sensation désagréable + risque de pluie |
| Pluie (`weathersit ≥ 3`) | ↓ forte | Impact négatif majeur (~50% de baisse) |
| Hiver (`season=4`) | ↓ | Températures basses → forte réduction de la demande |

---

## Fichiers produits

| Fichier | Description |
|---|---|
| `projet7_bike_sharing.ipynb` | Notebook complet (toutes les actions du cahier des charges) |
| `README_projet7.md` | Ce fichier — documentation détaillée |
| `fig1_serie_temporelle.png` | Série temporelle de `cnt` avec moyenne mobile 30 jours |
| `fig2_boxplot_saison.png` | Boxplot + stripplot de `cnt` par saison |
| `fig3_temperature_demande.png` | Scatter température/demande + heatmap de corrélation |
| `fig4_tendances.png` | Grille 2×2 : demande par mois, saison, jour, météo |
| `fig5_split_temporel.png` | Visualisation du découpage train/test |
| `fig6_comparaison_modeles.png` | Comparaison MAE / RMSE / R² des 4 modèles |
| `fig7_residus.png` | Résidus vs prédictions / distribution / réel vs prédit |
| `fig8_feature_importance.png` | Importance des variables du meilleur modèle |

---

## Erreurs à éviter (rappel cahier des charges)

1. **Ne pas utiliser `cnt` comme feature** — c'est la variable à prédire.
2. **Ne pas inclure `casual` ni `registered` comme features** — elles sont des composantes de `cnt` et créent un data leakage parfait.
3. **Justifier le split** — un split aléatoire sur une série temporelle est une erreur méthodologique grave.
4. **Ne pas scaler les données pour Random Forest / Gradient Boosting** — ces modèles basés sur des arbres de décision sont invariants à l'échelle.

---

## Comment lancer le notebook

```bash
# Installer les dépendances si nécessaire
pip install numpy pandas matplotlib seaborn scikit-learn jupyter

# Lancer Jupyter
jupyter notebook projet7_bike_sharing.ipynb
```

Exécutez les cellules dans l'ordre. Le bloc 2 télécharge automatiquement les données depuis UCI.
