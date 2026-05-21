# ⚽ Machine Learning Approach for Optimal Football XI Selection Based on Opponent Tactical Analysis

> **Submitted to ICCA 2026** | American International University – Bangladesh (AIUB) | Section J, Group 02

A full end-to-end machine learning pipeline that predicts an optimal starting eleven for any La Liga fixture by jointly modelling player rolling form, opponent tactical context, formation compatibility, and head-to-head history — then estimates the match win probability.

---

## 📄 Paper

**"Machine Learning Approach for Optimal Football XI Selection Based on Opponent Tactical Analysis"**  
*Submitted to the 4th International Conference on Computing Advancements (ICCA 2026), Dhaka, Bangladesh*

**Authors:** Md. Sifti Al Mahmud · Md. Towhidul Islam · Anika Tabassum · Abrar Kabir

---

## 🗂️ Repository Structure

```
├── J_G02_Code.ipynb          # Main Jupyter notebook (full pipeline)
├── data/
│   └── README.md             # Dataset links and description
├── outputs/
│   ├── roc_curves.png
│   ├── confusion_matrix.png
│   ├── calibration_curves.png
│   ├── shap_beeswarm.png
│   └── feature_importance.png
└── README.md
```

> 📁 **Full dataset and output graphs:** [Google Drive](https://drive.google.com/drive/folders/1oxVY7wjn9pQIv54tcuPTkiKKpKfp_r3?usp=drive_link)  
> 💻 **Interactive notebook:** [Google Colab](https://colab.research.google.com/drive/1Hha5EYF62jQrT0uJs4JX3fXrySBxxMYN?usp=sharing)  
> 📝 **LaTeX source:** [Overleaf](https://www.overleaf.com/9579271149yfymjmvcftpg#963377)

---

## 📊 Dataset

- **Source:** [FBref – La Liga Stats](https://fbref.com/en/comps/12/La-Liga-Stats)
- **Seasons:** 2024–25 and 2025–26
- **Scope:** 23 La Liga clubs, 810 unique players
- **Final modelling table:** 25,187 player-match observations (after cleaning)
- **Label:** `Started` — binary (1 = in starting XI, 0 = substitute/did not start)

Six raw FBref tables were merged: per-player match logs, team scores & fixtures, per-team playing time, per-player season totals, pre-computed reference ratings, and a previously merged player-match table.

---

## 🔧 Pipeline Overview

```
Stage 1 — Data Ingestion
  └─ Merge 6 FBref tables on (Date, Squad); generate synthetic Match_ID

Stage 2 — Cleaning & Custom Rating
  └─ Remove ghost rows, fix unicode variants, build position-weighted
     match rating with Z-score normalisation + exponential decay (λ=0.1)

Stage 3 — Feature Engineering (178 → 126 features)
  └─ Rolling form | Fatigue/rest | Head-to-head vs opponent |
     Formation fit | Team & opponent context | Season baseline

Stage 4 — Model Benchmarking (5 classifiers)
  └─ Decision Tree | MLP | Random Forest | XGBoost ★ | Stacking
     Time-based 80/20 split + 3-fold TimeSeriesSplit + SMOTE (train only)

Stage 5 — Deployment
  └─ 8-formation search → slot-aware XI selection → win probability estimate
```

---

## ✨ Key Features

### Custom Position-Weighted Match Rating
Each player receives a per-match rating based on position-relevant events (e.g. clean sheets for GKs, goals/assists for FWs), Z-score normalised within their `(Match_ID, Team, Broad_Position)` peer group, then smoothed with exponential decay:

$$R_p^{(t)} = \frac{\sum_i \exp(-\lambda(t-i)) \cdot \text{MatchRating}_p^{(i)}}{\sum_i \exp(-\lambda(t-i))}, \quad \lambda = 0.1$$

### Feature Groups (126 final features)
| Group | Description |
|---|---|
| Rolling form | Mean/sum of key stats over last 3, 5, 10 matches |
| Fatigue / rest | Minutes load, days between matches, suspension risk |
| Head-to-head | Per `(Team, Player, Opponent)` historical averages |
| Formation fit | Historical start rate & rating per formation slot |
| Team / opponent context | Rolling goals, possession, win rate, points |

### Leakage-Free Design
All features are computed strictly from prior matches using `shift(1)`. SMOTE is applied only to the training fold. Time-based splits prevent future data from influencing past predictions.

---

## 📈 Results

### Lineup Classification (Validation Set)

| Model | AUC | F1 Macro | F1 Weighted | Accuracy |
|---|---|---|---|---|
| **XGBoost** ★ | **0.8654** | **0.7552** | **0.8009** | **0.8109** |
| Random Forest | 0.8111 | 0.7116 | 0.7585 | 0.7613 |
| Decision Tree | 0.7913 | 0.7064 | 0.7524 | 0.7533 |
| Neural Network | 0.7805 | 0.6904 | 0.7445 | 0.7520 |
| Stacking | 0.7076 | 0.6048 | 0.6250 | 0.6115 |

### Cross-Validation (3-Fold TimeSeriesSplit)

| Model | AUC | F1 Macro | Accuracy | Brier |
|---|---|---|---|---|
| **XGBoost** ★ | **0.8484 ± 0.013** | **0.7408 ± 0.012** | **0.7989 ± 0.010** | **0.1387 ± 0.006** |
| Random Forest | 0.8097 ± 0.007 | 0.7102 ± 0.003 | 0.7628 ± 0.010 | 0.1623 ± 0.005 |
| Decision Tree | 0.7656 ± 0.015 | 0.6871 ± 0.014 | 0.7266 ± 0.018 | 0.1872 ± 0.011 |
| Stacking | 0.7149 ± 0.004 | 0.6508 ± 0.003 | 0.6827 ± 0.007 | 0.2062 ± 0.003 |
| Neural Network | 0.7066 ± 0.058 | 0.6202 ± 0.050 | 0.7343 ± 0.015 | 0.2496 ± 0.017 |

### Win Probability Model
- **AUC:** 0.6818 | **Accuracy:** 0.6117 | **F1 Macro:** 0.6115
- Trained on 1,598 match-team observations; consistent with accepted standards for single-league football outcome prediction.

---

## 🚀 Deployment Example

> **FC Barcelona (H) vs Real Madrid** — 2025–26 La Liga

The system iterates over 8 candidate formations and selects the one with the highest mean starter probability:

- **Recommended formation:** 4-2-3-1 (mean starter prob = **0.887**)
- **Predicted XI:** Joan Garcia, Pau Cubasi, Gerard Martin, Jules Kounde, Joao Cancelo, Eric Garcia, Pedri, Lamine Yamal, Raphinha, Fermin Lopez, Ferran Torres
- **Win probability:** **56.6%** (home advantage factored in)
- **Accuracy:** 9/11 starters correctly identified ✅

---

## 🛠️ Installation & Usage

```bash
# Clone the repo
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# Install dependencies
pip install -r requirements.txt

# Or open directly in Colab
# https://colab.research.google.com/drive/1Hha5EYF62jQrT0uJs4JX3fXrySBxxMYN
```

### Requirements
```
pandas
numpy
scikit-learn
xgboost
lightgbm
imbalanced-learn
shap
matplotlib
seaborn
```

---

## 👥 Team

| Name | Student ID | Contributions |
|---|---|---|
| Md. Sifti Al Mahmud | 23-50099-1 | Dataset construction & cleaning, feature engineering, XGBoost tuning, SHAP analysis |
| Md. Towhidul Islam | 23-55036-3 | Position-weighted rating, win probability model, deployment pipeline, discussion & conclusion |
| Anika Tabassum | 23-55070-3 | Introduction, literature review, research gap analysis, calibration results, limitations |
| Abrar Kabir | 23-55095-3 | Model benchmarking (DT, MLP, RF, Stacking), evaluation metrics, ROC/confusion matrix, SHAP interpretation |

---

## 📚 Citation

```bibtex
@unpublished{mahmud2026football,
  title  = {Machine Learning Approach for Optimal Football XI Selection Based on Opponent Tactical Analysis},
  author = {Md. Sifti Al Mahmud and Md. Towhidul Islam and Anika Tabassum and Abrar Kabir},
  note   = {Submitted to the 4th International Conference on Computing Advancements (ICCA 2026), Dhaka, Bangladesh},
  year   = {2026}
}
```

---

## ⚠️ Limitations

- Currently limited to La Liga (2024–26); generalisation to other leagues (Premier League, Bundesliga, etc.) is untested.
- FBref does not expose injury, suspension, or fitness data — the model cannot account for player unavailability.
- Manager-specific tactical preferences are not explicitly modelled.
- The model adapts slowly to sudden changes (injuries, managerial decisions) due to reliance on lagged rolling features.

---

## 🔮 Future Work

- Extend to multiple top-five European leagues
- Integrate injury/layoff data from Transfermarkt or licensed providers
- Apply LSTM / temporal convolutional networks for match-to-match dynamics
- Explore graph neural network approaches (inspired by TacticAI) adapted to box-score data
- Build a live matchday dashboard with daily FBref updates

---

*Department of Electrical and Electronic Engineering, AIUB — Section J, Group 02*
