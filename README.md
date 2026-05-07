# EPL Match Outcome Prediction 🏴󠁧󠁢󠁥󠁮󠁧󠁿⚽

> **AI in Business — Final Project | German International University (GIU)**  
> Multi-class classification pipeline to predict English Premier League match outcomes, with a Monte Carlo season simulator.

---

## 📌 Overview

This project builds an end-to-end machine learning pipeline on 3,800+ EPL matches spanning 11 seasons (2015–2026). The goal is to predict the outcome of any match — **Home Win**, **Draw**, or **Away Win** — using only pre-match historical data (no leakage), and to simulate the full 2025–26 season final standings using Monte Carlo methods.

---

## 👥 Team Members

| Name |
|------|
| Youssef Hassan |
| Moaz Abdelaleem |
| Omar Khaled |
| Omar Labib |

---

## 📁 Project Structure

```
├── data/
│   └── epl_all_seasons.csv      # Combined EPL dataset (2015–2026)
├── Epl_clean.ipynb               # Main notebook (full pipeline)
└── README.md
```

---

## 📊 Dataset

| Property | Detail |
|----------|--------|
| Source | EPL match results (2015–2026) |
| Size | 3,800+ matches across 11 seasons |
| Key columns | `HomeTeam`, `AwayTeam`, `FTHG`, `FTAG`, `FTR` |
| Target variable | `FTR` — Full-Time Result (H / D / A) |

**Outcome distribution:**
- Home Win: ~45%
- Away Win: ~32%
- Draw: ~23% *(hardest to predict)*

---

## 🔧 Pipeline

### 1. Data Cleaning & Preprocessing
- Filtered to essential columns; dropped nulls and duplicates
- Parsed and sorted dates chronologically (critical for temporal features)
- Encoded target: `Away=0`, `Draw=1`, `Home=2`

### 2. Feature Engineering (12 Features — No Data Leakage)
All features use `.shift(1)` to ensure only pre-match data is used.

| Feature | Description |
|---------|-------------|
| `Home_Goals_Roll` | Home team's avg goals scored (last 5 matches) |
| `Home_Conceded_Roll` | Home team's avg goals conceded (last 5 matches) |
| `Away_Goals_Roll` | Away team's avg goals scored (last 5 matches) |
| `Away_Conceded_Roll` | Away team's avg goals conceded (last 5 matches) |
| `Home_Form` | Home team's points per game (last 5 matches) |
| `Away_Form` | Away team's points per game (last 5 matches) |
| `Home_GD_Roll` | Home team's goal difference trend (last 5) |
| `Away_GD_Roll` | Away team's goal difference trend (last 5) |
| `Home_Season_Goals` | Home team's season-long avg goals scored |
| `Away_Season_Goals` | Away team's season-long avg goals scored |
| `GD_Diff` | Goal difference differential between teams |
| `Form_Diff` | Form points differential between teams |

### 3. Train / Test Split
- **Temporal split (80/20)** — training on earlier seasons, testing on later ones
- Features scaled with `StandardScaler`

### 4. Modeling — 5 Classifiers with GridSearchCV

| Model | Notes |
|-------|-------|
| Logistic Regression | Tuned `C`, `solver` |
| Decision Tree | Tuned `max_depth`, `min_samples_split` |
| Random Forest | Tuned `n_estimators`, `max_depth` |
| Gradient Boosting | Tuned `n_estimators`, `learning_rate` |
| XGBoost | Tuned `n_estimators`, `max_depth`, `learning_rate` |
| Voting Ensemble | Combines all 5 models |

### 5. Evaluation
- Metrics: Accuracy, Precision, Recall, F1-Score (weighted)
- 5-Fold Cross-Validation on training set
- Confusion matrix and classification report per model

---

## 🏆 Results

| Model | Accuracy |
|-------|----------|
| **XGBoost** | **56.2%** ✅ |
| Gradient Boosting | ~54% |
| Random Forest | ~53% |
| Logistic Regression | ~54% |
| Voting Ensemble | ~54% |
| *Random baseline* | *33.3%* |

**Top predictive features (XGBoost importance):**
1. `GD_Diff` — goal difference trend between teams
2. Season-long goal averages
3. Recent form differential

> Draws consistently had the lowest F1-score across all models — a known challenge in football prediction.

---

## 🎲 Monte Carlo Season Simulator

Using real **Gameweek 15 (December 2025)** standings as input, the simulator forecasts the full 2025–26 final table.

**How it works:**
1. Each team's win probability = Points Per Game / 3.0 (capped at 65%)
2. 500 full season simulations are run, each generating random outcomes for the remaining 23 matchdays
3. Final output: predicted points, Championship %, Top-4 %, and Relegation % per team

**2025–26 Predicted Champion: Arsenal** 🔴

---

## ⚙️ Installation & Usage

### Requirements
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
```

### Run
```bash
jupyter notebook Epl_clean.ipynb
```
Make sure `data/epl_all_seasons.csv` is in place before running.

---

## 🛠️ Tech Stack

`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `XGBoost` · `Matplotlib` · `Seaborn` · `Jupyter`

---

## 📄 License

This project was developed for academic purposes at GIU. Feel free to fork and build on it.
