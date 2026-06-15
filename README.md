# Kairos — Hybrid Recommendation Engine

A Streamlit-based recommendation system for educational content (lessons/courses), combining content-based filtering, collaborative filtering, and a hybrid approach.

## Features

- **Content-Based Filtering** — recommends lessons similar to what a user has already engaged with, using embeddings and metadata (tags, difficulty, duration)
- **Collaborative Filtering** — finds users with similar learning patterns and recommends what they liked
- **Hybrid Model** — blends both approaches with configurable weights, adapting based on how much interaction history a user has
- **Interactive Dashboard** — explore recommendations, evaluate model performance, and tune hyperparameters in real time

## Project Structure

```
├── app.py                  # Streamlit app entry point
├── data/
│   └── pipeline.py         # Data loading and preprocessing
├── models/
│   ├── content_based.py    # Content-based recommender
│   ├── collaborative_filtering.py
│   └── hybrid.py           # Hybrid recommender
├── utils/
│   └── evaluation.py       # Metrics (Precision@K, Recall@K, NDCG)
├── serving/                # Model serving utilities
└── requirements.txt
```

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the app

```bash
streamlit run app.py
```

### 3. Load data & train

1. Click **Load Data** in the sidebar
2. Click **Train Models**
3. Models are automatically saved to `output/models.pkl` and can be reloaded on the next run

## Data

| File | Description |
|---|---|
| `lessons.csv` | Lesson metadata + embeddings |
| `users.csv` | User profiles |
| `interactions.csv` | User–lesson interaction history |
| `onboarding.csv` | Onboarding quiz responses |

## Tech Stack

- [Streamlit](https://streamlit.io/) — UI
- [scikit-learn](https://scikit-learn.org/) — ML models
- [Plotly](https://plotly.com/) — charts
- NumPy / Pandas / SciPy
