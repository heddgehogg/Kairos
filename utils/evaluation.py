import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def precision_at_k(recommendations: pd.DataFrame, ground_truth: pd.DataFrame, k: int = 10) -> float:
    gt = ground_truth.groupby("USER_ID")["LESSON_ID"].apply(set).to_dict()
    scores = []
    for uid, group in recommendations.groupby("USER_ID"):
        top_k = group["LESSON_ID"].head(k).tolist()
        relevant = gt.get(uid, set())
        hits = sum(1 for l in top_k if l in relevant)
        scores.append(hits / k)
    return float(np.mean(scores)) if scores else 0.0


def recall_at_k(recommendations: pd.DataFrame, ground_truth: pd.DataFrame, k: int = 10) -> float:
    gt = ground_truth.groupby("USER_ID")["LESSON_ID"].apply(set).to_dict()
    scores = []
    for uid, group in recommendations.groupby("USER_ID"):
        top_k = group["LESSON_ID"].head(k).tolist()
        relevant = gt.get(uid, set())
        if not relevant:
            continue
        hits = sum(1 for l in top_k if l in relevant)
        scores.append(hits / len(relevant))
    return float(np.mean(scores)) if scores else 0.0


def ndcg_at_k(recommendations: pd.DataFrame, ground_truth: pd.DataFrame, k: int = 10) -> float:
    has_rating = "RATING" in ground_truth.columns

    if has_rating:
        gt = ground_truth.groupby("USER_ID").apply(
            lambda x: dict(zip(x["LESSON_ID"], x["RATING"].fillna(1)))
        ).to_dict()
    else:
        gt = ground_truth.groupby("USER_ID")["LESSON_ID"].apply(
            lambda x: {l: 1 for l in x}
        ).to_dict()

    scores = []
    for uid, group in recommendations.groupby("USER_ID"):
        top_k   = group["LESSON_ID"].head(k).tolist()
        rel_map = gt.get(uid, {})
        if not rel_map:
            continue

        dcg = sum(
            rel_map.get(l, 0) / np.log2(i + 2)
            for i, l in enumerate(top_k)
        )

        ideal_rels = sorted(rel_map.values(), reverse=True)[:k]
        idcg = sum(r / np.log2(i + 2) for i, r in enumerate(ideal_rels))

        scores.append(dcg / idcg if idcg > 0 else 0.0)

    return float(np.mean(scores)) if scores else 0.0


def catalogue_coverage(recommendations: pd.DataFrame, total_lessons: int) -> float:
    unique_recommended = recommendations["LESSON_ID"].nunique()
    return unique_recommended / total_lessons if total_lessons > 0 else 0.0


def novelty(recommendations: pd.DataFrame, interactions: pd.DataFrame) -> float:
    pop = interactions["LESSON_ID"].value_counts(normalize=True).to_dict()
    scores = []
    for lid in recommendations["LESSON_ID"]:
        p = pop.get(lid, 1e-10)
        scores.append(-np.log2(p))
    return float(np.mean(scores)) if scores else 0.0


def evaluate_model(
    model,
    test_interactions: pd.DataFrame,
    user_profiles: Optional[dict],
    user_histories: Optional[dict],
    k: int = 10,
    sample_users: int = 1000,
) -> dict:
    test_users = test_interactions["USER_ID"].unique()
    if len(test_users) > sample_users:
        rng = np.random.default_rng(42)
        test_users = rng.choice(test_users, size=sample_users, replace=False)

    logger.info("Evaluating on %d users …", len(test_users))

    recs = model.recommend_batch(
        user_ids=list(test_users),
        user_profiles=user_profiles,
        user_histories=user_histories,
        top_n=k,
    )

    gt = test_interactions[test_interactions["USER_ID"].isin(test_users)]

    total_lessons = test_interactions["LESSON_ID"].nunique()

    metrics = {
        f"precision@{k}": precision_at_k(recs, gt, k),
        f"recall@{k}": recall_at_k(recs, gt, k),
        f"ndcg@{k}": ndcg_at_k(recs, gt, k),
        "coverage": catalogue_coverage(recs, total_lessons),
        "novelty": novelty(recs, test_interactions),
    }

    logger.info("Evaluation results: %s", metrics)
    return metrics
