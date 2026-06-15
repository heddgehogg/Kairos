import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)

class CollaborativeFilteringRecommender:
    def __init__(
        self,
        n_components: int = 100,
        implicit_weight: float = 0.5,
        min_interactions: int = 3,
        random_state: int = 42,
    ):
        self.n_components = n_components
        self.implicit_weight = implicit_weight
        self.min_interactions = min_interactions
        self.random_state = random_state

        self._svd: Optional[TruncatedSVD] = None
        self._user_factors: Optional[np.ndarray] = None
        self._item_factors: Optional[np.ndarray] = None
        self._user_index: dict = {}  
        self._lesson_index: dict = {} 
        self._index_to_lesson: dict = {}
        self._interaction_counts: dict = {} 

    # ─────────────────────────────────────────
    # FIT
    # ─────────────────────────────────────────
    def fit(self, interactions: pd.DataFrame) -> "CollaborativeFilteringRecommender":
        logger.info("Building user-item matrix …")
        df = interactions.dropna(subset=["USER_ID", "LESSON_ID"]).copy()

        users = df["USER_ID"].unique()
        lessons = df["LESSON_ID"].unique()
        self._user_index = {u: i for i, u in enumerate(users)}
        self._lesson_index = {l: i for i, l in enumerate(lessons)}
        self._index_to_lesson = {i: l for l, i in self._lesson_index.items()}

        n_users, n_items = len(users), len(lessons)
        logger.info("Matrix size: %d users × %d items", n_users, n_items)

        df["SCORE"] = self._compute_score(df)

        self._interaction_counts = df.groupby("USER_ID").size().to_dict()

        row = df["USER_ID"].map(self._user_index).values
        col = df["LESSON_ID"].map(self._lesson_index).values
        val = df["SCORE"].values.astype(np.float32)

        sparse_matrix = csr_matrix((val, (row, col)), shape=(n_users, n_items))

        logger.info("Running TruncatedSVD (n_components=%d) …", self.n_components)
        self._svd = TruncatedSVD(
            n_components=min(self.n_components, min(n_users, n_items) - 1),
            random_state=self.random_state,
        )
        self._user_factors = self._svd.fit_transform(sparse_matrix)        
        self._item_factors = self._svd.components_.T            

        self._user_factors = normalize(self._user_factors, norm="l2")
        self._item_factors = normalize(self._item_factors, norm="l2")

        explained = self._svd.explained_variance_ratio_.sum()
        logger.info(
            "SVD fitted. Explained variance: %.2f%% | user_factors: %s | item_factors: %s",
            explained * 100, self._user_factors.shape, self._item_factors.shape,
        )
        return self

    # ─────────────────────────────────────────
    # RECOMMEND
    # ─────────────────────────────────────────
    def recommend(
        self,
        user_id: str,
        top_n: int = 10,
        exclude_lesson_ids: Optional[list] = None,
    ) -> pd.DataFrame:
        self._check_fitted()

        if user_id not in self._user_index:
            logger.debug("User %s not in training data (cold-start).", user_id)
            return pd.DataFrame()

        if self._interaction_counts.get(user_id, 0) < self.min_interactions:
            logger.debug("User %s has too few interactions for CF.", user_id)
            return pd.DataFrame()

        u_idx = self._user_index[user_id]
        u_vec = self._user_factors[u_idx]

        scores = self._item_factors @ u_vec     

        exclude_set = set(exclude_lesson_ids or [])
        results = []

        ranked_idx = np.argsort(scores)[::-1]
        for i in ranked_idx:
            lid = self._index_to_lesson[i]
            if lid not in exclude_set:
                results.append({"LESSON_ID": lid, "CF_SCORE": float(scores[i])})
                if len(results) == top_n:
                    break

        return pd.DataFrame(results)

    def get_user_factor(self, user_id: str) -> Optional[np.ndarray]:
        if user_id not in self._user_index:
            return None
        return self._user_factors[self._user_index[user_id]]

    def is_warm_user(self, user_id: str) -> bool:
        return (user_id in self._user_index and
                self._interaction_counts.get(user_id, 0) >= self.min_interactions)

    # ─────────────────────────────────────────
    # PERSIST
    # ─────────────────────────────────────────
    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("CF model saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "CollaborativeFilteringRecommender":
        with open(path, "rb") as f:
            return pickle.load(f)

    # ─────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────
    def _compute_score(self, df: pd.DataFrame) -> pd.Series:
        has_rating = df["RATING"].notna() if "RATING" in df.columns else pd.Series(False, index=df.index)
        implicit   = df["IMPLICIT_SIGNAL"].fillna(1).astype(float) if "IMPLICIT_SIGNAL" in df.columns \
                     else pd.Series(1.0, index=df.index)

        rating_norm = (df["RATING"].fillna(3) - 1) / 4.0 if "RATING" in df.columns \
                      else pd.Series(0.5, index=df.index)

        score = np.where(
            has_rating,
            (1 - self.implicit_weight) * rating_norm + self.implicit_weight * implicit,
            implicit,
        )
        return pd.Series(score, index=df.index)

    def _check_fitted(self):
        if self._svd is None:
            raise RuntimeError("Model not fitted. Call .fit(interactions) first.")
