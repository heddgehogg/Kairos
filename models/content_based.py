import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder

logger = logging.getLogger(__name__)

class ContentBasedRecommender:
    OB_TOPIC_TO_DOMAIN = {
        "spiritual_growth": "discipleship",
        "trauma_healing": "healing",
        "purify_your_mind": "identity",
        "attachment_healing": "relationships",
        "bible_in_year": "discipleship",
        "discipleship": "discipleship",
        "forgiveness": "forgiveness",
        "identity": "identity",
        "grace": "grace",
        "suffering": "suffering",
        "relationships": "relationships",
        "healing": "healing",
        "parenting": "parenting",
        "stewardship": "stewardship",
    }

    OB_GOAL_TO_DOMAIN = {
        "closer_to_god": "discipleship",
        "understand_bible": "wisdom",
        "find_peace": "suffering",
        "strengthen_my_faith": "faith",
        "god_word": "discipleship",
        "make_wiser": "wisdom",
    }

    OB_EXPERIENCE_TO_DIFFICULTY = {
        "quick_practical": "beginner",
        "uplifting_inspiring": "beginner",
        "guided_structured": "intermediate",
        "deep_thought": "advanced",
    }

    OB_MOTIVATION_TO_DOMAIN = {
        "deeper_meaning": "wisdom",
        "better_person": "discipleship",
        "overcoming_struggles": "suffering",
        "helping_others": "relationships",
    }

    OB_TIME_TO_MAX_MIN = {
        "five_minutes": 5,
        "ten_minutes": 10,
        "fifteen_minutes": 15,
        "twenty_minutes": 20,
    }

    def __init__(self, alpha: float = 0.7):
        self.alpha = alpha
        self._item_matrix: Optional[np.ndarray] = None
        self._lesson_ids: Optional[np.ndarray] = None
        self._lessons_df: Optional[pd.DataFrame] = None
        self._mlb_emotions: Optional[MultiLabelBinarizer] = None
        self._ohe_meta: Optional[OneHotEncoder] = None
        self._embedding_dim: int = 0
        self._meta_dim: int = 0
        self._rt_max: float = 1.0
        # Pre-extracted column arrays for fast _rank() access (avoids DataFrame.iloc)
        self._titles: Optional[np.ndarray] = None
        self._domains: Optional[np.ndarray] = None
        self._difficulties: Optional[np.ndarray] = None
        self._depths: Optional[np.ndarray] = None
        self._reading_times: Optional[np.ndarray] = None

    # ─────────────────────────────────────────
    # FIT
    # ─────────────────────────────────────────
    def fit(self, lessons: pd.DataFrame) -> "ContentBasedRecommender":
        df = lessons.dropna(subset=["EMBEDDING_VEC"]).reset_index(drop=True)
        self._lessons_df = df
        self._lesson_ids = df["LESSON_ID"].values

        emb_matrix = np.vstack(df["EMBEDDING_VEC"].values).astype(np.float32)
        emb_matrix = _l2_normalize(emb_matrix)
        self._embedding_dim = emb_matrix.shape[1]

        meta_matrix = self._build_meta_matrix(df, fit=True)
        meta_matrix = _l2_normalize(meta_matrix)
        self._meta_dim = meta_matrix.shape[1]

        combined = np.hstack([
            self.alpha * emb_matrix,
            (1 - self.alpha) * meta_matrix,
        ]).astype(np.float32)
        self._item_matrix = _l2_normalize(combined)

        # Pre-extract columns as numpy arrays — avoids slow DataFrame.iloc in _rank()
        def _col(name):
            return df[name].values if name in df.columns else np.full(len(df), None, dtype=object)
        self._titles = _col("TITLE")
        self._domains = _col("THEOLOGICAL_DOMAIN")
        self._difficulties = _col("DIFFICULTY_LEVEL")
        self._depths = _col("SPIRITUAL_DEPTH")
        self._reading_times = df["READING_TIME_MIN"].values if "READING_TIME_MIN" in df.columns else np.zeros(len(df))

        logger.info(
            "ContentBased fitted: %d lessons | emb_dim=%d | meta_dim=%d",
            len(df), self._embedding_dim, self._meta_dim,
        )
        return self

    # ─────────────────────────────────────────
    # RECOMMEND
    # ─────────────────────────────────────────
    def recommend_from_profile(
        self,
        user_profile: dict,
        top_n: int = 10,
        exclude_lesson_ids: Optional[list] = None,
    ) -> pd.DataFrame:
        self._check_fitted()
        query_vec = self._profile_to_query_vector(user_profile)
        return self._rank(query_vec, top_n, exclude_lesson_ids)

    def recommend_from_history(
        self,
        rated_lesson_ids: list,
        ratings: Optional[list] = None,
        top_n: int = 10,
        exclude_lesson_ids: Optional[list] = None,
    ) -> pd.DataFrame:
        self._check_fitted()
        query_vec = self.compute_query_from_history(rated_lesson_ids, ratings)
        if query_vec is None:
            return pd.DataFrame()
        exclude = list(exclude_lesson_ids or []) + list(rated_lesson_ids)
        return self._rank(query_vec, top_n, exclude)

    def compute_query_from_history(
        self,
        rated_lesson_ids: list,
        ratings: Optional[list] = None,
    ) -> Optional[np.ndarray]:
        """Return the CB query vector for a user history (no ranking). Returns None if no match."""
        rated_set = set(rated_lesson_ids)
        lid_to_rating = dict(zip(rated_lesson_ids, ratings)) if ratings else {}

        matched = [(i, lid) for i, lid in enumerate(self._lesson_ids) if lid in rated_set]
        if not matched:
            return None

        idx = [i for i, _ in matched]
        vecs = self._item_matrix[idx]

        if lid_to_rating:
            raw = np.array(
                [lid_to_rating.get(lid, 3.0) for _, lid in matched],
                dtype=np.float32,
            )
            centred = raw - raw.mean()
            pos_mask = centred > 0
            if pos_mask.any():
                pos_weights = centred[pos_mask]
                total = pos_weights.sum()
                return (vecs[pos_mask] * (pos_weights / total).reshape(-1, 1)).sum(axis=0)
            else:
                return vecs.mean(axis=0)
        return vecs.mean(axis=0)

    def rank_from_scores(
        self,
        scores: np.ndarray,
        top_n: int,
        exclude_lesson_ids: Optional[list] = None,
    ) -> pd.DataFrame:
        """Rank using pre-computed similarity scores (e.g. from batch matrix multiply)."""
        return self._rank_from_scores_impl(scores, top_n, exclude_lesson_ids)

    def get_similar_lessons(self, lesson_id: str, top_n: int = 10) -> pd.DataFrame:
        self._check_fitted()
        idx_list = np.where(self._lesson_ids == lesson_id)[0]
        if len(idx_list) == 0:
            raise ValueError(f"lesson_id '{lesson_id}' not found.")
        query_vec = self._item_matrix[idx_list[0]]
        return self._rank(query_vec, top_n + 1, exclude_lesson_ids=[lesson_id]).head(top_n)

    # ─────────────────────────────────────────
    # INTERNAL
    # ─────────────────────────────────────────
    def _rank(
        self,
        query_vec: np.ndarray,
        top_n: int,
        exclude_lesson_ids: Optional[list],
    ) -> pd.DataFrame:
        q = query_vec.astype(np.float32)
        norm = np.linalg.norm(q)
        if norm > 0:
            q /= norm
        scores = self._item_matrix @ q
        return self._rank_from_scores_impl(scores, top_n, exclude_lesson_ids)

    def _rank_from_scores_impl(
        self,
        scores: np.ndarray,
        top_n: int,
        exclude_lesson_ids: Optional[list],
    ) -> pd.DataFrame:
        ranked_idx = np.argsort(scores)[::-1]
        exclude_set = set(exclude_lesson_ids or [])
        seen_ids: set = set()
        seen_titles: set = set()
        results = []
        for i in ranked_idx:
            lid = self._lesson_ids[i]
            if lid in exclude_set or lid in seen_ids:
                continue
            title = self._titles[i]
            if title and title in seen_titles:
                continue
            seen_ids.add(lid)
            if title:
                seen_titles.add(title)
            results.append({
                "LESSON_ID": lid,
                "TITLE": title,
                "THEOLOGICAL_DOMAIN": self._domains[i],
                "DIFFICULTY_LEVEL": self._difficulties[i],
                "SPIRITUAL_DEPTH": self._depths[i],
                "READING_TIME_MIN": self._reading_times[i],
                "CB_SCORE": float(scores[i]),
            })
            if len(results) == top_n:
                break

        return pd.DataFrame(results)

    def _build_meta_matrix(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        parts = []

        cat_cols = [c for c in ["THEOLOGICAL_DOMAIN", "DIFFICULTY_LEVEL",
                                  "SPIRITUAL_DEPTH", "SPIRITUAL_CHALLENGE_LEVEL"]
                    if c in df.columns]
        if cat_cols:
            cat_data = df[cat_cols].fillna("unknown")
            if fit:
                self._ohe_meta = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                cat_enc = self._ohe_meta.fit_transform(cat_data)
            else:
                cat_enc = self._ohe_meta.transform(cat_data)
            parts.append(cat_enc)

        if "EMOTIONAL_TONE" in df.columns:
            tones = df["EMOTIONAL_TONE"].apply(lambda x: x if isinstance(x, list) else [])
            if fit:
                self._mlb_emotions = MultiLabelBinarizer()
                tone_enc = self._mlb_emotions.fit_transform(tones)
            else:
                tone_enc = self._mlb_emotions.transform(tones)
            parts.append(tone_enc.astype(np.float32))

        if "READING_TIME_MIN" in df.columns:
            rt = df["READING_TIME_MIN"].fillna(10).values.reshape(-1, 1).astype(np.float32)
            if fit:
                self._rt_max = float(rt.max()) or 1.0
            rt = rt / self._rt_max
            parts.append(rt)

        return np.hstack(parts).astype(np.float32) if parts else np.zeros((len(df), 1), dtype=np.float32)

    def _profile_to_query_vector(self, profile: dict) -> np.ndarray:
        p = {}
        for k, v in profile.items():
            key = k
            while key.upper().startswith("OB_"):
                key = key[3:]
            p[key.lower()] = v

        domain = (self.OB_TOPIC_TO_DOMAIN.get(p.get("topic", "")) or
                  self.OB_GOAL_TO_DOMAIN.get(p.get("goal", "")) or
                  self.OB_MOTIVATION_TO_DOMAIN.get(p.get("motivation", "")))

        diff = self.OB_EXPERIENCE_TO_DIFFICULTY.get(p.get("ideal_experience", ""))

        max_min = self.OB_TIME_TO_MAX_MIN.get(p.get("time", ""))

        df = self._lessons_df

        if domain and "THEOLOGICAL_DOMAIN" in df.columns:
            domain_mask = (df["THEOLOGICAL_DOMAIN"] == domain).values
        else:
            domain_mask = np.ones(len(df), dtype=bool)

        seed_idx = np.where(domain_mask)[0]

        if diff and "DIFFICULTY_LEVEL" in df.columns and len(seed_idx) > 5:
            diff_mask = (df["DIFFICULTY_LEVEL"] == diff).values
            refined = np.where(domain_mask & diff_mask)[0]
            if len(refined) >= 3:
                seed_idx = refined

        if max_min is not None and "READING_TIME_MIN" in df.columns and len(seed_idx) > 5:
            time_mask = (df["READING_TIME_MIN"].fillna(max_min) <= max_min).values
            current = seed_idx
            refined = np.array([i for i in current if time_mask[i]])
            if len(refined) >= 3:
                seed_idx = refined

        if len(seed_idx) == 0:
            seed_idx = np.arange(len(self._lesson_ids))

        if len(seed_idx) > 30:
            vecs = self._item_matrix[seed_idx]
            centroid  = vecs.mean(axis=0)
            sim = vecs @ centroid   
            top_local = np.argsort(sim)[::-1][:30]
            seed_idx = seed_idx[top_local]

        return self._item_matrix[seed_idx].mean(axis=0)

    def _check_fitted(self):
        if self._item_matrix is None:
            raise RuntimeError("Model not fitted. Call .fit(lessons) first.")


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return matrix / norms