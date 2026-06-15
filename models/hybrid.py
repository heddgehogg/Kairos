import logging
import math
from typing import Optional

import pandas as pd

from .content_based import ContentBasedRecommender
from .collaborative_filtering import CollaborativeFilteringRecommender

logger = logging.getLogger(__name__)

class HybridRecommender:
    def __init__(
        self,
        cb_model: ContentBasedRecommender,
        cf_model: CollaborativeFilteringRecommender,
        cf_weight_max: float = 0.6,
        warm_threshold: int = 20,
        diversity_penalty: float = 0.1,
    ):
        self.cb_model = cb_model
        self.cf_model = cf_model
        self.cf_weight_max = cf_weight_max
        self.warm_threshold = warm_threshold
        self.diversity_penalty = diversity_penalty

    # ─────────────────────────────────────────
    # MAIN ENTRY POINT
    # ─────────────────────────────────────────
    def recommend(
        self,
        user_id: str,
        user_profile: Optional[dict] = None,
        rated_lesson_ids: Optional[list] = None,
        ratings: Optional[list] = None,
        top_n: int = 10,
        exclude_lesson_ids: Optional[list] = None,
        candidate_pool: int = 50,
    ) -> pd.DataFrame:
        exclude = list(exclude_lesson_ids or [])
        interaction_count = self.cf_model._interaction_counts.get(user_id, 0)
        cf_weight = self._dynamic_cf_weight(interaction_count)
        is_cold   = not self.cf_model.is_warm_user(user_id)

        logger.debug(
            "User %s | interactions=%d | cf_weight=%.2f | cold=%s",
            user_id, interaction_count, cf_weight, is_cold,
        )
        if rated_lesson_ids and len(rated_lesson_ids) > 0:
            cb_df = self.cb_model.recommend_from_history(
                rated_lesson_ids, ratings, top_n=candidate_pool, exclude_lesson_ids=exclude
            )
            # Fallback: if history lookup failed (IDs not in corpus), use profile
            if cb_df.empty and user_profile:
                logger.debug("History lookup empty, falling back to profile-based CB.")
                cb_df = self.cb_model.recommend_from_profile(
                    user_profile, top_n=candidate_pool, exclude_lesson_ids=exclude
                )
        elif user_profile:
            cb_df = self.cb_model.recommend_from_profile(
                user_profile, top_n=candidate_pool, exclude_lesson_ids=exclude
            )
        else:
            cb_df = pd.DataFrame()

        if not is_cold:
            cf_df = self.cf_model.recommend(user_id, top_n=candidate_pool, exclude_lesson_ids=exclude)
        else:
            cf_df = pd.DataFrame()

        merged = self._merge_scores(cb_df, cf_df, cf_weight)

        if self.diversity_penalty > 0 and not merged.empty:
            merged = self._apply_diversity(merged, top_n)
        else:
            merged = merged.head(top_n)

        merged["USER_ID"] = user_id
        merged["CF_WEIGHT"] = cf_weight
        merged["IS_COLD"] = is_cold

        return merged.reset_index(drop=True)

    # ─────────────────────────────────────────
    # BATCH RECOMMENDATIONS
    # ─────────────────────────────────────────
    def recommend_batch(
        self,
        user_ids: list,
        user_profiles: Optional[dict] = None,
        user_histories: Optional[dict] = None,
        top_n: int = 10,
        candidate_pool: int = 50,
    ) -> pd.DataFrame:
        all_recs = []
        total = len(user_ids)

        for i, uid in enumerate(user_ids):
            if i % 1000 == 0:
                logger.info("Batch progress: %d / %d", i, total)

            profile = (user_profiles or {}).get(uid)
            history = (user_histories or {}).get(uid, {})
            lesson_ids = history.get("lesson_ids", [])
            rated_lesson_ids = history.get("rated_lesson_ids", lesson_ids)
            ratings = history.get("ratings", [])

            recs = self.recommend(
                user_id=uid,
                user_profile=profile,
                rated_lesson_ids=rated_lesson_ids if rated_lesson_ids else None,
                ratings=ratings if ratings else None,
                top_n=top_n,
                exclude_lesson_ids=lesson_ids if lesson_ids else None,
                candidate_pool=candidate_pool,
            )
            all_recs.append(recs)

        return pd.concat(all_recs, ignore_index=True) if all_recs else pd.DataFrame()

    # ─────────────────────────────────────────
    # INTERNAL
    # ─────────────────────────────────────────
    def _dynamic_cf_weight(self, interaction_count: int) -> float:
        if interaction_count == 0:
            return 0.0
        x = (interaction_count - self.warm_threshold / 2) / (self.warm_threshold / 6)
        sigmoid = 1 / (1 + math.exp(-x))
        return round(sigmoid * self.cf_weight_max, 4)

    def _merge_scores(
        self,
        cb_df: pd.DataFrame,
        cf_df: pd.DataFrame,
        cf_weight: float,
    ) -> pd.DataFrame:
        cb_weight = 1 - cf_weight

        if cb_df.empty and cf_df.empty:
            return pd.DataFrame()

        if cb_df.empty:
            cf_df = cf_df.copy()
            cf_df["CB_SCORE"] = 0.0
            cf_df["HYBRID_SCORE"] = cf_df["CF_SCORE"] * cf_weight
            cf_df = cf_df.sort_values("HYBRID_SCORE", ascending=False)
            lessons_df = getattr(self.cb_model, "_lessons_df", None)
            if lessons_df is not None:
                meta_cols = [c for c in ["LESSON_ID", "TITLE", "THEOLOGICAL_DOMAIN",
                                         "DIFFICULTY_LEVEL", "SPIRITUAL_DEPTH", "READING_TIME_MIN"]
                             if c in lessons_df.columns]
                cf_df = cf_df.merge(lessons_df[meta_cols], on="LESSON_ID", how="inner")
            return cf_df.sort_values("HYBRID_SCORE", ascending=False).reset_index(drop=True)

        if cf_df.empty:
            cb_df["CF_SCORE"] = 0.0
            cb_df["HYBRID_SCORE"] = cb_df["CB_SCORE"] * cb_weight
            return (cb_df
                    .sort_values("HYBRID_SCORE", ascending=False)
                    .drop_duplicates(subset=["LESSON_ID"], keep="first")
                    .reset_index(drop=True))

        merged = pd.merge(
            cb_df[["LESSON_ID", "TITLE", "THEOLOGICAL_DOMAIN", "DIFFICULTY_LEVEL",
                   "SPIRITUAL_DEPTH", "READING_TIME_MIN", "CB_SCORE"]],
            cf_df[["LESSON_ID", "CF_SCORE"]],
            on="LESSON_ID",
            how="outer",
        )
        merged["CB_SCORE"] = merged["CB_SCORE"].fillna(0.0)
        merged["CF_SCORE"] = merged["CF_SCORE"].fillna(0.0)

        # CF-only rows (TITLE is NaN): fill metadata from lessons corpus or drop ghost lessons
        cf_only_mask = merged["TITLE"].isna()
        if cf_only_mask.any():
            lessons_df = getattr(self.cb_model, "_lessons_df", None)
            if lessons_df is not None:
                meta_cols = [c for c in ["TITLE", "THEOLOGICAL_DOMAIN", "DIFFICULTY_LEVEL",
                                         "SPIRITUAL_DEPTH", "READING_TIME_MIN"]
                             if c in lessons_df.columns]
                cf_only = (merged.loc[cf_only_mask, ["LESSON_ID", "CB_SCORE", "CF_SCORE"]]
                           .merge(lessons_df[["LESSON_ID"] + meta_cols], on="LESSON_ID", how="inner"))
                merged = pd.concat([merged[~cf_only_mask], cf_only], ignore_index=True)
            else:
                merged = merged.dropna(subset=["TITLE"])

        merged["HYBRID_SCORE"] = (cb_weight * merged["CB_SCORE"] +
                                   cf_weight  * merged["CF_SCORE"])
        merged = (merged
                  .sort_values("HYBRID_SCORE", ascending=False)
                  .drop_duplicates(subset=["LESSON_ID"], keep="first")
                  .reset_index(drop=True))
        return merged

    def _apply_diversity(self, df: pd.DataFrame, top_n: int) -> pd.DataFrame:
        if "THEOLOGICAL_DOMAIN" not in df.columns:
            return df.head(top_n)

        selected = []
        selected_domains: set = set()
        candidates = df.copy()

        while len(selected) < top_n and not candidates.empty:
            candidates = candidates.copy()
            same_domain = candidates["THEOLOGICAL_DOMAIN"].isin(selected_domains).astype(float)
            candidates["_PENALISED"] = candidates["HYBRID_SCORE"] * (1 - self.diversity_penalty * same_domain)
            best_idx = candidates["_PENALISED"].idxmax()
            best_row = candidates.loc[best_idx]
            selected.append(best_row)
            selected_domains.add(best_row.get("THEOLOGICAL_DOMAIN"))
            candidates = candidates.drop(index=best_idx)

        return pd.DataFrame(selected).drop(columns=["_PENALISED"], errors="ignore")