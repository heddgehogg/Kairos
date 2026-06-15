import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# BATCH SERVER
# ─────────────────────────────────────────────

class BatchServer:

    def __init__(
        self,
        model,                          
        user_profiles: dict,      
        user_histories: dict,    
        top_n: int = 20,
        candidate_pool: int = 100,
        chunk_size: int = 10_000,
    ):
        self.model = model
        self.user_profiles = user_profiles
        self.user_histories = user_histories
        self.top_n = top_n
        self.candidate_pool = candidate_pool
        self.chunk_size = chunk_size

    def run(self, user_ids: list, output_path: str) -> pd.DataFrame:

        logger.info("Starting batch recommendation for %d users …", len(user_ids))
        t0 = time.time()

        chunks = [user_ids[i:i + self.chunk_size]
                  for i in range(0, len(user_ids), self.chunk_size)]

        all_dfs = []
        for chunk_num, chunk in enumerate(chunks):
            logger.info("Processing chunk %d / %d (%d users)",
                        chunk_num + 1, len(chunks), len(chunk))

            chunk_df = self.model.recommend_batch(
                user_ids=chunk,
                user_profiles=self.user_profiles,
                user_histories=self.user_histories,
                top_n=self.top_n,
                candidate_pool=self.candidate_pool,
            )
            all_dfs.append(chunk_df)

        result = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

        result["BATCH_GENERATED_AT"] = pd.Timestamp.utcnow()
        result["RANK"] = result.groupby("USER_ID").cumcount() + 1

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(output_path, index=False)

        elapsed = time.time() - t0
        logger.info(
            "Batch complete: %d recommendations for %d users in %.1fs → %s",
            len(result), result["USER_ID"].nunique(), elapsed, output_path,
        )
        return result


# ─────────────────────────────────────────────
# REAL-TIME SERVER
# ─────────────────────────────────────────────

class RealTimeServer:

    def __init__(
        self,
        model,
        batch_recs_path: Optional[str] = None,
        top_n: int = 10,
        candidate_pool: int = 50,
        max_latency_ms: float = 200.0,
    ):
        self.model = model
        self.top_n = top_n
        self.candidate_pool = candidate_pool
        self.max_latency_ms = max_latency_ms
        self._batch_cache: Optional[dict] = None  

        if batch_recs_path and Path(batch_recs_path).exists():
            self._load_batch_cache(batch_recs_path)

    def get_recommendations(
        self,
        user_id: str,
        user_profile: Optional[dict] = None,
        rated_lesson_ids: Optional[list] = None,
        ratings: Optional[list] = None,
        exclude_lesson_ids: Optional[list] = None,
        force_live: bool = False,
    ) -> dict:
        t0 = time.time()

        if not force_live and self._batch_cache and user_id in self._batch_cache:
            recs = self._batch_cache[user_id]
            source = "batch_cache"
        else:
            recs_df = self.model.recommend(
                user_id=user_id,
                user_profile=user_profile,
                rated_lesson_ids=rated_lesson_ids,
                ratings=ratings,
                top_n=self.top_n,
                exclude_lesson_ids=exclude_lesson_ids,
                candidate_pool=self.candidate_pool,
            )
            recs = recs_df.to_dict(orient="records") if not recs_df.empty else []
            source = "live"

        latency_ms = (time.time() - t0) * 1000

        if latency_ms > self.max_latency_ms:
            logger.warning("Slow real-time rec: %.1fms for user %s", latency_ms, user_id)

        return {
            "user_id": user_id,
            "recommendations": recs,
            "source": source,
            "latency_ms": round(latency_ms, 2),
            "count": len(recs),
        }

    def refresh_cache(self, new_batch_path: str):
        self._load_batch_cache(new_batch_path)
        logger.info("Batch cache refreshed from %s", new_batch_path)

    def _load_batch_cache(self, path: str):
        logger.info("Loading batch cache from %s …", path)
        df = pd.read_parquet(path)

        self._batch_cache = {}
        for uid, group in df.sort_values("RANK").groupby("USER_ID"):
            self._batch_cache[uid] = group.drop(columns=["USER_ID", "RANK",
                                                          "BATCH_GENERATED_AT"],
                                                 errors="ignore").to_dict(orient="records")

        logger.info("Batch cache loaded: %d users", len(self._batch_cache))
