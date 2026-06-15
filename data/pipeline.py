import ast
import json
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. LOADERS
# ─────────────────────────────────────────────
# lessons
def load_lessons(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    airbyte_cols = [c for c in df.columns if c.startswith("_AIRBYTE")]
    df = df.drop(columns=airbyte_cols, errors="ignore")

    def parse_embedding(s):
        try:
            return np.array(json.loads(s), dtype=np.float32)
        except Exception:
            return None

    logger.info("Parsing lesson embeddings …")
    df["EMBEDDING_VEC"] = df["EMBEDDING"].apply(parse_embedding)
    df = df.drop(columns=["EMBEDDING"], errors="ignore")

    def flatten_payload(s):
        try:
            return json.loads(s)
        except Exception:
            return {}

    payloads = df["CONTENT_PAYLOAD"].apply(flatten_payload)
    payload_df = pd.json_normalize(payloads).add_prefix("CP_")
    df = pd.concat([df.drop(columns=["CONTENT_PAYLOAD"]), payload_df], axis=1)

    list_cols = ["EMOTIONAL_TONE", "TRIGGERS_NEEDS", "SATISFIES_NEEDS", "TARGET_EMOTIONAL_STATES"]
    for col in list_cols:
        if col in df.columns:
            df[col] = df[col].apply(_safe_parse_list)

    for col in ["SPIRITUAL_DEPTH", "DIFFICULTY_LEVEL", "SPIRITUAL_CHALLENGE_LEVEL", "THEOLOGICAL_DOMAIN"]:
        if col in df.columns:
            df[col] = df[col].str.lower().str.strip()

    if "LESSON_ID" in df.columns and "ID" in df.columns:
        df = df.drop(columns=["ID"])
    elif "ID" in df.columns:
        df = df.rename(columns={"ID": "LESSON_ID"})
    logger.info("Lessons loaded: %d rows", len(df))
    return df

# users
def load_users(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    airbyte_cols = [c for c in df.columns if c.startswith("_AIRBYTE")]
    df = df.drop(columns=airbyte_cols, errors="ignore")
    df["CREATED_AT"] = pd.to_datetime(df["CREATED_AT"], utc=True, errors="coerce")
    logger.info("Users loaded: %d rows", len(df))
    return df

# onboarding
def load_onboarding(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    airbyte_cols = [c for c in df.columns if c.startswith("_AIRBYTE")]
    df = df.drop(columns=airbyte_cols, errors="ignore")

    if "ANSWERS" not in df.columns:
        rename = {}
        for col in df.columns:
            if col.upper() not in ("USER_ID", "CREATED_AT", "QUIZ_ID"):
                upper = col.upper()
                if not upper.startswith("OB_"):
                    rename[col] = "OB_" + upper
                else:
                    rename[col] = upper
        df = df.rename(columns=rename)
        df["CREATED_AT"] = pd.to_datetime(df["CREATED_AT"], utc=True, errors="coerce")
        logger.info("Onboarding loaded (flat format): %d rows", len(df))
        return df

    def parse_answers(s):
        try:
            d = json.loads(s) if isinstance(s, str) else {}
        except Exception:
            d = {}
        normalised = {}
        for k, v in d.items():
            canonical = _canonical_ob_key(k)
            if v:
                normalised[canonical] = v
        return normalised

    answers_parsed = df["ANSWERS"].apply(parse_answers)
    answers_df = pd.json_normalize(answers_parsed).add_prefix("OB_")
    df = pd.concat([df.drop(columns=["ANSWERS"]), answers_df], axis=1)

    df["CREATED_AT"] = pd.to_datetime(df["CREATED_AT"], utc=True, errors="coerce")
    logger.info("Onboarding loaded (JSON format): %d rows", len(df))
    return df

# interactions
def load_interactions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    airbyte_cols = [c for c in df.columns if c.startswith("_AIRBYTE")]
    df = df.drop(columns=airbyte_cols, errors="ignore")

    df["UNLOCKED_AT"] = pd.to_datetime(df["UNLOCKED_AT"], utc=True, errors="coerce")
    df["UPDATED_AT"]  = pd.to_datetime(df["UPDATED_AT"],  utc=True, errors="coerce")

    df["RATING"] = pd.to_numeric(df["RATING"], errors="coerce")

    df["IMPLICIT_SIGNAL"] = df["UNLOCKED_AT"].notna().astype(np.int8)

    df = (df
          .sort_values("UNLOCKED_AT", ascending=False)
          .drop_duplicates(subset=["USER_ID", "LESSON_ID"])
          .reset_index(drop=True))

    logger.info("Interactions loaded: %d rows, %d unique users, %d unique lessons",
                len(df), df["USER_ID"].nunique(), df["LESSON_ID"].nunique())
    return df

# ─────────────────────────────────────────────
# 2. MERGE 
# ─────────────────────────────────────────────
def build_master_interactions(
    interactions: pd.DataFrame,
    lessons: pd.DataFrame,
    users: pd.DataFrame,
    onboarding: pd.DataFrame,
) -> pd.DataFrame:
    lesson_meta = lessons.drop(columns=["EMBEDDING_VEC", "LESSON_ID"], errors="ignore")
    df = interactions.merge(
        lessons[["LESSON_ID"] + [c for c in lessons.columns if c != "LESSON_ID" and c != "EMBEDDING_VEC"]],
        on="LESSON_ID", how="left"
    )

    if "CREATED_AT" in users.columns:
        users = users.rename(columns={"CREATED_AT": "USER_CREATED_AT"})
    df = df.merge(users, on="USER_ID", how="left")

    ob_cols = ["USER_ID"] + [c for c in onboarding.columns if c.startswith("OB_")]
    df = df.merge(onboarding[ob_cols], on="USER_ID", how="left")

    logger.info("Master interactions built: %d rows × %d cols", *df.shape)
    return df


def build_user_profiles(onboarding: pd.DataFrame, interactions: pd.DataFrame, lessons: pd.DataFrame) -> pd.DataFrame:
    ob_cols = ["USER_ID"] + [c for c in onboarding.columns if c.startswith("OB_")]
    profiles = onboarding[ob_cols].copy()

    rated = interactions[interactions["RATING"].notna()]
    if not rated.empty:
        rated = rated.merge(lessons[["LESSON_ID", "THEOLOGICAL_DOMAIN", "SPIRITUAL_DEPTH",
                                     "DIFFICULTY_LEVEL", "EMOTIONAL_TONE"]], on="LESSON_ID", how="left")

        top_domain = (rated.groupby(["USER_ID", "THEOLOGICAL_DOMAIN"])["RATING"]
                           .mean().reset_index()
                           .sort_values("RATING", ascending=False)
                           .drop_duplicates("USER_ID")
                           .rename(columns={"THEOLOGICAL_DOMAIN": "BEH_TOP_DOMAIN",
                                            "RATING": "BEH_TOP_DOMAIN_AVG_RATING"}))
        profiles = profiles.merge(top_domain, on="USER_ID", how="left")

        avg_rating = (rated.groupby("USER_ID")["RATING"]
                           .mean().reset_index()
                           .rename(columns={"RATING": "BEH_AVG_RATING"}))
        profiles = profiles.merge(avg_rating, on="USER_ID", how="left")

        lesson_count = (interactions.groupby("USER_ID")["LESSON_ID"]
                                    .count().reset_index()
                                    .rename(columns={"LESSON_ID": "BEH_LESSON_COUNT"}))
        profiles = profiles.merge(lesson_count, on="USER_ID", how="left")

    logger.info("User profiles built: %d rows", len(profiles))
    return profiles

# ─────────────────────────────────────────────
# 3. TRAIN
# ─────────────────────────────────────────────
def temporal_split(
    interactions: pd.DataFrame,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    timestamp_col: str = "UNLOCKED_AT",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    df = interactions.dropna(subset=[timestamp_col]).sort_values(timestamp_col)
    n = len(df)
    train_end = int(n * (1 - val_frac - test_frac))
    val_end = int(n * (1 - test_frac))

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    logger.info("Split → train: %d | val: %d | test: %d", len(train), len(val), len(test))
    return train, val, test

# ─────────────────────────────────────────────
# 4. HELPERS
# ─────────────────────────────────────────────
def _safe_parse_list(val) -> list:
    if isinstance(val, list):
        return val
    try:
        result = json.loads(val)
        return result if isinstance(result, list) else []
    except Exception:
        return []


_OB_KEY_MAP = {
    "ob_goals": "ob_goal",
    "ob_motivations": "ob_motivation",
    "ob_connects_god": "ob_connect_with_god",
    "ob_quote_hard": "ob_quote_hard_quotes",
    "ob_quote_spend_time": "ob_quote_spending_time",
}

def _canonical_ob_key(key: str) -> str:
    return _OB_KEY_MAP.get(key, key)