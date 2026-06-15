import json
import pickle
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from data.pipeline import (
    load_interactions, load_lessons, load_onboarding,
    load_users, build_user_profiles, temporal_split,
)
from models.content_based import ContentBasedRecommender
from models.collaborative_filtering import CollaborativeFilteringRecommender
from models.hybrid import HybridRecommender
from utils.evaluation import evaluate_model


_BROKEN_YT_IDS = {
    'J-likHDmgY0','xmFQFAOUS4o','WLjYMFBMgXo','MN5spj_hTc8',
    'OaU0cLMzD2I','zaaVnI-dX8k','7LGtIBCrTR8','GCErb8ZYQUM',
    'vDurY1LVJSE','2G1TNs8zGXE','G06avU6P31A','DKKN88CdYq4',
    'SFayfMV4pCY','fMVNSWuF1_U','NT1LNZGYqF8','qfkPRVSfFmw',
    'MFJkqPXhVGM','KGX_CJIxvkI','Z-UxeLlbGAQ','J7w5OzNu3k0',
    '5YTv-OkC9Ss','5NLgJvGmAaE','7RaE09Pz4yY','SWsGGk1XORA',
    'wr0V1pOXhPc','1D6EHI_DCBQ','oGpNZts7T9c','LNFfFzGLByk',
    'hYip_Vuv8J0',
}

def _yt_available(url) -> bool:
    import re
    if not isinstance(url, str) or not url:
        return False
    m = re.search(r'[?&]v=([^&]+)', url)
    return bool(m) and m.group(1) not in _BROKEN_YT_IDS


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Kairos",
    page_icon="🕊️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stSidebar"] { background: #f8f7ff; }
[data-testid="collapsedControl"] { display: none; }
.topbar {
    background: #085041; border-radius: 12px;
    padding: .75rem 1.25rem; display: flex;
    align-items: center; justify-content: space-between;
    margin-bottom: 1.5rem;
}
.topbar-logo { font-size: 14px; font-weight: 500; color: #E1F5EE; }
.banner {
    background: #E1F5EE; border-radius: 12px;
    padding: 1.25rem 1.5rem; margin-bottom: 1.5rem;
    display: flex; align-items: center; justify-content: space-between;
    border: 0.5px solid #9FE1CB;
}
.banner-title { font-size: 18px; font-weight: 500; color: #085041; margin-bottom: 4px; }
.banner-sub   { font-size: 13px; color: #0F6E56; }
.banner-right { font-size: 12px; color: #0F6E56; text-align: right; }
.rec-card { background:var(--color-background-primary); border:0.5px solid var(--color-border-tertiary); border-radius:12px; padding:1rem; margin-bottom:4px; }
.rec-card.top { border-color:#1D9E75; border-width:1.5px; }
.rec-rank  { font-size:10px; color:var(--color-text-tertiary); margin-bottom:4px; }
.rec-title { font-size:13px; font-weight:500; color:var(--color-text-primary); line-height:1.3; margin-bottom:6px; }
.tag-domain { display:inline-block; padding:2px 8px; border-radius:10px; font-size:11px; font-weight:500; background:#E1F5EE; color:#085041; }
.tag-depth  { display:inline-block; padding:2px 8px; border-radius:10px; font-size:11px; font-weight:500; background:#EEEDFE; color:#3C3489; margin-left:4px; }
.score-row  { display:flex; align-items:baseline; justify-content:space-between; margin-top:8px; }
.score-num  { font-size:20px; font-weight:500; color:#0F6E56; }
.score-time { font-size:11px; color:var(--color-text-tertiary); }
.score-bar  { height:3px; background:#E1F5EE; border-radius:2px; margin-top:6px; }
.score-fill { height:3px; background:#1D9E75; border-radius:2px; }
.src-label  { font-size:11px; margin-top:4px; }
.src-cold   { color:#854F0B; } .src-hybrid { color:#0F6E56; }
.detail-box { background:var(--color-background-secondary); border-radius:8px; padding:1rem; margin-bottom:1rem; font-size:14px; line-height:1.7; }
.creative-card { background:var(--color-background-secondary); border:0.5px solid var(--color-border-tertiary); border-radius:10px; padding:.875rem; margin-bottom:8px; }
.creative-card.best { border-color:#1D9E75; }
button[kind="primary"] { background-color:#085041 !important; border-color:#085041 !important; color:#E1F5EE !important; }
button[kind="primary"]:hover { background-color:#0F6E56 !important; border-color:#0F6E56 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
_DEFAULTS = {
    "lessons":None, "users":None, "onboarding":None, "interactions":None,
    "cb_model":None, "cf_model":None, "hybrid_model":None,
    "user_profiles":None, "user_histories":None,
    "train":None, "test":None, "metrics":None,
    "data_loaded":False, "models_trained":False,
    "selected_lesson":None,
    "last_recs":None, "last_recs_tab":None, "last_recs_meta":None,
    "cold_shown_ids":[],
    "page":"Dashboard",
    "wizard_step":1,
    "wizard":{"goal":None, "topic":None, "experience":None, "time_pref":None, "motivation":None, "connect":None},
}
for _k,_v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k]=_v

# ─────────────────────────────────────────────
# LOADERS
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False, persist="disk")
def _load_lessons(p): return load_lessons(p)
@st.cache_data(show_spinner=False, persist="disk")
def _load_users(p): return load_users(p)
@st.cache_data(show_spinner=False, persist="disk")
def _load_onboarding(p): return load_onboarding(p)
@st.cache_data(show_spinner=False, persist="disk")
def _load_interactions(p): return load_interactions(p)

def _build_lookups(up_df, interactions):
    ob_cols=[c for c in up_df.columns if c.startswith("OB_")]
    profiles={}
    for _,row in up_df.iterrows():
        uid=row["USER_ID"]
        profiles[uid]={c:row[c] for c in ob_cols if pd.notna(row.get(c))}
    histories={}
    for uid,grp in interactions.groupby("USER_ID"):
        rated=grp[grp["RATING"].notna()]
        histories[uid]={
            "lesson_ids": grp["LESSON_ID"].tolist(),
            "rated_lesson_ids": rated["LESSON_ID"].tolist(),
            "ratings": rated["RATING"].tolist(),
        }
    return profiles,histories

# ─────────────────────────────────────────────
# TOP NAV
# ─────────────────────────────────────────────
PAGES=["Dashboard","Recommender","User Explorer","Evaluation"]

def _render_topbar(active_page):
    cols=st.columns([1,1,1,1,1.5])
    for i,p in enumerate(PAGES):
        with cols[i]:
            if st.button(f"**{p}**" if p==active_page else p,key=f"nav_{p}",use_container_width=True):
                st.session_state.page=p
                st.session_state.selected_lesson=None
                st.rerun()
    with cols[4]:
        st.markdown(f'<div style="text-align:right;font-size:12px;color:var(--color-text-tertiary);padding-top:8px">{"● Data" if st.session_state.data_loaded else "○ Data"} &nbsp; {"● Models" if st.session_state.models_trained else "○ Models"}</div>',unsafe_allow_html=True)
    st.markdown('<hr style="border:none;border-top:0.5px solid var(--color-border-tertiary);margin:0.5rem 0 1.5rem">',unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CREATIVE PANEL
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# LESSON DETAIL
# ─────────────────────────────────────────────
def _show_lesson_detail(row, lessons_df):
    lesson_id=row.get("LESSON_ID","")
    if lessons_df is not None and lesson_id:
        match=lessons_df[lessons_df["LESSON_ID"]==lesson_id]
        if not match.empty: row=match.iloc[0]

    title = row.get("TITLE","—")
    domain = row.get("THEOLOGICAL_DOMAIN","")
    depth = row.get("SPIRITUAL_DEPTH","")
    rt = row.get("READING_TIME_MIN","")
    bible = row.get("PRIMARY_BIBLE_BOOK","")
    summary = row.get("ARTICLE","") or ""
    core,points = "",[]
    if not summary:
        payload=row.get("CONTENT_PAYLOAD","")
        if isinstance(payload,str) and payload.strip().startswith("{"):
            try:
                p=json.loads(payload)
                summary=p.get("aiSummary",""); core=p.get("coreInsight","")
                points=p.get("practicalApplicationPoints",[])
            except: summary=payload
    if not core: core=row.get("REFLECTION","") or ""
    prayer  =row.get("PRAYER","") or ""
    one_time=row.get("ONE_TIME_ACTION","") or ""

    col_back,_=st.columns([1,6])
    with col_back:
        st.button("← Back",on_click=lambda:st.session_state.update({"selected_lesson":None}))

    tag_domain=f'<span style="background:#E1F5EE;color:#085041;border-radius:20px;padding:3px 12px;font-size:12px;font-weight:600;">{domain}</span>' if domain else ""
    tag_depth =f'<span style="background:#EEEDFE;color:#3C3489;border-radius:20px;padding:3px 12px;font-size:12px;font-weight:600;">{depth}</span>' if depth else ""
    tag_rt    =f'<span style="background:rgba(255,255,255,0.12);color:#E1F5EE;border-radius:20px;padding:3px 12px;font-size:12px;">{rt} min</span>' if rt else ""
    tag_bible =f'<span style="background:rgba(255,255,255,0.12);color:#E1F5EE;border-radius:20px;padding:3px 12px;font-size:12px;">{bible}</span>' if bible else ""
    st.markdown(f"""
    <div style="background:linear-gradient(120deg,#085041 0%,#0F6E56 60%,#1D9E75 100%);
                border-radius:16px;padding:1.5rem 2rem;margin-bottom:1.25rem;
                display:flex;align-items:center;justify-content:space-between;gap:1rem;">
      <div>
        <div style="font-size:11px;font-weight:600;letter-spacing:.08em;color:#9FE1CB;
                    text-transform:uppercase;margin-bottom:.4rem;">Lesson</div>
        <div style="font-size:24px;font-weight:600;color:#E1F5EE;margin-bottom:.75rem;line-height:1.2;">{title}</div>
        <div style="display:flex;gap:.5rem;flex-wrap:wrap;">{tag_domain}{tag_depth}{tag_rt}{tag_bible}</div>
      </div>
    </div>""",unsafe_allow_html=True)

    col_main,col_side=st.columns([2,1])

    with col_main:
        if summary:
            st.markdown("**Article**")
            st.markdown(summary)
        if core:
            st.markdown("""<div style="height:1px;background:var(--color-border-tertiary);margin:1.25rem 0"></div>""",unsafe_allow_html=True)
            st.markdown(f"""
            <div style="background:#E1F5EE;border-left:3px solid #1D9E75;border-radius:0 10px 10px 0;padding:1rem 1.25rem;">
              <div style="font-size:11px;font-weight:600;color:#0F6E56;text-transform:uppercase;letter-spacing:.07em;margin-bottom:.5rem;">Reflection</div>
              <div style="font-size:14px;color:#085041;line-height:1.75;">{core}</div>
            </div>""",unsafe_allow_html=True)
        if one_time:
            st.markdown(f"""
            <div style="background:#EEEDFE;border-left:3px solid #3C3489;border-radius:0 10px 10px 0;padding:1rem 1.25rem;margin-top:.75rem;">
              <div style="font-size:11px;font-weight:600;color:#3C3489;text-transform:uppercase;letter-spacing:.07em;margin-bottom:.5rem;">One-time action</div>
              <div style="font-size:14px;color:#3C3489;line-height:1.75;">{one_time}</div>
            </div>""",unsafe_allow_html=True)
        if points:
            st.markdown("""<div style="height:1px;background:var(--color-border-tertiary);margin:1.25rem 0"></div>""",unsafe_allow_html=True)
            st.markdown('<div style="font-size:11px;font-weight:600;color:#085041;text-transform:uppercase;letter-spacing:.07em;margin-bottom:.75rem;">Practical steps</div>',unsafe_allow_html=True)
            if isinstance(points,str):
                try: points=json.loads(points)
                except: points=[points]
            for i,pt in enumerate(points):
                st.markdown(f"""<div style="display:flex;gap:.75rem;align-items:flex-start;margin-bottom:.5rem;">
                  <span style="background:#085041;color:#E1F5EE;border-radius:50%;width:20px;height:20px;display:inline-flex;align-items:center;justify-content:center;font-size:10px;font-weight:700;flex-shrink:0;margin-top:2px;">{i+1}</span>
                  <span style="font-size:14px;color:var(--color-text-primary);line-height:1.6;">{pt}</span>
                </div>""",unsafe_allow_html=True)

    with col_side:
        st.markdown("""<div style="font-size:11px;font-weight:600;color:#085041;text-transform:uppercase;
                        letter-spacing:.07em;margin-bottom:.75rem;">Your path continues here</div>""",unsafe_allow_html=True)
        yt_url = row.get("YOUTUBE_URL", "") or ""
        if _yt_available(yt_url):
            st.video(yt_url)
        elif title:
            import urllib.parse
            q = urllib.parse.quote_plus(f"{title} {row.get('THEOLOGICAL_DOMAIN', '')} christian")
            st.markdown(f"""<a href="https://www.youtube.com/results?search_query={q}" target="_blank"
                style="display:flex;align-items:center;gap:.6rem;background:#E1F5EE;border:0.5px solid #9FE1CB;
                       border-radius:10px;padding:.75rem 1rem;text-decoration:none;margin-bottom:.75rem;">
                <span style="font-size:18px;">▶</span>
                <span style="font-size:13px;font-weight:500;color:#085041;">Search video on YouTube</span>
              </a>""",unsafe_allow_html=True)
        if prayer:
            st.markdown('<div style="font-size:11px;font-weight:600;color:#085041;text-transform:uppercase;letter-spacing:.07em;margin-top:1rem;margin-bottom:.5rem;">Prayer</div>',unsafe_allow_html=True)
            with st.expander("Open prayer"):
                st.markdown(prayer)

# ─────────────────────────────────────────────
# CARD GRID
# ─────────────────────────────────────────────
def _open_lesson(lesson_dict):
    st.session_state.selected_lesson=lesson_dict

def _render_rec_grid(recs_df):
    recs_df=recs_df.reset_index(drop=True)
    for chunk_start in range(0,len(recs_df),3):
        chunk=recs_df.iloc[chunk_start:chunk_start+3]
        cols=st.columns(3)
        for col_obj,(idx,row) in zip(cols,chunk.iterrows()):
            rank=int(idx)+1
            title=row.get("TITLE",row.get("LESSON_ID","—"))
            domain=row.get("THEOLOGICAL_DOMAIN","")
            depth=row.get("SPIRITUAL_DEPTH","")
            rt=row.get("READING_TIME_MIN","")
            score=row.get("HYBRID_SCORE",row.get("CB_SCORE",0))
            is_cold=row.get("IS_COLD",True)
            pct=int(score*100)
            lid=str(row.get("LESSON_ID",rank))
            top_cls="top" if rank==1 else ""
            rank_lbl="#1 · best match" if rank==1 else f"#{rank}"
            src_txt="CB only" if is_cold else "Hybrid"
            src_cls="src-cold" if is_cold else "src-hybrid"
            dom_tag=f'<span class="tag-domain">{domain}</span>' if domain else ""
            dep_tag=f'<span class="tag-depth">{depth}</span>' if depth else ""
            with col_obj:
                st.markdown(f"""<div class="rec-card {top_cls}">
                  <div class="rec-rank">{rank_lbl}</div>
                  <div class="rec-title">{title}</div>
                  <div>{dom_tag}{dep_tag}</div>
                  <div class="score-row">
                    <div class="score-num">{pct}%</div>
                    <div class="score-time">{rt} min</div>
                  </div>
                  <div class="score-bar"><div class="score-fill" style="width:{pct}%"></div></div>
                  <div class="src-label {src_cls}">{src_txt}</div>
                </div>""",unsafe_allow_html=True)
                st.button("Open lesson →",key=f"open_{lid}_{rank}",on_click=_open_lesson,args=(row.to_dict(),))

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("**Data sources**")
    lessons_path = st.text_input("Lessons CSV", value="lessons.csv")
    users_path = st.text_input("Users CSV", value="users.csv")
    onboarding_path = st.text_input("Onboarding CSV", value="onboarding.csv")
    interactions_path =st.text_input("Interactions CSV", value="interactions.csv")
    if st.button("Load data",use_container_width=True):
        missing=[p for p in [lessons_path,users_path,onboarding_path,interactions_path] if not Path(p).exists()]
        if missing: st.error("Not found:\n"+"\n".join(missing))
        else:
            with st.spinner("Loading…"):
                st.session_state.lessons =_load_lessons(lessons_path)
                st.session_state.users =_load_users(users_path)
                st.session_state.onboarding =_load_onboarding(onboarding_path)
                st.session_state.interactions =_load_interactions(interactions_path)
                st.session_state.data_loaded = True
            st.success("Data loaded!")
    st.markdown("---")
    st.markdown("**Model settings**")
    cb_alpha = st.slider("CB alpha", 0.0,1.0,0.7,0.05)
    cf_components = st.slider("CF dimensions", 10,200,100,10)
    cf_weight_max = st.slider("Max CF weight", 0.0,1.0,0.6,0.05)
    diversity_pen = st.slider("Diversity penalty",0.0,0.5,0.1,0.05)
    warm_threshold = st.slider("Warm threshold", 5,50,20,5)
    _MODEL_CACHE = Path("output/models.pkl")
    if _MODEL_CACHE.exists():
        if st.button("Load saved models", use_container_width=True):
            with st.spinner("Loading saved models…"):
                with open(_MODEL_CACHE, "rb") as _f:
                    _saved = pickle.load(_f)
                st.session_state.update(_saved)
            st.success("Models loaded from disk!")

    if st.button("Train models",use_container_width=True,type="primary"):
        if not st.session_state.data_loaded: st.error("Load data first.")
        else:
            with st.spinner("Training…"):
                train,val,test=temporal_split(st.session_state.interactions)
                st.session_state.train=train; st.session_state.test=test
                cb=ContentBasedRecommender(alpha=cb_alpha); cb.fit(st.session_state.lessons)
                cf=CollaborativeFilteringRecommender(n_components=cf_components); cf.fit(train)
                hybrid=HybridRecommender(cb_model=cb,cf_model=cf,cf_weight_max=cf_weight_max,warm_threshold=warm_threshold,diversity_penalty=diversity_pen)
                up_df=build_user_profiles(st.session_state.onboarding,st.session_state.interactions,st.session_state.lessons)
                profiles,histories=_build_lookups(up_df,train)
                st.session_state.cb_model=cb; st.session_state.cf_model=cf
                st.session_state.hybrid_model=hybrid
                st.session_state.user_profiles=profiles; st.session_state.user_histories=histories
                st.session_state.models_trained=True
            st.success("Models ready!")
            Path("output").mkdir(exist_ok=True)
            with open(_MODEL_CACHE, "wb") as _f:
                pickle.dump({
                    "cb_model": st.session_state.cb_model,
                    "cf_model": st.session_state.cf_model,
                    "hybrid_model": st.session_state.hybrid_model,
                    "user_profiles": st.session_state.user_profiles,
                    "user_histories": st.session_state.user_histories,
                    "train": st.session_state.train,
                    "test": st.session_state.test,
                    "models_trained": True,
                }, _f)
            st.info("Models saved to disk ✓")

# ─────────────────────────────────────────────
# ROUTING
# ─────────────────────────────────────────────
page=st.session_state.page
if st.session_state.selected_lesson:
    _render_topbar(page)
    _show_lesson_detail(st.session_state.selected_lesson,st.session_state.lessons)
    st.stop()
_render_topbar(page)
_ch = '<div style="font-size:13px;font-weight:600;color:#085041;text-transform:uppercase;letter-spacing:.06em;margin-bottom:.5rem">{}</div>'

# ─────────────────────────────────────────────
# PAGE 1 — DASHBOARD
# ─────────────────────────────────────────────
if page=="Dashboard":
    st.markdown("""<div class="banner"><div class="banner-left">
      <div class="banner-title">Dashboard</div>
      <div class="banner-sub">Dataset health and model overview</div>
    </div></div>""",unsafe_allow_html=True)

    st.markdown("""
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:1.5rem;">

      <div style="grid-column:1/3;background:#085041;border-radius:14px;padding:1.75rem 2rem;">
        <div style="font-size:11px;font-weight:600;letter-spacing:.08em;color:#9FE1CB;text-transform:uppercase;margin-bottom:.5rem;">About Kairos</div>
        <div style="font-size:22px;font-weight:600;color:#E1F5EE;margin-bottom:.75rem;line-height:1.3;">
          The right lesson, at the right moment.
        </div>
        <div style="font-size:13.5px;color:#B8E8D8;line-height:1.75;max-width:560px;">
          Kairos is a hybrid recommendation engine built for spiritual growth. It combines
          <strong style="color:#E1F5EE;">content-based filtering</strong> — matching lessons to a user's theological interests,
          emotional state, and depth preference — with
          <strong style="color:#E1F5EE;">collaborative filtering</strong> that learns from the collective patterns
          of thousands of learners. New users are served through an onboarding wizard;
          returning users receive increasingly personalised paths as their history grows.
        </div>
      </div>

      <div style="display:flex;flex-direction:column;gap:12px;">
        <div style="background:#E1F5EE;border-radius:12px;padding:1.25rem 1.5rem;flex:1;border:0.5px solid #9FE1CB;">
          <div style="font-size:11px;font-weight:600;color:#0F6E56;text-transform:uppercase;letter-spacing:.07em;margin-bottom:.4rem;">How it works</div>
          <div style="font-size:12.5px;color:#085041;line-height:1.7;">
            🔍 &nbsp;CB model scores lessons on embeddings + metadata<br>
            🤝 &nbsp;CF model finds learners with similar journeys<br>
            ⚖️ &nbsp;Hybrid blends both, with a diversity penalty<br>
            🌱 &nbsp;Cold-start wizard bridges zero-history users
          </div>
        </div>
        <div style="background:#EEEDFE;border-radius:12px;padding:1.25rem 1.5rem;flex:1;border:0.5px solid #C8C6F7;">
          <div style="font-size:11px;font-weight:600;color:#3C3489;text-transform:uppercase;letter-spacing:.07em;margin-bottom:.4rem;">Personalisation signals</div>
          <div style="font-size:12.5px;color:#3C3489;line-height:1.7;">
            🎯 &nbsp;Spiritual goal &amp; theological domain<br>
            🧠 &nbsp;Experience level &amp; depth preference<br>
            💬 &nbsp;Emotional tone of content<br>
            📖 &nbsp;Interaction history &amp; ratings
          </div>
        </div>
      </div>

    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.data_loaded:
        st.info("Load your data files using the sidebar ← to get started."); st.stop()
    lessons=st.session_state.lessons; interactions=st.session_state.interactions
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Total lessons",      f"{len(lessons):,}")
    c2.metric("Unique users",       f"{interactions['USER_ID'].nunique():,}")
    c3.metric("Total interactions", f"{len(interactions):,}")
    rated=interactions["RATING"].notna().sum()
    c4.metric("Ratings coverage",   f"{rated/max(len(interactions),1)*100:.1f}%")
    st.markdown('<div style="height:1px;background:var(--color-border-tertiary);margin:1rem 0 1.5rem"></div>',unsafe_allow_html=True)
    col_a,col_b=st.columns(2)
    with col_a:
        st.markdown(_ch.format("Lessons by theological domain"),unsafe_allow_html=True)
        dom=lessons["THEOLOGICAL_DOMAIN"].value_counts().reset_index(); dom.columns=["Domain","Count"]
        fig=px.bar(dom.head(15),x="Count",y="Domain",orientation="h",color="Count",color_continuous_scale=["#E1F5EE","#085041"],height=400)
        fig.update_layout(showlegend=False,coloraxis_showscale=False,margin=dict(l=0,r=0,t=8,b=0),plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)")
        fig.update_yaxes(title=""); st.plotly_chart(fig,use_container_width=True)
    with col_b:
        st.markdown(_ch.format("Lessons by spiritual depth"),unsafe_allow_html=True)
        dep=lessons["SPIRITUAL_DEPTH"].value_counts().reset_index(); dep.columns=["Depth","Count"]
        fig2=go.Figure(go.Pie(
            labels=dep["Depth"], values=dep["Count"],
            hole=0.62,
            marker=dict(
                colors=["#085041","#1D9E75","#5DCAA5","#9FE1CB","#C7F0E3"],
                line=dict(color="rgba(255,255,255,0.6)", width=2),
            ),
            textinfo="label+percent",
            textfont=dict(size=12, color="#085041"),
            hovertemplate="<b>%{label}</b><br>%{value} lessons<br>%{percent}<extra></extra>",
            pull=[0.04]+[0]*(len(dep)-1),
            sort=True,
        ))
        fig2.update_layout(
            height=400,
            margin=dict(l=0,r=0,t=8,b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=True,
            legend=dict(
                orientation="h", x=0.5, y=-0.08, xanchor="center",
                font=dict(size=11, color="#085041"),
                bgcolor="rgba(0,0,0,0)",
            ),
            annotations=[dict(
                text="Depth<br>mix", x=0.5, y=0.5,
                font=dict(size=13, color="#085041", family="sans-serif"),
                showarrow=False,
            )],
        )
        st.plotly_chart(fig2,use_container_width=True)
    col_c,col_d=st.columns(2)
    with col_c:
        st.markdown(_ch.format("Reading time distribution"),unsafe_allow_html=True)
        fig3=px.histogram(lessons,x="READING_TIME_MIN",nbins=10,color_discrete_sequence=["#1D9E75"],height=400,labels={"READING_TIME_MIN":"Minutes"})
        fig3.update_layout(margin=dict(l=0,r=0,t=0,b=0),plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)"); st.plotly_chart(fig3,use_container_width=True)
    with col_d:
        st.markdown(_ch.format("Rating distribution"),unsafe_allow_html=True)
        if interactions["RATING"].notna().any():
            fig4=px.histogram(interactions.dropna(subset=["RATING"]),x="RATING",nbins=5,color_discrete_sequence=["#5DCAA5"],height=400)
            fig4.update_layout(margin=dict(l=0,r=0,t=0,b=0),plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)"); st.plotly_chart(fig4,use_container_width=True)
        else: st.info("No ratings in this sample.")
    st.markdown(_ch.format("Interactions per user"),unsafe_allow_html=True)
    ipu=interactions.groupby("USER_ID").size().reset_index(name="Count")
    fig5=px.histogram(ipu,x="Count",nbins=30,color_discrete_sequence=["#1D9E75"],height=400,labels={"Count":"Interactions per user"})
    fig5.update_layout(margin=dict(l=0,r=0,t=0,b=0),plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)"); st.plotly_chart(fig5,use_container_width=True)

# ─────────────────────────────────────────────
# PAGE 2 — RECOMMENDER
# ─────────────────────────────────────────────
elif page=="Recommender":
    if not st.session_state.models_trained:
        st.markdown("""<div class="banner"><div class="banner-left">
          <div class="banner-title">Recommendation explorer</div>
          <div class="banner-sub">Train models first using the sidebar ←</div>
        </div></div>""",unsafe_allow_html=True); st.stop()
    hybrid=st.session_state.hybrid_model
    profiles=st.session_state.user_profiles
    histories=st.session_state.user_histories
    tab1,tab2=st.tabs(["Cold-start (new user)","Warm user (existing)"])

    with tab1:
        col1,col2=st.columns(2)
        with col1:
            goal=st.selectbox("Goal",["closer_to_god","understand_bible","find_peace","strengthen_my_faith","make_wiser"])
            topic=st.selectbox("Topic",["discipleship","stewardship","identity","forgiveness","suffering","parenting","relationships","grace","healing"])
            experience=st.selectbox("Style",["quick_practical","uplifting_inspiring","guided_structured","deep_thought"])
        with col2:
            time_pref=st.selectbox("Time",["five_minutes","ten_minutes","fifteen_minutes","twenty_minutes"])
            motivation=st.selectbox("Motivation",["deeper_meaning","better_person","overcoming_struggles","helping_others"])
            connect=st.selectbox("Connection with God",["prayer","reading_bible","journaling_thoughts","worship_music","reflecting_nature"])
        top_n=st.slider("Number of recommendations",3,20,9)
        col_btn, _spacer, col_reset = st.columns([3, 1, 1])
        with col_btn:
            get_recs = st.button("Get recommendations", key="cold_btn", type="primary")
        with col_reset:
            if st.button("Reset seen", key="cold_reset", use_container_width=True):
                st.session_state.cold_shown_ids = []
                st.session_state.last_recs = None
                st.rerun()
        if get_recs:
            profile={"OB_ob_goal":goal,"OB_ob_topic":topic,"OB_ob_ideal_experience":experience,"OB_ob_time":time_pref,"OB_ob_motivation":motivation,"OB_ob_connect_with_god":connect}
            with st.spinner("Computing…"):
                recs=hybrid.recommend(user_id="cold_user_demo",user_profile=profile,top_n=top_n,exclude_lesson_ids=st.session_state.cold_shown_ids)
            if recs.empty: st.warning("No more new recommendations — click Reset seen to start over.")
            else:
                st.session_state.cold_shown_ids = st.session_state.cold_shown_ids + recs["LESSON_ID"].tolist()
                st.session_state.last_recs=recs; st.session_state.last_recs_tab="cold"
                st.session_state.last_recs_meta={"count":len(recs),"type":"Cold-start","cf_w":recs["CF_WEIGHT"].iloc[0]}

    with tab2:
        all_users=[uid for uid in profiles.keys() if len(histories.get(uid,{}).get("lesson_ids",[]))>=3]

        # ── user picker ──
        st.markdown("""<div style="font-size:11px;font-weight:600;color:#085041;text-transform:uppercase;
                        letter-spacing:.07em;margin-bottom:.4rem;">Select a learner</div>""",unsafe_allow_html=True)
        selected_user=st.selectbox("",all_users[:500],label_visibility="collapsed")
        st.markdown("<div style='height:.5rem'></div>",unsafe_allow_html=True)

        user_hist=histories.get(selected_user,{})
        lesson_ids=user_hist.get("lesson_ids",[])
        rated_lesson_ids=user_hist.get("rated_lesson_ids", lesson_ids)
        ratings=user_hist.get("ratings",[])
        user_prof=profiles.get(selected_user,{})

        # ── user type badge ──
        n_int=len(lesson_ids)
        if n_int>=20:   utype,utag_bg,utag_fg="Warm learner","#C7F0E3","#085041"
        elif n_int>=3:  utype,utag_bg,utag_fg="Growing","#E1F5EE","#0F6E56"
        else:           utype,utag_bg,utag_fg="Cold start","#FEF3E2","#854F0B"

        avg_r=round(sum(ratings)/max(len(ratings),1),2) if ratings else None
        stars_html=""
        if avg_r:
            full=int(round(avg_r)); stars_html="".join(
                ['<span style="color:#1D9E75;font-size:14px;">★</span>' if j<full
                 else '<span style="color:#D1E8E2;font-size:14px;">★</span>' for j in range(5)]
            )

        # ── profile card ──
        uid_display=str(selected_user); uid_short=(uid_display[:22]+"…") if len(uid_display)>22 else uid_display
        initials=(uid_display[:2].upper())

        # build onboarding chips HTML
        ob_chips=""
        _ob_icons={"goal":"🎯","topic":"📖","experience":"✨","time":"⏱️",
                   "motivation":"💡","connect_with_god":"🙏","depth":"🔍"}
        if user_prof:
            for k,v in user_prof.items():
                lk=k.replace("OB_ob_","").replace("OB_","")
                icon=next((ic for key,ic in _ob_icons.items() if key in lk),"•")
                label=lk.replace("_"," ").title()
                ob_chips+=f'<div style="display:flex;align-items:center;gap:.5rem;padding:.45rem .75rem;background:#fff;border-radius:8px;border:0.5px solid #9FE1CB;margin-bottom:6px;">\
<span style="font-size:14px;">{icon}</span>\
<div><div style="font-size:10px;font-weight:600;color:#0F6E56;text-transform:uppercase;letter-spacing:.06em;">{label}</div>\
<div style="font-size:12px;color:#085041;font-weight:500;">{str(v).replace("_"," ")}</div></div></div>'
        else:
            ob_chips='<div style="font-size:12px;color:#9FE1CB;font-style:italic;padding:.5rem 0;">No onboarding answers recorded.</div>'

        rating_line=(f'<div style="font-size:12px;color:#0F6E56;margin-top:.25rem;">{stars_html} <span style="color:#6B9E8F;">{avg_r:.2f} avg</span></div>' if avg_r else "")

        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#085041 0%,#0F6E56 100%);
                    border-radius:16px;padding:1.5rem 1.75rem;margin-bottom:1rem;
                    display:flex;gap:1.5rem;align-items:flex-start;">

          <!-- avatar + stats -->
          <div style="flex-shrink:0;text-align:center;width:80px;">
            <div style="width:60px;height:60px;border-radius:50%;background:rgba(255,255,255,0.15);
                        display:flex;align-items:center;justify-content:center;
                        font-size:22px;font-weight:700;color:#E1F5EE;margin:0 auto .5rem;">
              {initials}
            </div>
            <span style="background:{utag_bg};color:{utag_fg};border-radius:20px;
                         padding:2px 8px;font-size:10px;font-weight:700;">{utype}</span>
          </div>

          <!-- main info -->
          <div style="flex:1;min-width:0;">
            <div style="font-size:14px;font-weight:600;color:#E1F5EE;margin-bottom:.2rem;
                        white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{uid_short}</div>
            <div style="display:flex;gap:1.25rem;margin-bottom:.75rem;">
              <div style="text-align:center;">
                <div style="font-size:22px;font-weight:700;color:#E1F5EE;line-height:1;">{n_int}</div>
                <div style="font-size:10px;color:#9FE1CB;text-transform:uppercase;letter-spacing:.06em;">lessons</div>
              </div>
              <div style="width:1px;background:rgba(255,255,255,0.15);"></div>
              <div style="text-align:center;">
                <div style="font-size:22px;font-weight:700;color:#E1F5EE;line-height:1;">{len(user_prof)}</div>
                <div style="font-size:10px;color:#9FE1CB;text-transform:uppercase;letter-spacing:.06em;">ob answers</div>
              </div>
            </div>
            {rating_line}
          </div>

        </div>""",unsafe_allow_html=True)

        col_l,col_r=st.columns([1,1])

        with col_l:
            st.markdown("""<div style="font-size:11px;font-weight:600;color:#085041;text-transform:uppercase;
                            letter-spacing:.07em;margin-bottom:.6rem;">Onboarding profile</div>""",unsafe_allow_html=True)
            st.markdown(f'<div style="background:#F5FFFB;border-radius:12px;padding:.75rem;border:0.5px solid #C7F0E3;">{ob_chips}</div>',unsafe_allow_html=True)

        with col_r:
            st.markdown("""<div style="font-size:11px;font-weight:600;color:#085041;text-transform:uppercase;
                            letter-spacing:.07em;margin-bottom:.6rem;">Generate recommendations</div>""",unsafe_allow_html=True)
            st.markdown(f"""<div style="background:#F5FFFB;border-radius:12px;padding:1rem 1.25rem;
                              border:0.5px solid #C7F0E3;margin-bottom:.75rem;">
              <div style="font-size:12px;color:#0F6E56;line-height:1.6;">
                {'🌿 &nbsp;Hybrid mode — personal history + onboarding blend' if n_int>=3 else '🌱 &nbsp;Cold-start mode — onboarding-driven recommendations'}<br>
                {'📚 &nbsp;'+str(n_int)+' lessons in history' if n_int else '📭 &nbsp;No history yet'}
              </div>
            </div>""",unsafe_allow_html=True)
            top_n_warm=st.slider("Number of lessons",3,20,9,key="warm_n")
            if st.button("✨  Get my reading list",key="warm_btn",type="primary",use_container_width=True):
                with st.spinner("Finding the right lessons…"):
                    recs=hybrid.recommend(user_id=selected_user,user_profile=user_prof or None,rated_lesson_ids=rated_lesson_ids or None,ratings=ratings or None,exclude_lesson_ids=lesson_ids,top_n=top_n_warm)
                if recs.empty: recs=hybrid.recommend(user_id=selected_user+"_fb",user_profile=user_prof,top_n=top_n_warm)
                if not recs.empty:
                    st.session_state.last_recs=recs; st.session_state.last_recs_tab="warm"
                    st.session_state.last_recs_meta={"count":len(recs),"type":"Cold" if recs["IS_COLD"].all() else "Warm","cf_w":recs["CF_WEIGHT"].iloc[0]}

    if st.session_state.get("last_recs") is not None:
        meta=st.session_state.get("last_recs_meta",{})
        cf_w=meta.get("cf_w",0)
        top_domains=" · ".join(st.session_state.last_recs["THEOLOGICAL_DOMAIN"].value_counts().head(2).index.tolist())
        cb_pct=int((1-cf_w)*100); cf_pct=int(cf_w*100)
        rec_type=meta.get('type','')
        rec_count=meta.get('count','')
        st.markdown(f"""
        <div style="background:linear-gradient(120deg,#085041 0%,#0F6E56 60%,#1D9E75 100%);
                    border-radius:16px;padding:1.5rem 2rem;margin-top:1.5rem;margin-bottom:1rem;
                    display:flex;align-items:center;justify-content:space-between;gap:1rem;">
          <div>
            <div style="font-size:11px;font-weight:600;letter-spacing:.08em;color:#9FE1CB;
                        text-transform:uppercase;margin-bottom:.4rem;">Personalised for you</div>
            <div style="font-size:22px;font-weight:600;color:#E1F5EE;margin-bottom:.5rem;">
              Your spiritual reading list
            </div>
            <div style="display:flex;gap:.5rem;flex-wrap:wrap;align-items:center;">
              <span style="background:rgba(255,255,255,0.12);border-radius:20px;padding:3px 12px;
                           font-size:12px;color:#E1F5EE;">{rec_count} lessons</span>
              <span style="background:rgba(255,255,255,0.12);border-radius:20px;padding:3px 12px;
                           font-size:12px;color:#E1F5EE;">{rec_type}</span>
              <span style="background:rgba(255,255,255,0.12);border-radius:20px;padding:3px 12px;
                           font-size:12px;color:#9FE1CB;">CB {cb_pct}% · CF {cf_pct}%</span>
            </div>
          </div>
          <div style="text-align:right;flex-shrink:0;">
            <div style="font-size:11px;color:#9FE1CB;margin-bottom:.3rem;">Top themes</div>
            <div style="font-size:14px;font-weight:500;color:#E1F5EE;">{top_domains.replace(" · ","<br>")}</div>
          </div>
        </div>""",unsafe_allow_html=True)
        _render_rec_grid(st.session_state.last_recs)

# ─────────────────────────────────────────────
# PAGE 3 — USER EXPLORER
# ─────────────────────────────────────────────
elif page=="User Explorer":
    st.markdown("""
    <div style="background:linear-gradient(120deg,#085041 0%,#0F6E56 60%,#1D9E75 100%);
                border-radius:16px;padding:1.5rem 2rem;margin-bottom:1rem;
                display:flex;align-items:center;justify-content:space-between;gap:1rem;">
      <div>
        <div style="font-size:11px;font-weight:600;letter-spacing:.08em;color:#9FE1CB;
                    text-transform:uppercase;margin-bottom:.4rem;">Analytics</div>
        <div style="font-size:22px;font-weight:600;color:#E1F5EE;margin-bottom:.5rem;">
          User Explorer
        </div>
        <div style="display:flex;gap:.5rem;flex-wrap:wrap;">
          <span style="background:rgba(255,255,255,0.12);border-radius:20px;padding:3px 12px;
                       font-size:12px;color:#E1F5EE;">Activity distribution</span>
          <span style="background:rgba(255,255,255,0.12);border-radius:20px;padding:3px 12px;
                       font-size:12px;color:#E1F5EE;">Onboarding analysis</span>
        </div>
      </div>
    </div>""",unsafe_allow_html=True)
    if not st.session_state.data_loaded: st.warning("Load data first."); st.stop()
    interactions=st.session_state.interactions; onboarding=st.session_state.onboarding
    ipu=interactions.groupby("USER_ID").size().reset_index(name="interactions")

    # ── stat cards row 1 ──
    c1,c2,c3=st.columns(3)
    for col,label,val,sub in [
        (c1,"Total users",       f"{ipu['USER_ID'].nunique():,}",      "unique learners"),
        (c2,"Median interactions",f"{ipu['interactions'].median():.0f}","lessons per user"),
        (c3,"Max interactions",  f"{ipu['interactions'].max():,}",      "most active user"),
    ]:
        col.markdown(f"""<div style="background:#E1F5EE;border:0.5px solid #9FE1CB;border-radius:12px;
                         padding:1rem 1.25rem;text-align:center;">
          <div style="font-size:11px;font-weight:600;color:#0F6E56;text-transform:uppercase;
                      letter-spacing:.07em;margin-bottom:.3rem;">{label}</div>
          <div style="font-size:28px;font-weight:600;color:#085041;line-height:1.1;">{val}</div>
          <div style="font-size:11px;color:#0F6E56;margin-top:.25rem;">{sub}</div>
        </div>""",unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>",unsafe_allow_html=True)

    # ── activity chart — ranked scatter ──
    st.markdown(_ch.format("Activity heatmap — day × hour"), unsafe_allow_html=True)

    interactions = st.session_state.interactions.copy()
    interactions["UNLOCKED_AT"] = pd.to_datetime(interactions["UNLOCKED_AT"], errors="coerce")
    interactions = interactions.dropna(subset=["UNLOCKED_AT"])

    interactions["hour"] = interactions["UNLOCKED_AT"].dt.hour
    interactions["day"]  = interactions["UNLOCKED_AT"].dt.dayofweek  # 0=Mon

    heatmap_df = (
        interactions.groupby(["day", "hour"])
        .size()
        .reset_index(name="count")
    )

    pivot = heatmap_df.pivot(index="day", columns="hour", values="count").fillna(0)
    pivot = pivot.reindex(index=range(7), columns=range(24), fill_value=0)

    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[f"{h:02d}:00" for h in range(24)],
        y=day_labels,
        colorscale=[
            [0.0,  "#E1F5EE"],
            [0.25, "#9FE1CB"],
            [0.5,  "#5DCAA5"],
            [0.75, "#1D9E75"],
            [1.0,  "#085041"],
        ],
        hovertemplate="<b>%{y} %{x}</b><br>Interactions: %{z:,}<extra></extra>",
        showscale=True,
        colorbar=dict(
            thickness=10,
            len=0.8,
            tickfont=dict(size=10, color="#085041"),
            outlinewidth=0,
        ),
        xgap=2,
        ygap=2,
    ))

    fig.update_layout(
        height=280,
        margin=dict(l=0, r=0, t=8, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            tickfont=dict(size=10),
            tickvals=[f"{h:02d}:00" for h in [0, 6, 12, 18, 23]],
            ticktext=["midnight", "6am", "noon", "6pm", "11pm"],
            gridcolor="rgba(0,0,0,0)",
        ),
        yaxis=dict(
            tickfont=dict(size=11),
            autorange="reversed",
            gridcolor="rgba(0,0,0,0)",
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── user type cards ──
    cold=(ipu["interactions"]<3).sum(); warm=(ipu["interactions"]>=20).sum(); mid=len(ipu)-cold-warm
    total=len(ipu)
    c1,c2,c3=st.columns(3)
    for col,label,val,pct,color,bg in [
        (c1,"Cold start",   cold, cold/total, "#854F0B","#FEF3E2"),
        (c2,"Growing",      mid,  mid/total,  "#0F6E56","#E1F5EE"),
        (c3,"Warm",         warm, warm/total, "#085041","#C7F0E3"),
    ]:
        bar_w=int(pct*100)
        col.markdown(f"""<div style="background:{bg};border-radius:12px;padding:1rem 1.25rem;">
          <div style="font-size:11px;font-weight:600;color:{color};text-transform:uppercase;
                      letter-spacing:.07em;margin-bottom:.3rem;">{label}</div>
          <div style="font-size:26px;font-weight:600;color:{color};">{val:,}</div>
          <div style="height:4px;background:rgba(0,0,0,0.08);border-radius:2px;margin:.5rem 0;">
            <div style="height:4px;width:{bar_w}%;background:{color};border-radius:2px;"></div>
          </div>
          <div style="font-size:11px;color:{color};">{pct*100:.1f}% of users</div>
        </div>""",unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>",unsafe_allow_html=True)

    # ── onboarding distribution ──
    st.markdown(_ch.format("Onboarding answer distribution"),unsafe_allow_html=True)
    ob_cols=[c for c in onboarding.columns if c.startswith("OB_")]
    if ob_cols:
        sel=st.selectbox("Question",ob_cols,format_func=lambda x:x.replace("OB_ob_","").replace("OB_","").replace("_"," ").title())
        vc=onboarding[sel].value_counts().reset_index(); vc.columns=["Answer","Count"]
        vc=vc.sort_values("Count",ascending=False).reset_index(drop=True)
        total_ob=vc["Count"].sum()
        rows_html=""
        palette=["#085041","#0F6E56","#1D9E75","#5DCAA5","#9FE1CB","#C7F0E3"]
        for i,row in vc.iterrows():
            pct=row["Count"]/total_ob*100
            color=palette[min(i,len(palette)-1)]
            rank_badge=f'<span style="background:{color};color:#fff;border-radius:50%;width:20px;height:20px;display:inline-flex;align-items:center;justify-content:center;font-size:10px;font-weight:700;flex-shrink:0;">#{i+1}</span>'
            rows_html+=f"""
            <div style="margin-bottom:10px;">
              <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
                {rank_badge}
                <span style="font-size:13px;font-weight:500;color:#085041;flex:1;">{row['Answer']}</span>
                <span style="font-size:12px;font-weight:600;color:{color};">{pct:.1f}%</span>
                <span style="font-size:11px;color:#6B9E8F;">{row['Count']:,} users</span>
              </div>
              <div style="height:6px;background:#E1F5EE;border-radius:3px;margin-left:30px;">
                <div style="height:6px;width:{pct:.1f}%;background:linear-gradient(90deg,{color},{palette[max(0,i-1)]});border-radius:3px;transition:width .3s;"></div>
              </div>
            </div>"""
        st.markdown(f'<div style="background:var(--color-background-secondary);border-radius:12px;padding:1.25rem 1.5rem;">{rows_html}</div>',unsafe_allow_html=True)

    # ── user summary table ──
    st.markdown(_ch.format("User summary"),unsafe_allow_html=True)
    summary=(interactions.groupby("USER_ID").agg(interactions=("LESSON_ID","count"),avg_rating=("RATING","mean"),last_activity=("UNLOCKED_AT","max")).reset_index().sort_values("interactions",ascending=False))
    summary["avg_rating"]=summary["avg_rating"].round(2)
    summary["user_type"]=summary["interactions"].apply(lambda x:"Cold" if x<3 else("Warm" if x>=20 else "Growing"))
    top=summary.head(200)
    max_int=top["interactions"].max() or 1
    type_styles={"Cold":("#854F0B","#FEF3E2"),"Growing":("#0F6E56","#E1F5EE"),"Warm":("#085041","#C7F0E3")}
    row_list=[]
    for i,(_, r) in enumerate(top.iterrows()):
        tc,bg=type_styles.get(r["user_type"],("#085041","#E1F5EE"))
        bar_w=int(r["interactions"]/max_int*100)
        rating=f"{r['avg_rating']:.2f}" if pd.notna(r["avg_rating"]) else "—"
        stars=int(round(r["avg_rating"])) if pd.notna(r["avg_rating"]) else 0
        star_html="".join(['<span style="color:#1D9E75">★</span>' if j<stars else '<span style="color:#D1E8E2">★</span>' for j in range(5)])
        last=str(r["last_activity"])[:10] if pd.notna(r["last_activity"]) else "—"
        row_bg="rgba(225,245,238,0.3)" if i%2==1 else "transparent"
        uid=str(r['USER_ID']); uid_short=(uid[:16]+"…") if len(uid)>16 else uid
        row_list.append(f"""<tr style="border-top:0.5px solid #E1F5EE;background:{row_bg};">
          <td style="padding:8px 12px;font-size:12px;color:#085041;font-weight:500;white-space:nowrap;">{uid_short}</td>
          <td style="padding:8px 12px;"><span style="background:{bg};color:{tc};border-radius:20px;padding:2px 10px;font-size:11px;font-weight:600;">{r['user_type']}</span></td>
          <td style="padding:8px 12px;min-width:140px;"><div style="display:flex;align-items:center;gap:8px;"><div style="flex:1;height:5px;background:#E1F5EE;border-radius:3px;"><div style="width:{bar_w}%;height:5px;background:#1D9E75;border-radius:3px;"></div></div><span style="font-size:12px;font-weight:600;color:#085041;min-width:28px;text-align:right;">{r['interactions']}</span></div></td>
          <td style="padding:8px 12px;font-size:12px;">{star_html} <span style="color:#6B9E8F;font-size:11px;">{rating}</span></td>
          <td style="padding:8px 12px;font-size:11px;color:#6B9E8F;">{last}</td>
        </tr>""")
    th="padding:10px 12px;text-align:left;font-size:11px;font-weight:600;color:#9FE1CB;letter-spacing:.06em;text-transform:uppercase;"
    st.markdown(f"""<div style="border-radius:12px;overflow:hidden;border:0.5px solid #9FE1CB;max-height:480px;overflow-y:auto;">
      <table style="width:100%;border-collapse:collapse;">
        <thead><tr style="background:#085041;position:sticky;top:0;">
          <th style="{th}">User ID</th><th style="{th}">Type</th>
          <th style="{th}">Interactions</th><th style="{th}">Avg rating</th>
          <th style="{th}">Last active</th>
        </tr></thead>
        <tbody>{"".join(row_list)}</tbody>
      </table></div>""",unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE 4 — EVALUATION
# ─────────────────────────────────────────────
elif page=="Evaluation":
    st.markdown("""
    <div style="background:linear-gradient(120deg,#085041 0%,#0F6E56 60%,#1D9E75 100%);
                border-radius:16px;padding:1.5rem 2rem;margin-bottom:1rem;
                display:flex;align-items:center;justify-content:space-between;gap:1rem;">
      <div>
        <div style="font-size:11px;font-weight:600;letter-spacing:.08em;color:#9FE1CB;text-transform:uppercase;margin-bottom:.4rem;">Model quality</div>
        <div style="font-size:22px;font-weight:600;color:#E1F5EE;margin-bottom:.5rem;">Evaluation</div>
        <div style="display:flex;gap:.5rem;flex-wrap:wrap;">
          <span style="background:rgba(255,255,255,0.12);border-radius:20px;padding:3px 12px;font-size:12px;color:#E1F5EE;">Precision@K</span>
          <span style="background:rgba(255,255,255,0.12);border-radius:20px;padding:3px 12px;font-size:12px;color:#E1F5EE;">Recall@K</span>
          <span style="background:rgba(255,255,255,0.12);border-radius:20px;padding:3px 12px;font-size:12px;color:#E1F5EE;">NDCG@K</span>
          <span style="background:rgba(255,255,255,0.12);border-radius:20px;padding:3px 12px;font-size:12px;color:#E1F5EE;">Coverage</span>
          <span style="background:rgba(255,255,255,0.12);border-radius:20px;padding:3px 12px;font-size:12px;color:#E1F5EE;">Novelty</span>
        </div>
      </div>
    </div>""",unsafe_allow_html=True)

    if not st.session_state.models_trained: st.warning("Train the models first using the sidebar."); st.stop()

    # ── config row ──
    st.markdown(_ch.format("Run settings"),unsafe_allow_html=True)
    cfg1,cfg2,cfg3=st.columns([1,1,1])
    with cfg1:
        eval_k=st.slider("K (top-K metrics)",5,20,10)
    with cfg2:
        sample_users=st.slider("Users to evaluate",100,5000,500,100)
    with cfg3:
        st.markdown("<div style='height:8px'></div>",unsafe_allow_html=True)
        run_eval=st.button("Run evaluation",type="primary",use_container_width=True)

    if run_eval:
        with st.spinner(f"Evaluating on {sample_users} users…"):
            metrics=evaluate_model(model=st.session_state.hybrid_model,test_interactions=st.session_state.test,user_profiles=st.session_state.user_profiles,user_histories=st.session_state.user_histories,k=eval_k,sample_users=sample_users)
            st.session_state.metrics=metrics

    if st.session_state.metrics:
        m=st.session_state.metrics
        prec=m.get(f"precision@{eval_k}",0)
        rec =m.get(f"recall@{eval_k}",0)
        ndcg=m.get(f"ndcg@{eval_k}",0)
        cov =m.get("coverage",0)
        nov =m.get("novelty",0)

        st.markdown("<div style='height:1rem'></div>",unsafe_allow_html=True)
        st.markdown(_ch.format("Metric scores"),unsafe_allow_html=True)

        # ── metric cards ──
        descs={
            f"Precision@{eval_k}": "Of recommended lessons, how many were relevant",
            f"Recall@{eval_k}": "Of all relevant lessons, how many were found",
            f"NDCG@{eval_k}": "Ranking quality — relevant items ranked higher",
            "Coverage": "Fraction of lesson catalogue recommended",
            "Novelty": "How non-obvious / surprising the recommendations are",
        }
        card_data=[
            (f"Precision@{eval_k}", f"{prec*100:.2f}%", prec, descs[f"Precision@{eval_k}"]),
            (f"Recall@{eval_k}", f"{rec*100:.2f}%", rec, descs[f"Recall@{eval_k}"]),
            (f"NDCG@{eval_k}", f"{ndcg*100:.2f}%", ndcg, descs[f"NDCG@{eval_k}"]),
            ("Coverage", f"{cov*100:.1f}%", cov, descs["Coverage"]),
            ("Novelty", f"{nov:.2f}", min(nov/20,1.0), descs["Novelty"]),
        ]
        cols=st.columns(5)
        for col,(label,val_str,ratio,desc) in zip(cols,card_data):
            bar_w=int(ratio*100)
            col.markdown(f"""<div style="background:#E1F5EE;border:0.5px solid #9FE1CB;border-radius:12px;padding:1rem 1.1rem;height:160px;display:flex;flex-direction:column;justify-content:space-between;">
              <div>
                <div style="font-size:10px;font-weight:600;color:#0F6E56;text-transform:uppercase;letter-spacing:.07em;margin-bottom:.3rem;">{label}</div>
                <div style="font-size:26px;font-weight:600;color:#085041;line-height:1.1;margin-bottom:.4rem;">{val_str}</div>
              </div>
              <div>
                <div style="height:4px;background:rgba(8,80,65,0.12);border-radius:2px;margin-bottom:.5rem;">
                  <div style="height:4px;width:{bar_w}%;background:linear-gradient(90deg,#1D9E75,#085041);border-radius:2px;"></div>
                </div>
                <div style="font-size:10px;color:#0F6E56;line-height:1.4;">{desc}</div>
              </div>
            </div>""",unsafe_allow_html=True)

        st.markdown("<div style='height:1.5rem'></div>",unsafe_allow_html=True)

        # ── radar chart ──
        st.markdown("<div style='height:1.5rem'></div>",unsafe_allow_html=True)
        st.markdown(_ch.format("Radar overview"),unsafe_allow_html=True)
        cats=[f"Precision@{eval_k}",f"Recall@{eval_k}",f"NDCG@{eval_k}","Coverage","Novelty"]
        vals=[prec,rec,ndcg,cov,min(nov/20,1.0)]
        fig=go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=vals+[vals[0]], theta=cats+[cats[0]],
            fill="toself",
            fillcolor="rgba(29,158,117,0.18)",
            line=dict(color="#085041",width=2),
            marker=dict(color="#085041",size=7),
        ))
        fig.update_layout(
            polar=dict(
                bgcolor="rgba(225,245,238,0.3)",
                radialaxis=dict(visible=True,range=[0,1],tickfont=dict(size=9,color="#0F6E56"),gridcolor="#9FE1CB",linecolor="#9FE1CB"),
                angularaxis=dict(tickfont=dict(size=11,color="#085041"),linecolor="#9FE1CB",gridcolor="#9FE1CB"),
            ),
            margin=dict(l=60,r=60,t=40,b=40),
            height=420,
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig,use_container_width=True)