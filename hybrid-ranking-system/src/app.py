# app.py — tag-odaklı hibrit arama (BM25 only tags) + CE + cosine (+ opsiyonel CatBoost)
import os, time, math, hashlib, io, json
import numpy as np
import polars as pl
import streamlit as st
from sentence_transformers import SentenceTransformer
import lightgbm as lgb
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sentence_transformers import CrossEncoder

# --- yeni: sadece tag'lerden BM25
from rank_bm25 import BM25Okapi  # pip install rank_bm25
try:
    from catboost import CatBoostRanker
    _HAS_CATBOOST = True
except Exception:
    _HAS_CATBOOST = False

# --------------------------- Paths ---------------------------
ART_DIR = "artifacts"
ORDER_MODEL_PATH = f"{ART_DIR}/lgb_order.txt"
CLICK_MODEL_PATH = f"{ART_DIR}/lgb_click.txt"
BLEND_PATH       = f"{ART_DIR}/blend.json"
FEATCOLS_PATH    = f"{ART_DIR}/feature_cols.json"

CONTENT_FEATS_PATH      = f"{ART_DIR}/content_feats.parquet"
TERM_CONTENT_FEATS_PATH = f"{ART_DIR}/term_content_feats.parquet"
TERM_FEATS_PATH         = f"{ART_DIR}/term_feats.parquet"

DATA_DIR      = "trendyol-teknofest-hackathon"
POOL_PATH     = f"{DATA_DIR}/hackathon_2nd_phase_data/content_pool.parquet"
FRONTEND_PATH = f"{DATA_DIR}/hackathon_2nd_phase_data/frontend_data.parquet"
META_PATH     = f"{DATA_DIR}/hackathon_data/content/metadata.parquet"

# opsiyonel CatBoost modeli (varsa)
CATBOOST_MODEL_PATH = f"{ART_DIR}/catboost_ltr.cbm"

CSS_PATH = Path(__file__).parent / "assets" / "app.css"

# --------------------------- Page ----------------------------
st.set_page_config(page_title="lakaskob", page_icon="🍆", layout="wide")

# --- CSS ---
def local_css():
    try:
        css = CSS_PATH.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS bulunamadı: {CSS_PATH}. 'assets/app.css' oluşturup stilleri oraya taşıyın.")

local_css()
st.markdown("<hr>", unsafe_allow_html=True)

# --------------------------- Load artifacts ------------------
@st.cache_resource(show_spinner=True)
def load_models_and_meta():
    order_bst = lgb.Booster(model_file=ORDER_MODEL_PATH)
    click_bst = lgb.Booster(model_file=CLICK_MODEL_PATH)
    featcols  = json.load(open(FEATCOLS_PATH))
    blend_w   = float(json.load(open(BLEND_PATH))["w"])
    content_feats      = pl.read_parquet(CONTENT_FEATS_PATH)
    term_content_feats = pl.read_parquet(TERM_CONTENT_FEATS_PATH)
    term_feats         = pl.read_parquet(TERM_FEATS_PATH)
    return order_bst, click_bst, featcols, blend_w, content_feats, term_content_feats, term_feats

ORDER_BST, CLICK_BST, FEATCOLS, BLEND_W, CONTENT_FEATS, TERM_CONTENT_FEATS, TERM_FEATS = load_models_and_meta()

# --------------------------- Utils ---------------------------
def fmt_tl(n: float) -> str:
    try:
        s = f"{float(n):,.2f}"
    except Exception:
        s = "0.00"
    return "₺" + s.replace(",", "X").replace(".", ",").replace("X", ".")

def _safe_cat_path(p: dict) -> str:
    l1 = (p.get("level1_category_name") or "").strip()
    l2 = (p.get("level2_category_name") or "").strip()
    lf = (p.get("leaf_category_name") or "").strip()
    parts = [x for x in [l1, l2, lf] if x]
    return " › ".join(parts)

def render_product_card(product: dict):
    title = product.get("content_title", "No Title Available")
    image_url = product.get("image_url", "https://placehold.co/400x400/CCCCCC/FFFFFF?text=No+Image")
    selling_price = float(product.get("selling_price", 0.0) or 0.0)
    original_price = float(product.get("original_price", 0.0) or 0.0)
    rate_avg = float(product.get("content_rate_avg", 0.0) or 0.0)
    review_count = int(product.get("content_review_count", 0) or 0)
    cv_tags = (product.get("cv_tags") or "").strip()
    cat_path = _safe_cat_path(product)

    price_html = f"<span class='selling-price'>{fmt_tl(selling_price)}</span>"
    if original_price > selling_price > 0:
        price_html += f"<span class='original-price'>{fmt_tl(original_price)}</span>"

    stars_int = int(math.floor(max(0.0, min(5.0, rate_avg))))
    stars = "★" * stars_int + "☆" * (5 - stars_int)
    rating_html = f"<div class='rating-wrapper'><span>{stars}</span><span class='review-count'>({review_count})</span></div>"

    st.markdown(f"""
    <div class='product-card'>
      <div>
        <div class='cat-path'>{cat_path}</div>
        <p class='product-title'>{title}</p>
        <div class='cv-tags'>{cv_tags}</div>
        <img class='product-image' src="{image_url}" onerror="this.onerror=null;this.src='https://placehold.co/400x400/CCCCCC/FFFFFF?text=Error';">
      </div>
      <div>
        <div class='price-wrapper'>{price_html}</div>
        {rating_html}
        <div class='mock-button'>🛒 Sepete Ekle</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# --------------------------- TR normalize --------------------
TR_MAP = str.maketrans({"ç":"c","ğ":"g","ı":"i","ö":"o","ş":"s","ü":"u","Ç":"c","Ğ":"g","İ":"i","I":"i","Ö":"o","Ş":"s","Ü":"u"})

def tr_norm_py(s: str) -> str:
    s = (s or "").translate(TR_MAP).lower()
    out, prev_space = [], False
    for ch in s:
        if ch.isalnum():
            out.append(ch); prev_space = False
        else:
            if not prev_space:
                out.append(" "); prev_space = True
    return ("".join(out)).strip()

def tr_norm_expr(col: pl.Expr) -> pl.Expr:
    return (
        col.fill_null("")
           .str.to_lowercase()
           .str.replace_all("ç", "c").str.replace_all("ğ", "g")
           .str.replace_all("ı", "i").str.replace_all("ö", "o")
           .str.replace_all("ş", "s").str.replace_all("ü", "u")
           .str.replace_all(r"[^a-z0-9\s]+", " ")
           .str.replace_all(r"\s+", " ")
           .str.strip_chars()
    )

# --------------------------- Data load -----------------------
@st.cache_data(show_spinner=False)
def load_pool_and_ui():
    df_pool = pl.read_parquet(POOL_PATH).select("content_id_hashed").unique()
    df_ui   = pl.read_parquet(FRONTEND_PATH)
    df_meta = (
        pl.read_parquet(META_PATH)
          .select(["content_id_hashed",
                   "level1_category_name","level2_category_name","leaf_category_name",
                   "cv_tags"])
          .fill_null("")
    )
    base = df_pool.join(df_ui, on="content_id_hashed", how="left")
    base = base.join(df_meta, on="content_id_hashed", how="left")
    base = base.with_columns([
        (pl.col("leaf_category_name").fill_null("") + " " + pl.col("cv_tags").fill_null("")).alias("_prod_text_raw")
    ])
    base = base.with_columns([tr_norm_expr(pl.col("_prod_text_raw")).alias("_prod_text_norm")])
    return base

catalog = load_pool_and_ui()
if catalog.height == 0:
    st.error("content_pool boş görünüyor. Dosya yolunu kontrol edin.")
    st.stop()

# --------------------------- BM25 (sadece TAG) ---------------------------
@st.cache_resource(show_spinner=False)
def build_bm25(catalog: pl.DataFrame):
    tags = catalog["cv_tags"].fill_null("").to_list()
    docs = [tr_norm_py(t).split() for t in tags]   # sadece tag tokenları
    return BM25Okapi(docs)

BM25 = build_bm25(catalog)

@st.cache_resource
def train_query_classifier(catalog):
    labels = catalog["leaf_category_name"].drop_nulls().unique().to_list()
    X = [tr_norm_py(x) for x in labels]
    y = labels
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=1)
    Xv = vec.fit_transform(X)
    clf = LogisticRegression(max_iter=500)
    clf.fit(Xv, y)
    return vec, clf

VEC, CLF = train_query_classifier(catalog)

def predict_product_type(query: str):
    qn = tr_norm_py(query)
    Xq = VEC.transform([qn])
    pred = CLF.predict(Xq)[0]
    return pred

# --------------------------- Model & Emb cache ---------------
@st.cache_resource(show_spinner=True)
def get_model():
    # EMBEDDING MODELİNİ DEGISTIRMIYORUZ
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Reranker (daha kaliteli ve hızlı)
RERANKER_NAME = "BAAI/bge-reranker-v2-m3"  # dilersen eskiye dönebilirsin

@st.cache_resource(show_spinner=True)
def get_reranker():
    return CrossEncoder(RERANKER_NAME)

@st.cache_resource(show_spinner=True)
def build_product_embeddings_cached(texts: list[str]):
    h = hashlib.sha1(("||".join(texts)).encode("utf-8")).hexdigest()
    cache_path = f"/tmp/prod_emb_{h}.npz"
    if os.path.exists(cache_path):
        return np.load(cache_path)["emb"]
    model = get_model()
    embs = model.encode(texts, batch_size=128, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
    np.savez_compressed(cache_path, emb=embs)
    return embs

prod_texts = catalog["_prod_text_norm"].to_list()
prod_emb   = build_product_embeddings_cached(prod_texts)
id_list    = catalog["content_id_hashed"].to_list()

# --------------------------- Topbar (logo + search + sort) ----------------------
st.markdown('<div class="topbar">', unsafe_allow_html=True)
col_logo, col_search, col_btn, col_sort = st.columns([1.3, 5.5, 0.9, 1.6])

with col_logo:
    st.markdown('<div class="brand-logo">hacetchills</div>', unsafe_allow_html=True)

with col_search:
    query = st.text_input(
        label="Arama",
        value="ofis içi klasik erkek giyim",
        placeholder="ofis içi klasik erkek giyim",
        label_visibility="collapsed",
        key="q_top"
    )

with col_btn:
    search_clicked = st.button("Ara", key="btn_top")

with col_sort:
    sort_pref = st.selectbox(
        "Sırala",
        ["Önerilen", "Ucuz önce", "Pahalı önce", "Rating yüksek", "İndirim yüksek", "ID (stabil)"],
        index=0,
        label_visibility="collapsed",
        key="sort_top"
    )
st.markdown("</div>", unsafe_allow_html=True)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# --------------------------- Sidebar ------------------------
with st.sidebar:
    st.header("Filtreler")
    categories = ["Hepsi"] + sorted(catalog["leaf_category_name"].drop_nulls().unique().to_list())
    selected_cat = st.selectbox("Kategori", categories)

    price_col = catalog["selling_price"].drop_nulls()
    pmin = float(price_col.min()) if price_col.len() else 0.0
    pmax = float(price_col.max()) if price_col.len() else 10000.0
    price_range = st.slider("Fiyat aralığı (₺)", 0.0, float(pmax), (max(0.0, pmin), float(pmax)), 1.0)

    sort_pref = st.selectbox(
        "Eşit skorda ikinci sıralama",
        ["Ucuz önce", "Pahalı önce", "Rating yüksek", "İndirim yüksek", "ID (stabil)"],
        index=0
    )

    st.markdown("---")
    TOPK = 50

# --------------------------- Query emb ----------------------
q_text = tr_norm_py(query)
model = get_model()
q_emb = model.encode([q_text], convert_to_numpy=True, normalize_embeddings=True)[0]
cos_full = (prod_emb @ q_emb)
cos_df = pl.DataFrame({"content_id_hashed": id_list, "cos_sim": cos_full})

# --------------------------- Filters ------------------------
filtered = catalog
if selected_cat != "Hepsi":
    filtered = filtered.filter(pl.col("leaf_category_name") == selected_cat)
filtered = filtered.filter((pl.col("selling_price") >= price_range[0]) & 
                           (pl.col("selling_price") <= price_range[1]))
if filtered.height == 0:
    st.info("Filtre sonrası sonuç bulunamadı, filtreler kaldırıldı.")
    filtered = catalog

def tr_norm_for_terms(s: str) -> str:
    s = (s or "").translate(TR_MAP).lower()
    out, prev_sep = [], False
    for ch in s:
        if ch.isalnum():
            out.append(ch)
            prev_sep = False
        else:
            if not prev_sep:
                out.append("_")
                prev_sep = True
    return ("".join(out)).strip("_")

q_norm = tr_norm_for_terms(query)

ui = (
    filtered.select([
        "content_id_hashed","content_title","image_url",
        "selling_price","original_price","content_rate_avg","content_review_count",
        "level1_category_name","level2_category_name","leaf_category_name","cv_tags"
    ])
    .join(cos_df, on="content_id_hashed", how="left")
    .with_columns([
        (pl.when((pl.col("original_price") > 0) & (pl.col("selling_price") > 0))
           .then(((pl.col("original_price") - pl.col("selling_price")) / pl.col("original_price")).clip(0, 1))
           .otherwise(0.0)).alias("_discount"),
        pl.col("cos_sim").fill_null(0.0),
        pl.lit(q_norm).alias("search_term_normalized")
    ])
)

# --------------------------- FE build (LGBM için; opsiyonel) -----------------------
def build_feature_matrix(cand: pl.DataFrame, q_norm: str) -> pl.DataFrame:
    tc = TERM_CONTENT_FEATS.filter(pl.col("search_term_normalized") == q_norm)
    tf = TERM_FEATS.filter(pl.col("search_term_normalized") == q_norm)

    if q_norm not in TERM_FEATS["search_term_normalized"].unique().to_list():
        st.info(f"'{query}' terimi TERM_FEATS içinde yok. Sadece embedding + content feature'larla skorlanıyor.")

    if tc.height == 0 or tf.height == 0:
        Xpl = (
            cand.join(CONTENT_FEATS, on="content_id_hashed", how="left")
                .with_columns(pl.col("cos_sim").alias("query_prod_cos"))
        )
    else:
        Xpl = (
            cand.join(CONTENT_FEATS, on="content_id_hashed", how="left")
                .join(tc, on=["content_id_hashed","search_term_normalized"], how="left")
                .join(tf, on="search_term_normalized", how="left")
                .with_columns(pl.col("cos_sim").alias("query_prod_cos"))
        )
    missing = [c for c in FEATCOLS if c not in Xpl.columns]
    for c in missing:
        Xpl = Xpl.with_columns(pl.lit(0.0).alias(c))
    Xpl = Xpl.with_columns([pl.col(c).fill_null(0.0).cast(pl.Float32) for c in FEATCOLS])
    return Xpl

# --------------------------- Hibrit ön eleme: Dense + BM25(tag) → RRF ---------------------------
def rrf_merge(rank_arrays, k=60):
    agg = {}
    for arr in rank_arrays:
        for r, idx in enumerate(arr):
            agg[idx] = agg.get(idx, 0.0) + 1.0 / (k + r + 1)
    return agg

# 1) Dense (cosine)
k_dense = min(2000, len(cos_full))
dense_sorted = np.argsort(-cos_full)[:k_dense]

# 2) BM25 (sadece tag)
q_tokens = tr_norm_py(query).split()
bm25_scores = BM25.get_scores(q_tokens)          # len = tüm katalog
k_bm25 = min(2000, len(bm25_scores))
bm25_sorted = np.argsort(-bm25_scores)[:k_bm25]

# 3) RRF birleştirme
rrf = rrf_merge([dense_sorted, bm25_sorted], k=60)
k_final = 1500
cand_indices = [i for i, _ in sorted(rrf.items(), key=lambda x: -x[1])[:k_final]]

# Adayların id ve BM25 skorları
id_arr = np.array(id_list)
cand_ids = id_arr[cand_indices].tolist()
cand_bm25 = np.array(bm25_scores)[cand_indices]
df_bm25 = pl.DataFrame({"content_id_hashed": cand_ids, "_bm25_raw": cand_bm25})

# Aday DataFrame'i
cand_df = ui.filter(pl.col("content_id_hashed").is_in(cand_ids))

# Reranker metni (TAG'LER ÖNCE)
cand_df = cand_df.with_columns([
    (
        pl.col("cv_tags").fill_null("") + " | " +           # tags önce
        pl.col("content_title").fill_null("") + " | " +
        pl.col("level1_category_name").fill_null("") + " > " +
        pl.col("level2_category_name").fill_null("") + " > " +
        pl.col("leaf_category_name").fill_null("")
    ).alias("_rerank_text")
])

# BM25 skorunu ekle + normalize et
cand_df = (
    cand_df.join(df_bm25, on="content_id_hashed", how="left")
           .with_columns(pl.col("_bm25_raw").fill_null(0.0))
)
bm_min = float(cand_df["_bm25_raw"].min())
bm_max = float(cand_df["_bm25_raw"].max())
bm_den = max(1e-8, bm_max - bm_min)
cand_df = cand_df.with_columns(((pl.col("_bm25_raw") - bm_min) / bm_den).alias("_bm25_norm"))

# --------------------------- Ucuz skor → top N → CrossEncoder ---------------------------
# ucuz skor: cosine + BM25(tag) + küçük indirim etkisi
cos_cand = cand_df["cos_sim"].fill_null(0.0).to_numpy()
cmin, cmax = float(cos_cand.min()), float(cos_cand.max())
cden = max(1e-8, cmax - cmin)
cos_norm = (cos_cand - cmin) / cden

disc = cand_df["_discount"].fill_null(0.0).to_numpy()

# BM25(tag) ağırlığı yüksek (kural yok)
cheap = 0.55 * cos_norm + 0.40 * cand_df["_bm25_norm"].to_numpy() + 0.05 * disc

top_for_ce = min(400, cand_df.height)

# take() kullanmadan: ucuz skoru kolona yaz, ona göre sırala, top N al
cand_df = cand_df.with_columns(pl.Series("_cheap", cheap))
cand_small = cand_df.sort("_cheap", descending=True).head(top_for_ce)


# --------------------------- CrossEncoder (sadece top N) ---------------------------
reranker = get_reranker()
pairs = list(zip([query] * cand_small.height, cand_small["_rerank_text"].to_list()))
r_scores = reranker.predict(pairs, batch_size=128)
r_min, r_max = float(np.min(r_scores)), float(np.max(r_scores))
r_norm = (r_scores - r_min) / max(1e-8, (r_max - r_min))
cand_small = cand_small.with_columns(pl.Series(name="_ce_norm", values=r_norm.astype(np.float32)))

# cosine'u alt küme için normalize et
cos_sub = cand_small["cos_sim"].fill_null(0.0).to_numpy()
cs_min, cs_max = float(cos_sub.min()), float(cos_sub.max())
cs_den = max(1e-8, cs_max - cs_min)
cos_sub_norm = (cos_sub - cs_min) / cs_den

# --------------------------- (Opsiyonel) CatBoost LTR harmanı ---------------------------
cb_norm = None
if _HAS_CATBOOST and os.path.exists(CATBOOST_MODEL_PATH):
    try:
        @st.cache_resource(show_spinner=False)
        def get_catboost():
            m = CatBoostRanker()
            m.load_model(CATBOOST_MODEL_PATH)
            return m
        cb = get_catboost()
        feat_cols_cb = ["_ce_norm", "_bm25_norm", "cos_norm_cb", "_discount", "content_rate_avg", "content_review_count"]
        cand_small = cand_small.with_columns(pl.Series("cos_norm_cb", cos_sub_norm.astype(np.float32)))
        for c in feat_cols_cb:
            if c not in cand_small.columns:
                cand_small = cand_small.with_columns(pl.lit(0.0).alias(c))
        X_cb = cand_small.select(feat_cols_cb).to_numpy()
        cb_score = cb.predict(X_cb).astype("float32")
        cb_min, cb_max = float(cb_score.min()), float(cb_score.max())
        cb_norm = (cb_score - cb_min) / max(1e-8, (cb_max - cb_min))
    except Exception:
        cb_norm = None

# --------------------------- Final skor (kural YOK) ---------------------------
if cb_norm is None:
    final_score = (
        0.6 * cand_small["_ce_norm"].fill_null(0.0).to_numpy() +
        0.25 * cand_small["_bm25_norm"].fill_null(0.0).to_numpy() +
        0.15 * cos_sub_norm
    )
else:
    final_score = (
        0.5  * cand_small["_ce_norm"].fill_null(0.0).to_numpy() +
        0.25 * cand_small["_bm25_norm"].fill_null(0.0).to_numpy() +
        0.10 * cos_sub_norm +
        0.15 * cb_norm
    )

# --------------------------- TOP-K --------------------------
N = final_score.shape[0]
k = min(int(TOPK), N) if N > 0 else 0
if k == 0:
    st.warning("Filtrelerden sonra sonuç bulunamadı.")
    st.stop()

top_idx = np.argsort(final_score)[-k:][::-1]
ui_top = cand_small.with_row_index(name="rn")
top_df = pl.DataFrame({"rn": pl.Series(top_idx).cast(pl.UInt32), "score": pl.Series(final_score[top_idx])})
ui_top = ui_top.join(top_df, on="rn", how="inner")

# --------------------------- Sort ---------------------------
by_cols  = ["score", "cos_sim"]
desc_flg = [True, True]

if   sort_pref == "Ucuz önce":      by_cols += ["selling_price"];    desc_flg += [False]
elif sort_pref == "Pahalı önce":    by_cols += ["selling_price"];    desc_flg += [True]
elif sort_pref == "Rating yüksek":  by_cols += ["content_rate_avg"]; desc_flg += [True]
elif sort_pref == "İndirim yüksek": by_cols += ["_discount"];        desc_flg += [True]
else:
    by_cols += ["content_title"]
    desc_flg += [False]
res = ui_top.sort(by=by_cols, descending=desc_flg)

# --------------------------- Result header ---------------------------
res_count = int(res.height)
st.markdown(
    f'<div class="result-header">"<span>{query}</span>" araması için {res_count} sonuç listeleniyor</div>',
    unsafe_allow_html=True
)

# --------------------------- Grid ---------------------------
products_to_display = res.to_dicts()
cols_per_row = 4
cols = st.columns(cols_per_row)
for i, product in enumerate(products_to_display):
    with cols[i % cols_per_row]:
        render_product_card(product)
    if (i + 1) % cols_per_row == 0 and (i + 1) < len(products_to_display):
        cols = st.columns(cols_per_row)

# --------------------------- Debug --------------------------
with st.expander("ℹ️ Debug"):
    dbg = {
        "query": query,
        "q_norm": q_norm,
        "reranker": RERANKER_NAME,
        "bm25_source": "cv_tags only",
        "n_candidates_hybrid": int(len(cand_indices)),
        "reranked_on": int(cand_small.height),
        "catboost_used": bool(_HAS_CATBOOST and os.path.exists(CATBOOST_MODEL_PATH)),
        "pool_rows_used": int(ui.height)
    }
    st.write(dbg)
