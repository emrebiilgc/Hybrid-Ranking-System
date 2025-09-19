import polars as pl
import numpy as np
import os
from sentence_transformers import SentenceTransformer, util
from sklearn.model_selection import GroupKFold

from src.utils import mm


DATA_PATH = "data"


# -----------------------------
# 1) Load raw data
# -----------------------------
def load_data():
    train_sessions = pl.read_parquet(f"{DATA_PATH}/train_sessions.parquet")
    test_sessions = pl.read_parquet(f"{DATA_PATH}/test_sessions.parquet")

    content_metadata = pl.read_parquet(f"{DATA_PATH}/content/metadata.parquet")
    content_price_data = pl.read_parquet(f"{DATA_PATH}/content/price_rate_review_data.parquet")
    content_sw = pl.read_parquet(f"{DATA_PATH}/content/sitewide_log.parquet").with_columns(
        pl.col("date").cast(pl.Date)
    )
    ctt = pl.read_parquet(f"{DATA_PATH}/content/top_terms_log.parquet").with_columns(
        pl.col("date").cast(pl.Date)
    )
    user_meta = pl.read_parquet(f"{DATA_PATH}/user/metadata.parquet")
    u_site = pl.read_parquet(f"{DATA_PATH}/user/sitewide_log.parquet")
    uf = pl.read_parquet(f"{DATA_PATH}/user/fashion_sitewide_log.parquet")
    term = pl.read_parquet(f"{DATA_PATH}/term/search_log.parquet")
    ut = pl.read_parquet(f"{DATA_PATH}/user/top_terms_log.parquet")
    max_d = ctt.select(pl.max("date")).item()
    ctt60 = ctt.filter(pl.col("date") > (max_d - pl.duration(days=60)))

    return (
        train_sessions, test_sessions,
        content_metadata, content_price_data, content_sw,
        ctt, user_meta, u_site, uf, term, ut,ctt60
    )


# -----------------------------
# 2) Feature engineering steps
# -----------------------------
def add_basic_dates(train, test, content_price_data):
    train = train.with_columns(train["ts_hour"].cast(pl.Date).alias("ts_date"))
    test = test.with_columns(test["ts_hour"].cast(pl.Date).alias("ts_date"))
    content_price_data = content_price_data.with_columns(content_price_data["update_date"].cast(pl.Date).alias("ts_date"))
    return train, test, content_price_data
def add_popularity(train, test, content_sw):
    max_d = content_sw.select(pl.max("date")).item()
    d7 = max_d - pl.duration(days=7)
    d30 = max_d - pl.duration(days=30)

    sw7 = (
        content_sw.filter(pl.col("date") > d7)
        .group_by("content_id_hashed")
        .agg(click7=pl.sum("total_click"), order7=pl.sum("total_order"))
    )
    sw30 = (
        content_sw.filter(pl.col("date") > d30)
        .group_by("content_id_hashed")
        .agg(click30=pl.sum("total_click"), order30=pl.sum("total_order"))
    )

    pop = sw30.join(sw7, on="content_id_hashed", how="full").fill_null(0.0)
    pop = (
        pop
        .with_columns([
            mm("order7").alias("order7_n"),
            mm("click7").alias("click7_n"),
            mm("order30").alias("order30_n"),
            mm("click30").alias("click30_n"),
        ])
        .with_columns(
            (0.6*pl.col("order7_n") + 0.2*pl.col("click7_n")
             + 0.15*pl.col("order30_n") + 0.05*pl.col("click30_n")).alias("pop_score")
        )
        .select(["content_id_hashed", "pop_score"])
    )

    train = train.join(pop, on="content_id_hashed", how="left").with_columns(pl.col("pop_score").fill_null(0.0))
    test = test.join(pop, on="content_id_hashed", how="left").with_columns(pl.col("pop_score").fill_null(0.0))
    return train, test
def add_term_ctr(train, test, data_path=DATA_PATH, window_days=60):
    """
    Top terms log verilerinden content × term CTR çıkarır.
    - Son 'window_days' gün için hesaplanır (default=60).
    - 'term_ctr' feature'ı train ve test'e eklenir.
    """

    ctt = pl.read_parquet(f"{data_path}/content/top_terms_log.parquet").with_columns(
        pl.col("date").cast(pl.Date)
    )

    # son N gün
    max_d = ctt.select(pl.max("date")).item()
    win = max_d - pl.duration(days=window_days)

    ctt60 = (
        ctt.filter(pl.col("date") > win)
        .with_columns(
            (pl.col("total_search_click") / (pl.col("total_search_impression") + 1e-9)).alias("ctr60")
        )
    )

    term_rel = (
        ctt60.group_by(["content_id_hashed", "search_term_normalized"])
        .agg(pl.mean("ctr60").clip(0, 1).alias("term_ctr"))
    )

    join_keys = ["content_id_hashed", "search_term_normalized"]
    train = train.join(term_rel, on=join_keys, how="left").with_columns(pl.col("term_ctr").fill_null(0.0))
    test = test.join(term_rel, on=join_keys, how="left").with_columns(pl.col("term_ctr").fill_null(0.0))

    print("✅ Term CTR eklendi (term_ctr)")
    return train, test
def add_user_features(train, test, user_meta, u_site):
    """
    User metadata + 30 günlük kullanıcı aktiviteleri (click/order) ekler.
    """

    # Kullanıcı meta: yaş hesapla
    user_meta = user_meta.select(
        ["user_id_hashed", "user_birth_year", "user_tenure_in_days", "user_gender"]
    ).with_columns(
        (2025 - pl.col("user_birth_year")).alias("user_age")
    )

    # Kullanıcı sitewide log -> son 30 gün
    u_site = u_site.with_columns(pl.col("ts_hour").cast(pl.Date).alias("ts_date"))
    max_ut = u_site.select(pl.max("ts_date")).item()
    u30 = u_site.filter(pl.col("ts_date") > (max_ut - pl.duration(days=30)))

    # 30 günlük aggregations
    u30_agg = (
        u30.group_by("user_id_hashed")
           .agg([
               pl.col("total_click").sum().alias("u30_click_sum"),
               pl.col("total_cart").sum().alias("u30_cart_sum"),
               pl.col("total_fav").sum().alias("u30_fav_sum"),
               pl.col("total_order").sum().alias("u30_order_sum"),
           ])
           .with_columns([
               (pl.col("u30_click_sum") + 1e-6).log().alias("u30_click_log"),
               (pl.col("u30_order_sum") + 1e-6).log().alias("u30_order_log"),
               (pl.col("u30_order_sum")/(pl.col("u30_click_sum")+1e-9)).clip(0,1).alias("u30_conv_rate"),
           ])
    )

    # Join user_meta + u30_agg
    def join_user_features(df):
        return (
            df.join(user_meta, on="user_id_hashed", how="left")
              .join(u30_agg, on="user_id_hashed", how="left")
              .with_columns([
                  pl.col("user_age").fill_null(0),
                  pl.col("user_tenure_in_days").fill_null(0),
                  pl.col("user_gender").fill_null("U"),
                  pl.col("u30_click_log").fill_null(0.0),
                  pl.col("u30_order_log").fill_null(0.0),
                  pl.col("u30_conv_rate").fill_null(0.0),
              ])
        )

    train = join_user_features(train)
    test = join_user_features(test)

    print("✅ User meta + 30d activity eklendi")
    return train, test
def add_ewma_popularity(train, test, content_sw, alpha=0.3, window_days=60):
    """
    İçerik bazlı son X günlük EWMA popülerlik skorunu ekler.
    - click ve order günlük toplanır
    - EWMA ile ağırlıklı ortalama alınır
    - normalize edilip 'pop_ewma' olarak join edilir
    """

    # Son X günü al
    max_d = content_sw.select(pl.max("date")).item()
    content_sw = content_sw.filter(pl.col("date") > (max_d - pl.duration(days=window_days)))

    # Günlük bazda toplam
    daily = (
        content_sw
        .group_by(["content_id_hashed", "date"])
        .agg([
            pl.col("total_click").sum().alias("click"),
            pl.col("total_order").sum().alias("order"),
        ])
        .sort(["content_id_hashed", "date"])
    )

    # ewma_t = alpha*x_t + (1-alpha)*ewma_(t-1)
    def ewma_expr(col):
        return (alpha * pl.col(col)) + (1 - alpha) * pl.col(f"{col}_ewma_prev")

    # EWMA hesapla
    for metric in ["click", "order"]:
        daily = daily.with_columns(pl.col(metric).alias(f"{metric}_ewma"))
        daily = daily.with_columns(
            pl.col(f"{metric}_ewma").shift(1).over("content_id_hashed").alias(f"{metric}_ewma_prev")
        )
        daily = daily.with_columns(
            ewma_expr(metric).over("content_id_hashed").alias(f"{metric}_ewma")
        )

    # Son gün + normalize
    ewma_last = (
        daily.group_by("content_id_hashed")
        .agg([
            pl.col("click_ewma").last().alias("click_ewma_last"),
            pl.col("order_ewma").last().alias("order_ewma_last"),
        ])
        .with_columns([
            ((pl.col("order_ewma_last") - pl.min("order_ewma_last")) /
             (pl.max("order_ewma_last") - pl.min("order_ewma_last") + 1e-12)).alias("o_ewma_n"),
            ((pl.col("click_ewma_last") - pl.min("click_ewma_last")) /
             (pl.max("click_ewma_last") - pl.min("click_ewma_last") + 1e-12)).alias("c_ewma_n"),
        ])
        .with_columns((0.75 * pl.col("o_ewma_n") + 0.25 * pl.col("c_ewma_n")).alias("pop_ewma"))
        .select(["content_id_hashed", "pop_ewma"])
    )

    # Train/Test'e join
    train = train.join(ewma_last, on="content_id_hashed", how="left").with_columns(pl.col("pop_ewma").fill_null(0.0))
    test  = test.join(ewma_last,  on="content_id_hashed", how="left").with_columns(pl.col("pop_ewma").fill_null(0.0))

    print("✅ EWMA popülerlik feature eklendi")
    return train, test
def add_session_pop_rank(train, test):
    """
    Her oturum içinde içerikleri 'pop_ewma' değerine göre sıralar.
    - 'pop_rank_in_sess': oturum içi sıralama (ordinal rank)
    - 'pop_rr_in_sess' : [0,1] arasında normalize edilmiş relatif rank
    """

    def _rank(df: pl.DataFrame) -> pl.DataFrame:
        return (
            df.with_columns([
                pl.col("pop_ewma").rank("ordinal", descending=True).over("session_id").alias("pop_rank_in_sess"),
                pl.len().over("session_id").alias("_n")
            ])
            .with_columns(((pl.col("pop_rank_in_sess") - 1) / (pl.col("_n") - 1 + 1e-9)).alias("pop_rr_in_sess"))
            .drop("_n")
        )

    train = _rank(train)
    test = _rank(test)

    print("✅ Session popularity relative rank eklendi")
    return train, test
def add_term_ctr_smoothed(train, test, data_path=DATA_PATH, window_days=60, k=50.0):
    """
    Top terms log verilerinden smoothed CTR hesaplar.
    - Son 'window_days' gün için Beta smoothing uygulanır (default=60).
    - 'term_ctr_smoothed' feature'ı train ve test'e eklenir.
    """

    ctt = pl.read_parquet(f"{data_path}/content/top_terms_log.parquet").with_columns(
        pl.col("date").cast(pl.Date)
    )

    # son N gün
    max_d = ctt.select(pl.max("date")).item()
    win = max_d - pl.duration(days=window_days)
    ctt60 = ctt.filter(pl.col("date") > win)

    # global ctr
    glob_ctr = (
        ctt60["total_search_click"].sum() /
        (ctt60["total_search_impression"].sum() + 1e-9)
    )

    alpha0 = glob_ctr * k
    beta0  = (1.0 - glob_ctr) * k

    agg = (
        ctt60.group_by(["content_id_hashed", "search_term_normalized"])
        .agg([
            pl.col("total_search_click").sum().alias("click"),
            pl.col("total_search_impression").sum().alias("imp")
        ])
        .with_columns(
            ((pl.col("click") + alpha0) / (pl.col("imp") + alpha0 + beta0))
            .alias("term_ctr_smoothed")
        )
    )

    # join et
    join_cols = ["content_id_hashed", "search_term_normalized"]
    train = train.join(agg.select(join_cols + ["term_ctr_smoothed"]), on=join_cols, how="left") \
                 .with_columns(pl.col("term_ctr_smoothed").fill_null(0.0))
    test  = test.join(agg.select(join_cols + ["term_ctr_smoothed"]), on=join_cols, how="left") \
                 .with_columns(pl.col("term_ctr_smoothed").fill_null(0.0))

    print("✅ Term CTR (smoothed) eklendi: term_ctr_smoothed")
    return train, test
def add_metadata_and_price(train, test, content_metadata, content_price_data):
    """
    content_metadata ve content_price_data'yı session tablolarına join eder.
    - content_metadata: ürünün temel bilgileri
    - content_price_data: ürünün tarihsel fiyat bilgileri (ts_date üzerinden)
    """

    train = train.join(content_metadata, on="content_id_hashed", how="left")
    train = train.join(content_price_data, on=["content_id_hashed", "ts_date"], how="left")

    test = test.join(content_metadata, on="content_id_hashed", how="left")
    test = test.join(content_price_data, on=["content_id_hashed", "ts_date"], how="left")

    print("✅ Metadata & Price eklendi")
    return train, test
def add_discount_rate(train, test):
    """
    original_price ve selling_price üzerinden discount_rate hesaplar.
    - discount_rate = (original_price - selling_price) / original_price
    - NaN veya sonsuz değerleri 0.0 ile değiştirir
    """

    def _add(df: pl.DataFrame) -> pl.DataFrame:
        return (
            df.with_columns(
                ((pl.col("original_price") - pl.col("selling_price")) /
                 (pl.col("original_price") + 1e-9)).alias("discount_rate")
            )
            .with_columns(
                pl.when(pl.col("discount_rate").is_nan() | ~pl.col("discount_rate").is_finite())
                .then(0.0).otherwise(pl.col("discount_rate"))
                .alias("discount_rate")
            )
        )

    train = _add(train)
    test = _add(test)

    print("✅ Discount rate eklendi")
    return train, test
def add_media_review_ratio(train, test):
    """
    Fotoğraf/videolu yorum oranını hesaplar:
    media_review_ratio = content_review_wth_media_count / content_review_count
    - Eksik değerleri 0.0 ile doldurur
    """

    def _add(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            (pl.col("content_review_wth_media_count") / (pl.col("content_review_count") + 1e-9))
            .fill_null(0.0)
            .alias("media_review_ratio")
        )

    train = _add(train)
    test = _add(test)

    print("✅ Media review ratio eklendi")
    return train, test
def add_price_vs_leaf_med(train, test):
    """
    Her içerik için, aynı gün ve aynı kategori içindeki median fiyat ile
    kıyaslama oranını hesaplar:
        price_vs_leaf_med = selling_price / median(selling_price in same leaf+date)

    - Eksik değerleri korur.
    """

    both = pl.concat([
        train.select(["leaf_category_name", "ts_date", "selling_price"]).with_columns(pl.lit("tr").alias("_src")),
        test.select(["leaf_category_name", "ts_date", "selling_price"]).with_columns(pl.lit("te").alias("_src")),
    ])

    leaf_day = both.group_by(["leaf_category_name", "ts_date"]).agg(
        med_price=pl.median("selling_price")
    )

    def _add(df: pl.DataFrame) -> pl.DataFrame:
        return (
            df.join(leaf_day, on=["leaf_category_name", "ts_date"], how="left")
              .with_columns((pl.col("selling_price") / (pl.col("med_price") + 1e-9)).alias("price_vs_leaf_med"))
              .drop("med_price")
        )

    train = _add(train)
    test = _add(test)

    print("✅ price_vs_leaf_med eklendi")
    return train, test
def add_price_trend_and_bayes(train, test, content_price_data):
    """
    - Son 7gün / 30gün ortalama ve std fiyatlarını hesaplar
    - price_vs_p7 / price_vs_p30 oranlarını ekler
    - Bayesian rating hesaplar (content_rate_avg, content_rate_count üzerinden)
    """

    # === PRICE TREND / VOLATILITY ===
    pcols = content_price_data.select(["content_id_hashed", "ts_date", "selling_price"])
    max_pd = pcols.select(pl.max("ts_date")).item()

    p7 = pcols.filter(pl.col("ts_date") > (max_pd - pl.duration(days=7)))
    p30 = pcols.filter(pl.col("ts_date") > (max_pd - pl.duration(days=30)))

    p7_agg = p7.group_by("content_id_hashed").agg([
        pl.col("selling_price").mean().alias("p7_mean"),
        pl.col("selling_price").std(ddof=1).fill_null(0.0).alias("p7_std"),
    ])

    p30_agg = p30.group_by("content_id_hashed").agg([
        pl.col("selling_price").mean().alias("p30_mean"),
        pl.col("selling_price").std(ddof=1).fill_null(0.0).alias("p30_std"),
    ])

    def _add_trend(df: pl.DataFrame) -> pl.DataFrame:
        return (
            df.join(p7_agg, on="content_id_hashed", how="left")
              .join(p30_agg, on="content_id_hashed", how="left")
              .with_columns([
                  pl.col("p7_mean").fill_null(pl.col("selling_price")),
                  pl.col("p30_mean").fill_null(pl.col("selling_price")),
                  pl.col("p7_std").fill_null(0.0),
                  pl.col("p30_std").fill_null(0.0),
                  (pl.col("selling_price") / (pl.col("p7_mean") + 1e-9)).alias("price_vs_p7"),
                  (pl.col("selling_price") / (pl.col("p30_mean") + 1e-9)).alias("price_vs_p30"),
              ])
        )

    train = _add_trend(train)
    test = _add_trend(test)

    # === BAYESIAN RATING ===
    global_mean = float(train["content_rate_avg"].mean())
    k_bayes = 20.0

    def _add_bayes(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            (
                (pl.col("content_rate_avg") * pl.col("content_rate_count") + global_mean * k_bayes) /
                (pl.col("content_rate_count") + k_bayes + 1e-9)
            ).alias("rating_bayes")
        ).with_columns(pl.col("rating_bayes").fill_null(global_mean))

    train = _add_bayes(train)
    test = _add_bayes(test)

    print("✅ Price trend & Bayesian rating eklendi")
    return train, test
def add_term_global_ctr(train, test, data_path: str):
    """
    Term global CTR (smoothed) hesaplar:
    - Son 60 gün search_log verisi üzerinden
    - Arama terimi bazlı CTR
    - Smoothed (Bayesian) oran
    """

    # Search log verisi oku
    term = pl.read_parquet(f"{data_path}/term/search_log.parquet").with_columns(
        pl.col("ts_hour").cast(pl.Date).alias("ts_date")
    )

    # Son 60 gün
    max_dt = term.select(pl.max("ts_date")).item()
    term60 = term.filter(pl.col("ts_date") > (max_dt - pl.duration(days=60)))

    # Global CTR
    glob_ctr_t = (
        term60["total_search_click"].sum() /
        (term60["total_search_impression"].sum() + 1e-9)
    )

    # Bayesian smoothing parametreleri
    k = 50.0
    a0 = glob_ctr_t * k
    b0 = (1 - glob_ctr_t) * k

    # Term bazlı CTR hesapla
    term_ctr = (
        term60.group_by("search_term_normalized")
        .agg([
            pl.col("total_search_click").sum().alias("clk"),
            pl.col("total_search_impression").sum().alias("imp"),
        ])
        .with_columns(((pl.col("clk") + a0) / (pl.col("imp") + a0 + b0)).alias("term_global_ctr_sm"))
        .select(["search_term_normalized", "term_global_ctr_sm"])
    )

    # Train ve test'e join et
    def _join_ctr(df: pl.DataFrame) -> pl.DataFrame:
        return df.join(term_ctr, on="search_term_normalized", how="left")\
                 .with_columns(pl.col("term_global_ctr_sm").fill_null(0.0))

    train = _join_ctr(train)
    test = _join_ctr(test)

    print("✅ Term global CTR (smoothed) eklendi")
    return train, test
def add_user_term_ctr(train, test, data_path: str):
    """
    User × Term CTR (smoothed) hesaplar:
    - Son 60 gün user/top_terms_log verisi üzerinden
    - Kullanıcı × arama terimi bazlı CTR
    - Bayesian smoothing uygulanır
    """

    ut = pl.read_parquet(f"{data_path}/user/top_terms_log.parquet")\
           .with_columns(pl.col("ts_hour").cast(pl.Date).alias("ts_date"))

    # Son 60 gün
    max_du = ut.select(pl.max("ts_date")).item()
    ut60 = ut.filter(pl.col("ts_date") > (max_du - pl.duration(days=60)))

    # Global CTR
    glob = ut60["total_search_click"].sum() / (ut60["total_search_impression"].sum() + 1e-9)

    # Bayesian smoothing parametreleri
    k = 30.0
    a0 = glob * k
    b0 = (1 - glob) * k

    # Kullanıcı × term CTR hesapla
    u_t = (
        ut60.group_by(["user_id_hashed", "search_term_normalized"])
        .agg([
            pl.col("total_search_click").sum().alias("clk"),
            pl.col("total_search_impression").sum().alias("imp"),
        ])
        .with_columns(((pl.col("clk") + a0) / (pl.col("imp") + a0 + b0)).alias("user_term_ctr_sm"))
        .select(["user_id_hashed", "search_term_normalized", "user_term_ctr_sm"])
    )

    # Train ve test'e join et
    def _join_ctr(df: pl.DataFrame) -> pl.DataFrame:
        return df.join(u_t, on=["user_id_hashed", "search_term_normalized"], how="left")\
                 .with_columns(pl.col("user_term_ctr_sm").fill_null(0.0))

    train = _join_ctr(train)
    test = _join_ctr(test)

    print("✅ User × Term CTR (smoothed) eklendi")
    return train, test
def add_term_content_lift(train, test):
    """
    Term × Content Lift özelliğini ekler.
    Hesaplama:
        log( (term_ctr_smoothed + 1e-6) / (term_global_ctr_sm + 1e-6) )

    Çıktı:
        - term_content_lift_log
    """

    def _add(df):
        return df.with_columns(
            (
                ((pl.col("term_ctr_smoothed") + 1e-6) /
                 (pl.col("term_global_ctr_sm") + 1e-6))
                .log()
                .alias("term_content_lift_log")
            )
        )

    train = _add(train)
    test = _add(test)

    print("✅ Term × Content Lift eklendi")
    return train, test
def add_term_leaf_share(train, test, ctt60, content_metadata):
    """
    Intent Alignment (term × leaf share) özelliğini ekler.
    Hesaplama:
        term_leaf_share = clk / toplam_clk (her term için)
    """

    # İçerik → leaf eşleştirmesi
    leaf_map = content_metadata.select(["content_id_hashed", "leaf_category_name"])
    ctt60 = ctt60.join(leaf_map, on="content_id_hashed", how="left")

    # Her term × leaf toplam klik
    term_leaf = (
        ctt60.group_by(["search_term_normalized", "leaf_category_name"])
        .agg(pl.col("total_search_click").sum().alias("clk"))
    )

    # Her term için toplam klik
    term_tot = term_leaf.group_by("search_term_normalized").agg(
        pl.col("clk").sum().alias("clk_sum")
    )

    # Oranı hesapla
    term_leaf = (
        term_leaf.join(term_tot, on="search_term_normalized", how="left")
        .with_columns(
            (pl.col("clk") / (pl.col("clk_sum") + 1e-9)).alias("term_leaf_share")
        )
        .select(["search_term_normalized", "leaf_category_name", "term_leaf_share"])
    )

    # Train ve test'e join et
    def _add(df):
        return df.join(term_leaf, on=["search_term_normalized", "leaf_category_name"], how="left")\
                 .with_columns(pl.col("term_leaf_share").fill_null(0.0))

    train = _add(train)
    test = _add(test)

    print("✅ Term × Leaf Share eklendi")
    return train, test
def add_session_local(train, test):
    """
    Session-local özellik ekler:
    - price_rank_in_sess: ürünün fiyat sırasının session içindeki oranı
    """

    def _add(df):
        return (
            df.with_columns([
                pl.col("selling_price").rank("ordinal").over("session_id").alias("_rk"),
                pl.len().over("session_id").alias("_n")
            ])
            .with_columns(
                ((pl.col("_rk") - 1) / (pl.col("_n") - 1 + 1e-9)).alias("price_rank_in_sess")
            )
            .drop(["_rk", "_n"])
        )

    train = _add(train)
    test = _add(test)

    print("✅ Session-local feature eklendi: price_rank_in_sess")
    return train, test
def add_sess_size(train, test):
    """
    Session büyüklüğünü (kaç ürün içerdiğini) ekler.
    - sess_size: session_id başına item sayısı
    """

    def _add(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(pl.len().over("session_id").alias("sess_size"))

    train = _add(train)
    test = _add(test)

    print("✅ Session size feature eklendi: sess_size")
    return train, test
def add_user_personalization(train, test, content_metadata, data_path):
    """
    Kullanıcı bazlı kişiselleştirme özelliklerini ekler:
    - personal_pop_uc: user × content yakınlığı
    - personal_aff_leaf: user × leaf-category yakınlığı
    """

    # User fashion sitewide log
    UF_PATH = f"{data_path}/user/fashion_sitewide_log.parquet"
    uf = (
        pl.read_parquet(UF_PATH)
          .with_columns(pl.col("ts_hour").cast(pl.Date).alias("ts_date"))
          .select(["user_id_hashed", "content_id_hashed", "ts_date", "total_click", "total_order"])
    )

    # Son 60 gün
    max_du = uf.select(pl.max("ts_date")).item()
    uf = uf.filter(pl.col("ts_date") > (max_du - pl.duration(days=60)))

    # --- USER–CONTENT yakınlığı (personal_pop_uc)
    uc = (
        uf.group_by(["user_id_hashed", "content_id_hashed"])
          .agg([
              pl.col("total_click").sum().alias("uc_click"),
              pl.col("total_order").sum().alias("uc_order")
          ])
    )
    uc = uc.join(
        uc.group_by("user_id_hashed")
          .agg([
              pl.col("uc_click").max().alias("max_uc_click"),
              pl.col("uc_order").max().alias("max_uc_order"),
          ]),
        on="user_id_hashed", how="left"
    ).with_columns([
        (pl.col("uc_click")/(pl.col("max_uc_click")+1e-9)).alias("uc_click_n"),
        (pl.col("uc_order")/(pl.col("max_uc_order")+1e-9)).alias("uc_order_n"),
    ]).with_columns(
        (0.3*pl.col("uc_click_n") + 0.7*pl.col("uc_order_n")).alias("personal_pop_uc")
    ).select(["user_id_hashed", "content_id_hashed", "personal_pop_uc"])

    # --- USER–LEAF-CATEGORY yakınlığı (personal_aff_leaf)
    leaf_map = content_metadata.select(["content_id_hashed", "leaf_category_name"])
    uf_leaf = uf.join(leaf_map, on="content_id_hashed", how="left")

    ul = (
        uf_leaf.group_by(["user_id_hashed", "leaf_category_name"])
               .agg([
                   pl.col("total_click").sum().alias("ul_click"),
                   pl.col("total_order").sum().alias("ul_order"),
               ])
    )
    ul = ul.join(
        ul.group_by("user_id_hashed")
          .agg([
              pl.col("ul_click").max().alias("max_ul_click"),
              pl.col("ul_order").max().alias("max_ul_order"),
          ]),
        on="user_id_hashed", how="left"
    ).with_columns([
        (pl.col("ul_click")/(pl.col("max_ul_click")+1e-9)).alias("ul_click_n"),
        (pl.col("ul_order")/(pl.col("max_ul_order")+1e-9)).alias("ul_order_n"),
    ]).with_columns(
        (0.3*pl.col("ul_click_n") + 0.7*pl.col("ul_order_n")).alias("personal_aff_leaf")
    ).select(["user_id_hashed", "leaf_category_name", "personal_aff_leaf"])

    # Train & Test join
    train = (
        train
        .join(uc, on=["user_id_hashed","content_id_hashed"], how="left")
        .join(ul, on=["user_id_hashed","leaf_category_name"], how="left")
        .with_columns([
            pl.col("personal_pop_uc").fill_null(0.0),
            pl.col("personal_aff_leaf").fill_null(0.0),
        ])
    )
    test = (
        test
        .join(uc, on=["user_id_hashed","content_id_hashed"], how="left")
        .join(ul, on=["user_id_hashed","leaf_category_name"], how="left")
        .with_columns([
            pl.col("personal_pop_uc").fill_null(0.0),
            pl.col("personal_aff_leaf").fill_null(0.0),
        ])
    )

    print("✅ User personalization features eklendi: personal_pop_uc, personal_aff_leaf")
    return train, test
def add_time_and_term(train, test):
    """
    Zaman ve arama terimi şekil özelliklerini ekler:
    - hour: ts_hour → saat
    - dow: ts_hour → haftanın günü (0=Mon, 6=Sun)
    - is_weekend: hafta sonu mu (1/0)
    - term_len: arama teriminin karakter uzunluğu
    - term_word_count: arama teriminin kelime sayısı
    """

    def _add(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns([
            pl.col("ts_hour").dt.hour().alias("hour"),
            pl.col("ts_hour").dt.weekday().alias("dow"),
            (pl.col("ts_hour").dt.weekday().is_in([5, 6])).cast(pl.Int8).alias("is_weekend"),
            pl.col("search_term_normalized").str.len_chars().alias("term_len"),
            (pl.col("search_term_normalized").str.count_matches(r"\s+") + 1).fill_null(1).alias("term_word_count"),
        ])

    train = _add(train)
    test = _add(test)

    print("✅ Time & Term-shape features eklendi: hour, dow, is_weekend, term_len, term_word_count")
    return train, test
def add_leaf_order_share(train, test, content_metadata, content_sw):
    """
    Leaf-category bazında order prior oranı ekler:
    - leaf_order_share = leaf_category_order / total_orders
    """

    leaf_map = content_metadata.select(["content_id_hashed", "leaf_category_name"])

    leaf_orders = (
        content_sw.join(leaf_map, on="content_id_hashed", how="left")
        .group_by("leaf_category_name")
        .agg(pl.col("total_order").sum().alias("leaf_orders"))
    )

    leaf_prior = (
        leaf_orders.with_columns(
            (pl.col("leaf_orders") / pl.col("leaf_orders").sum()).alias("leaf_order_share")
        )
        .select(["leaf_category_name", "leaf_order_share"])
    )

    def _add(df: pl.DataFrame) -> pl.DataFrame:
        return (
            df.join(leaf_prior, on="leaf_category_name", how="left")
              .with_columns(pl.col("leaf_order_share").fill_null(0.0))
        )

    train = _add(train)
    test = _add(test)

    print("✅ Leaf order share eklendi")
    return train, test
def add_oof_leaf_encoding(train, test, n_splits=5):
    """
    OOF target encoding for leaf_category_name.
    - te_leaf_order_oof: leaf bazlı order oranı
    - te_leaf_click_oof: leaf bazlı click oranı
    """

    # Train'i pandas'a çevir
    _te_pd = train.select(
        ["leaf_category_name", "ordered", "clicked", "user_id_hashed"]
    ).to_pandas()

    te_oof_leaf_order = np.zeros(len(_te_pd))
    te_oof_leaf_click = np.zeros(len(_te_pd))

    gkf_te = GroupKFold(n_splits=n_splits)

    for tr, va in gkf_te.split(_te_pd, groups=_te_pd["user_id_hashed"]):
        m_o = _te_pd.iloc[tr].groupby("leaf_category_name")["ordered"].mean()
        m_c = _te_pd.iloc[tr].groupby("leaf_category_name")["clicked"].mean()

        te_oof_leaf_order[va] = (
            _te_pd.iloc[va]["leaf_category_name"].map(m_o).fillna(m_o.mean()).values
        )
        te_oof_leaf_click[va] = (
            _te_pd.iloc[va]["leaf_category_name"].map(m_c).fillna(m_c.mean()).values
        )

    # Train'e kolon ekle
    train = train.with_columns([
        pl.Series("te_leaf_order_oof", te_oof_leaf_order),
        pl.Series("te_leaf_click_oof", te_oof_leaf_click),
    ])

    # Test için full encoding
    _full_o = _te_pd.groupby("leaf_category_name")["ordered"].mean()
    _full_c = _te_pd.groupby("leaf_category_name")["clicked"].mean()

    test_te_o = (
        test.select(["leaf_category_name"])
        .to_pandas()["leaf_category_name"]
        .map(_full_o).fillna(_full_o.mean()).values
    )
    test_te_c = (
        test.select(["leaf_category_name"])
        .to_pandas()["leaf_category_name"]
        .map(_full_c).fillna(_full_c.mean()).values
    )

    test = test.with_columns([
        pl.Series("te_leaf_order_oof", test_te_o),
        pl.Series("te_leaf_click_oof", test_te_c),
    ])

    print("✅ OOF Leaf target encoding eklendi: te_leaf_order_oof, te_leaf_click_oof")
    return train, test

def add_advanced_interactions(train, test):
    """
    Çeşitli advanced interaction özellikleri ekler:
    - Price × Quality, Discount × Quality, Price × Popularity
    - User × Context: Age × Price, Conversion × Popularity, Tenure × Quality
    - Search × Content: Term × Popularity, Term × Quality, UserTerm × Personal
    - Time × Context: Hour × Price, Weekend × Popularity
    - Session × Context: SessionSize × Position, Diversity × Popularity
    """

    def _add(df):
        return df.with_columns([
            # Price-Quality interactions
            (pl.col("selling_price") * pl.col("content_rate_avg")).alias("price_x_quality"),
            (pl.col("discount_rate") * pl.col("content_rate_avg")).alias("discount_x_quality"),
            (pl.col("selling_price") * pl.col("pop_ewma")).alias("price_x_popularity"),

            # User-Context interactions
            (pl.col("user_age") * pl.col("selling_price")).alias("age_x_price"),
            (pl.col("u30_conv_rate") * pl.col("pop_ewma")).alias("user_conversion_x_popularity"),
            (pl.col("user_tenure_in_days") * pl.col("content_rate_avg")).alias("tenure_x_quality"),

            # Search-Content interactions
            (pl.col("term_ctr_smoothed") * pl.col("pop_ewma")).alias("term_ctr_x_popularity"),
            (pl.col("term_global_ctr_sm") * pl.col("content_rate_avg")).alias("term_global_x_quality"),
            (pl.col("user_term_ctr_sm") * pl.col("personal_pop_uc")).alias("user_term_x_personal"),

            # Time-Context interactions
            (pl.col("hour") * pl.col("selling_price")).alias("hour_x_price"),
            (pl.col("is_weekend") * pl.col("pop_ewma")).alias("weekend_x_popularity"),

        ])

    train = _add(train)
    test = _add(test)

    print("✅ Advanced interaction features eklendi")
    return train, test

def add_cart_fav_features(train, test, content_sw):
    """
    Sitewide loglardan Cart/Fav bazlı özellikler ekler:
    - cart_to_order_7d / cart_to_order_30d
    - click_to_cart_7d / click_to_cart_30d
    - fav_cart_ratio_7d / fav_cart_ratio_30d
    - engagement_score_7d / engagement_score_30d
    """

    max_d = content_sw.select(pl.max("date")).item()

    # Son 7 gün
    sw7_cf = (
        content_sw.filter(pl.col("date") > (max_d - pl.duration(days=7)))
        .group_by("content_id_hashed")
        .agg([
            pl.col("total_cart").sum().alias("cart7"),
            pl.col("total_fav").sum().alias("fav7"),
            pl.col("total_click").sum().alias("click7_cf"),
            pl.col("total_order").sum().alias("order7_cf"),
        ])
    )

    # Son 30 gün
    sw30_cf = (
        content_sw.filter(pl.col("date") > (max_d - pl.duration(days=30)))
        .group_by("content_id_hashed")
        .agg([
            pl.col("total_cart").sum().alias("cart30"),
            pl.col("total_fav").sum().alias("fav30"),
            pl.col("total_click").sum().alias("click30_cf"),
            pl.col("total_order").sum().alias("order30_cf"),
        ])
    )

    cart_fav_features = sw7_cf.join(sw30_cf, on="content_id_hashed", how="full").fill_null(0.0)

    cart_fav_features = cart_fav_features.with_columns([
        (pl.col("order7_cf") / (pl.col("cart7") + 1e-9)).alias("cart_to_order_7d"),
        (pl.col("order30_cf") / (pl.col("cart30") + 1e-9)).alias("cart_to_order_30d"),

        (pl.col("cart7") / (pl.col("click7_cf") + 1e-9)).alias("click_to_cart_7d"),
        (pl.col("cart30") / (pl.col("click30_cf") + 1e-9)).alias("click_to_cart_30d"),

        (pl.col("fav7") / (pl.col("cart7") + 1e-9)).alias("fav_cart_ratio_7d"),
        (pl.col("fav30") / (pl.col("cart30") + 1e-9)).alias("fav_cart_ratio_30d"),

        ((pl.col("cart7") + pl.col("fav7")) / 2).log().alias("engagement_score_7d"),
        ((pl.col("cart30") + pl.col("fav30")) / 2).log().alias("engagement_score_30d"),
    ])

    cart_fav_features = cart_fav_features.select([
        "content_id_hashed",
        "cart_to_order_7d", "cart_to_order_30d",
        "click_to_cart_7d", "click_to_cart_30d",
        "fav_cart_ratio_7d", "fav_cart_ratio_30d",
        "engagement_score_7d", "engagement_score_30d",
    ])

    # Join train & test
    def _join(df):
        return df.join(cart_fav_features, on="content_id_hashed", how="left").with_columns([
            pl.col(c).fill_null(0.0) for c in cart_fav_features.columns if c != "content_id_hashed"
        ])

    train = _join(train)
    test = _join(test)

    print("✅ Cart/Fav features eklendi")
    return train, test




# (Buraya diğer adımlar gelecek: user_meta join, ewma, term_ctr, personalization, time features,
# price-based features, cart/fav features, embeddings vs.)


# -----------------------------
# 3) Embedding feature
# -----------------------------
def add_embeddings(train, test):
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    def prepare_texts(df):
        queries = df["search_term_normalized"].unique().to_list()
        products = (df["leaf_category_name"].fill_null("") + " " + df["cv_tags"].fill_null("")).unique().to_list()
        return queries, products

    def encode_unique(texts, fname):
        if os.path.exists(fname):
            return np.load(fname)
        embs = model.encode(texts, batch_size=128, convert_to_numpy=True, show_progress_bar=True)
        np.save(fname, embs)
        return embs

    def build_sim_feature(df, qmap, pmap):
        def cos_sim_row(q, p, t):
            q = q if q is not None else ""
            p = p if p is not None else ""
            t = t if t is not None else ""
            key = p + " " + t
            if q in qmap and key in pmap:
                return float(util.cos_sim(qmap[q], pmap[key]).item())
            return 0.0

        return (
            df.with_columns(
                pl.struct(["search_term_normalized", "leaf_category_name", "cv_tags"])
                .map_elements(lambda row: cos_sim_row(row["search_term_normalized"], row["leaf_category_name"], row["cv_tags"]))
                .alias("query_prod_cos")
            )
        )

    # Train embeddings
    q_texts, p_texts = prepare_texts(train)
    q_embs = encode_unique(q_texts, "query_emb.npy")
    p_embs = encode_unique(p_texts, "product_emb.npy")

    qmap = dict(zip(q_texts, q_embs))
    pmap = dict(zip(p_texts, p_embs))
    train = build_sim_feature(train, qmap, pmap)

    # Test embeddings
    q_texts_t, p_texts_t = prepare_texts(test)
    q_embs_t = encode_unique(q_texts_t, "query_emb_test.npy")
    p_embs_t = encode_unique(p_texts_t, "product_emb_test.npy")

    qmap_t = dict(zip(q_texts_t, q_embs_t))
    pmap_t = dict(zip(p_texts_t, p_embs_t))
    test = build_sim_feature(test, qmap_t, pmap_t)

    print("✅ Embedding feature eklendi: query_prod_cos")
    return train, test


# -----------------------------
# 4) Save
# -----------------------------
def save_features(train, test, out_dir="features"):
    os.makedirs(out_dir, exist_ok=True)
    train.write_parquet(f"{out_dir}/train_features.parquet")
    test.write_parquet(f"{out_dir}/test_features.parquet")
    print(f"✅ Features saved to {out_dir}/")


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    (
        train_sessions, test_sessions,
        content_metadata, content_price_data, content_sw,
        ctt, user_meta, u_site, uf, term, ut,ctt60
    ) = load_data()

    # Örnek pipeline (sadece ilk adımlar ve embeddingler)
    train_sessions, test_sessions, content_price_data = add_basic_dates(train_sessions, test_sessions, content_price_data)
    train_sessions, test_sessions = add_popularity(train_sessions, test_sessions, content_sw)

    train_sessions, test_sessions = add_term_ctr(train_sessions, test_sessions)
    train_sessions, test_sessions = add_user_features(train_sessions, test_sessions, user_meta, u_site)
    train_sessions, test_sessions = add_ewma_popularity(train_sessions, test_sessions, content_sw)
    train_sessions, test_sessions = add_session_pop_rank(train_sessions, test_sessions)
    train_sessions, test_sessions = add_term_ctr_smoothed(train_sessions, test_sessions)
    train_sessions, test_sessions = add_metadata_and_price(train_sessions, test_sessions, content_metadata, content_price_data)
    train_sessions, test_sessions = add_discount_rate(train_sessions, test_sessions)
    train_sessions, test_sessions = add_media_review_ratio(train_sessions, test_sessions)
    train_sessions, test_sessions = add_price_vs_leaf_med(train_sessions, test_sessions)
    train_sessions, test_sessions = add_price_trend_and_bayes(train_sessions, test_sessions, content_price_data)
    train_sessions, test_sessions = add_term_global_ctr(train_sessions, test_sessions, DATA_PATH)
    train_sessions, test_sessions = add_user_term_ctr(train_sessions, test_sessions, DATA_PATH)
    train_sessions, test_sessions = add_term_content_lift(train_sessions, test_sessions)
    train_sessions, test_sessions = add_term_leaf_share(train_sessions, test_sessions, ctt60, content_metadata)
    train_sessions, test_sessions = add_session_local(train_sessions, test_sessions)
    train_sessions, test_sessions = add_sess_size(train_sessions, test_sessions)
    train_sessions, test_sessions = add_user_personalization(train_sessions, test_sessions, content_metadata, DATA_PATH)
    train_sessions, test_sessions = add_time_and_term(train_sessions, test_sessions)
    train_sessions, test_sessions = add_leaf_order_share(train_sessions, test_sessions, content_metadata, content_sw)
    train_sessions, test_sessions = add_oof_leaf_encoding(train_sessions, test_sessions, n_splits=5)
    train_sessions, test_sessions = add_advanced_interactions(train_sessions, test_sessions)
    train_sessions, test_sessions = add_cart_fav_features(train_sessions, test_sessions, content_sw)

    train_sessions, test_sessions = add_embeddings(train_sessions, test_sessions)

    # (Not: diğer tüm feature adımları buraya fonksiyonlaştırılıp sırayla eklenecek)

    save_features(train_sessions, test_sessions)
