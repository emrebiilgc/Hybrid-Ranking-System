import polars as pl
import joblib
from glob import glob
import numpy as np


def load_test_data(features_path="features/test_features.parquet"):
    """
    Hazırlanmış test verisini parquet'ten okur.
    """
    df = pl.read_parquet(features_path)
    return df



def load_models(blend_params_path="models/blend_params.pkl"):
    order_paths = sorted(glob("models/model_order_f*.pkl"))
    click_paths = sorted(glob("models/model_click_f*.pkl"))
    if not order_paths or not click_paths:
        # geriye dönük uyumluluk (tek model varsa)
        order_paths = ["models/model_order.pkl"]
        click_paths = ["models/model_click.pkl"]

    models_order = [joblib.load(p) for p in order_paths]
    models_click = [joblib.load(p) for p in click_paths]
    w = joblib.load(blend_params_path).get("w", 0.7)
    print(f"🔎 Yüklendi: {len(models_order)} order, {len(models_click)} click modeli (w={w:.3f})")
    return models_order, models_click, w



def make_predictions(models_order, models_click, df_test, feature_cols, w, epsilon=0.02, out_file="submission.csv"):
    """
    Test seti üzerinde tahmin üretir, blend eder ve Kaggle submission dosyasını kaydeder.
    """

    X_test = df_test.select(feature_cols).to_pandas()

    order_stack = [m.predict_proba(X_test)[:, 1] for m in models_order]
    click_stack = [m.predict_proba(X_test)[:, 1] for m in models_click]
    test_order = np.mean(order_stack, axis=0)
    test_click = np.mean(click_stack, axis=0)


    # Blend: w * order + (1-w) * click
    test_blend = w * test_order + (1 - w) * test_click

    # Tie-breaker (pop_ewma ile)
    if "pop_ewma" in df_test.columns:
        pop_ewma_te = df_test.select("pop_ewma").to_pandas().values.ravel()
        test_final = test_blend + epsilon * pop_ewma_te
    else:
        test_final = test_blend

    # Predictionları ekle
    tmp = df_test.with_columns(
        pl.Series(name="prediction", values=test_final)
    ).select(["session_id", "content_id_hashed", "prediction"]) \
     .sort(["session_id", "prediction"], descending=True)

    # Session bazlı sıralı content listesi
    submission_df = tmp.group_by("session_id").agg(
        pl.col("content_id_hashed").alias("prediction")
    ).with_columns(
        pl.col("prediction").list.join(" ")
    )

    # Kaydet
    submission_df.write_csv(out_file)
    print(f"✅ Submission saved to {out_file} | w={w:.3f}")


if __name__ == "__main__":
    # Test verisini yükle
    df_test = load_test_data()

    # Modelleri ve blend parametrelerini yükle
    models_order, models_click, w = load_models()

    # Feature listesi (train.py ile aynı olmalı)
    feature_cols = [
        "content_rate_avg","filterable_label_count",
        "original_price","selling_price","discounted_price",
        "pop_score","term_ctr","pop_ewma","term_ctr_smoothed",
        "discount_rate","personal_pop_uc","personal_aff_leaf",
        "media_review_ratio","price_vs_leaf_med",
        "term_global_ctr_sm","user_term_ctr_sm","term_leaf_share",
        "price_rank_in_sess","user_age",
        "user_tenure_in_days","u30_click_log","u30_order_log","u30_conv_rate",
        "p7_mean","p30_mean","p7_std","p30_std","price_vs_p7","price_vs_p30",
        "rating_bayes","term_content_lift_log","pop_rr_in_sess",
        "leaf_order_share","hour","dow","is_weekend","term_len","term_word_count",
        "te_leaf_order_oof","te_leaf_click_oof","sess_size",
        "cart_to_order_7d", "cart_to_order_30d", "click_to_cart_7d", "click_to_cart_30d",
        "fav_cart_ratio_7d", "fav_cart_ratio_30d", "engagement_score_7d", "engagement_score_30d",
        "query_prod_cos"
    ]

    # Tahmin üret
    make_predictions(models_order, models_click, df_test, feature_cols, w)
