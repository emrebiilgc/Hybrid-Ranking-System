# src/train.py

import polars as pl
import numpy as np
import lightgbm as lgb
import joblib
import os

from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from src.utils import trendyol_score


def load_train_data(features_path="features/train_features.parquet"):
    """
    Hazırlanmış train verisini parquet'ten okur.
    """
    df = pl.read_parquet(features_path)
    return df


def train_lightgbm(df, feature_cols, y_order, y_click, session_id, user_groups, n_splits=5):
    """
    Order ve Click için ayrı LightGBM modelleri eğitir.
    En iyi blend katsayısını bulur.
    """
    gkf = GroupKFold(n_splits=n_splits)

    # OOF tahminler
    oof_order = np.zeros(len(df))
    oof_click = np.zeros(len(df))

    # Test dummy (bizde test burada yok, sadece eğitim için OOF önemli)
    models_order = []
    models_click = []

    params = dict(
        objective="binary",
        boosting_type="gbdt",
        n_estimators=5000,
        learning_rate=0.03,
        num_leaves=127,
        reg_alpha=0.5,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
        bagging_fraction=0.8,
        bagging_freq=1,
        min_data_in_leaf=200,
        feature_fraction=0.8,
        min_gain_to_split=0.03,
        lambda_l1= 0.5,
        lambda_l2= 3,
        max_bin= 255,
        drop_rate=0.1,
        skip_drop=0.5,
    )

    # === ORDER modeli ===
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(df, y_order, groups=user_groups), 1):
        print(f"\n=== ORDER Fold {fold}/{n_splits} ===")
        model = lgb.LGBMClassifier(**params)
        model.fit(
            df.iloc[tr_idx][feature_cols], y_order.iloc[tr_idx],
            eval_set=[(df.iloc[va_idx][feature_cols], y_order.iloc[va_idx])],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(200), lgb.log_evaluation(100)]
        )
        oof_order[va_idx] = model.predict_proba(df.iloc[va_idx][feature_cols])[:, 1]
        models_order.append(model)

    print("✅ Order model OOF AUC:", roc_auc_score(y_order, oof_order))

    # === CLICK modeli ===
    pos_c = y_click.sum()
    neg_c = len(y_click) - pos_c
    params_click = {**params, "scale_pos_weight": float(neg_c) / max(float(pos_c), 1.0)}

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(df, y_click, groups=user_groups), 1):
        print(f"\n=== CLICK Fold {fold}/{n_splits} ===")
        model = lgb.LGBMClassifier(**params_click)
        model.fit(
            df.iloc[tr_idx][feature_cols], y_click.iloc[tr_idx],
            eval_set=[(df.iloc[va_idx][feature_cols], y_click.iloc[va_idx])],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(200), lgb.log_evaluation(100)]
        )
        oof_click[va_idx] = model.predict_proba(df.iloc[va_idx][feature_cols])[:, 1]
        models_click.append(model)

    print("✅ Click model OOF AUC:", roc_auc_score(y_click, oof_click))

    # === Blend parametre araması ===
    ws = np.linspace(0.40, 0.95, 36)
    best = (-1, None, None, None)

    for w in ws:
        p = w * oof_order + (1 - w) * oof_click
        sc, auc_c, auc_o = trendyol_score(
            y_click.values, y_order.values, p, session_id.values
        )
        if sc > best[0]:
            best = (sc, w, auc_c, auc_o)

    print(f"\n[BLEND] best TrendyolScore={best[0]:.6f} | w={best[1]:.3f} "
          f"(click={best[2]:.5f}, order={best[3]:.5f})")

    return models_order, models_click, best[1]


if __name__ == "__main__":
    # Train verisini yükle
    df = load_train_data().to_pandas()

    # Hedef kolonlar
    y_order = df["ordered"].astype(int)
    y_click = df["clicked"].astype(int)
    session_id = df["session_id"]
    user_groups = df["user_id_hashed"]

    # Feature listesi (notebook ile aynı olmalı!)
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

    # Eğitim
    models_order, models_click, best_w = train_lightgbm(
        df, feature_cols, y_order, y_click, session_id, user_groups
    )

    # Son fold modellerini kaydet (istersen ensemble de yapabiliriz)
    os.makedirs("models", exist_ok=True)

    for i, m in enumerate(models_order, 1):
        joblib.dump(m, f"models/model_order_f{i}.pkl")
    for i, m in enumerate(models_click, 1):
        joblib.dump(m, f"models/model_click_f{i}.pkl")

    joblib.dump({"w": best_w}, "models/blend_params.pkl")
    print(f"💾 Kaydedildi: {len(models_order)} order ve {len(models_click)} click fold modeli, w={best_w:.3f}")
