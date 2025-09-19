# Trendyol E-Ticaret Hackathonu 2025 - Çözüm Paketi

Bu repo, Kaggle aşamasında elde edilen Private LB skorunu yeniden üretebilmek için hazırlanmıştır.  
Kodlar **veri hazırlama**, **model eğitimi** ve **tahmin üretme** adımlarına ayrılmıştır.  

---

## 📂 Proje Yapısı

```
solution/
│
├── requirements.txt        # Kullanılan kütüphaneler
│
├── src/                    # Kaynak kodlar
│   ├── __init__.py
│   ├── utils.py            # Yardımcı fonksiyonlar (AUC, skor metrikleri, scaler)
│   ├── prepare_data.py     # Veri hazırlama rutini
│   ├── train.py            # Model eğitme rutini
│   └── predict.py          # Tahmin üretme rutini
│   └── app.py              # Demo uygulaması ve ranking sistemi (Streamlit)
│
├── features/               # (Çalıştırınca oluşacak) İşlenmiş veriler
│   ├── train_features.parquet
│   └── test_features.parquet
│
├── models/                 # (Çalıştırınca oluşacak) Eğitilmiş modeller
│   └── model.pkl
│
└── submission.csv          # (Çalıştırınca oluşacak) Kaggle submission dosyası
```

---

## ⚙️ Ortam Kurulumu

Öncelikle Python ortamını kurun:

```bash
pip install -r requirements.txt
```

---

##  Çalıştırma Adımları

1. **Veri Hazırlama**  
   Kaggle dataset’inden verileri okuyup feature engineering yapmak:  
   ```bash
   python -m src.prepare_data
   ```
   Çıktı: `features/train_features.parquet`, `features/test_features.parquet`

2. **Model Eğitimi**  
   Hazırlanan feature set üzerinde LightGBM modeli eğitmek:  
   ```bash
   python -m src.train
   ```
   Çıktı: `models/model.pkl`

3. **Tahmin Üretme**  
   Eğitilmiş modeli kullanarak test seti üzerinde tahmin üretmek:  
   ```bash
   python -m src.predict
   ```
   Çıktı: `submission.csv`

---
