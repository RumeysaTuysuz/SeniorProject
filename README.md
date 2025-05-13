# Oda Doluluk Tahmini - Merkezi ve Federated Öğrenme Karşılaştırması

Bu proje, oda doluluk tahmini için klasik makine öğrenmesi modelleri ve federated learning yaklaşımlarını karşılaştırmaktadır.

## Veri Seti

Proje, aşağıdaki özellikleri içeren bir veri seti kullanmaktadır:
- Temperature (Sıcaklık)
- Humidity (Nem)
- Light (Işık)
- CO2
- HumidityRatio (Nem Oranı)
- Occupancy (Doluluk - Hedef değişken)

## Kurulum

1. Gerekli kütüphaneleri yükleyin:
```bash
pip install -r requirements.txt
```

2. Veri setini `data` klasörüne yerleştirin:
- `data/datatraining.txt`

## Çalıştırma

Projeyi çalıştırmak için:
```bash
python occupancy_prediction.py
```

## Özellikler

1. Klasik Makine Öğrenmesi Modelleri:
   - Random Forest
   - K-Nearest Neighbors (KNN)

2. Federated Learning Algoritmaları:
   - FedAvg
   - FedSGD
   - FedNova

3. Değerlendirme Metrikleri:
   - Accuracy (Doğruluk)
   - Precision (Kesinlik)
   - Recall (Duyarlılık)
   - F1 Score
   - Confusion Matrix (Karmaşıklık Matrisi)

## Çıktılar

Program çalıştırıldığında:
1. Veri seti yükleme ve ön işleme adımları
2. Klasik modellerin eğitimi ve değerlendirmesi
3. Federated learning modellerinin eğitimi ve değerlendirmesi
4. Tüm modellerin karşılaştırmalı sonuçları

terminal üzerinde gösterilecektir. 