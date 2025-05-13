import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time
import logging

# Logging ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_preprocess_data(file_path):
    """
    Veri setini yükle ve ön işleme yap
    """
    logger.info("Veri seti yükleniyor...")
    df = pd.read_csv(file_path)
    
    # Date kolonunu çıkar
    df = df.drop('date', axis=1)
    
    # Özellikler ve hedef değişkeni ayır
    X = df.drop('Occupancy', axis=1)
    y = df['Occupancy']
    
    # Veriyi eğitim ve test setlerine ayır
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Ölçekleme
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def train_classical_models(X_train, X_test, y_train, y_test):
    """
    Klasik makine öğrenmesi modellerini eğit ve değerlendir
    """
    logger.info("Klasik modeller eğitiliyor...")
    
    # Modelleri tanımla
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }
    
    results = {}
    
    for name, model in models.items():
        # Modeli eğit
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Tahminler
        y_pred = model.predict(X_test)
        
        # Metrikleri hesapla
        results[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred),
            'Confusion Matrix': confusion_matrix(y_test, y_pred),
            'Training Time': training_time
        }
        
        # Sonuçları yazdır
        logger.info(f"\n{name} Model Sonuçları:")
        logger.info(f"Accuracy: {results[name]['Accuracy']:.4f}")
        logger.info(f"Precision: {results[name]['Precision']:.4f}")
        logger.info(f"Recall: {results[name]['Recall']:.4f}")
        logger.info(f"F1 Score: {results[name]['F1 Score']:.4f}")
        logger.info(f"Training Time: {results[name]['Training Time']:.2f} seconds")
        logger.info(f"Confusion Matrix:\n{results[name]['Confusion Matrix']}")
    
    return results

def main():
    """
    Ana fonksiyon
    """
    try:
        # Veri setini yükle
        X_train, X_test, y_train, y_test = load_and_preprocess_data('data/datatraining.txt')
        
        # Modelleri eğit ve değerlendir
        results = train_classical_models(X_train, X_test, y_train, y_test)
        
        logger.info("\nTüm modellerin eğitimi ve değerlendirmesi tamamlandı.")
        
    except Exception as e:
        logger.error(f"Bir hata oluştu: {str(e)}")

if __name__ == "__main__":
    main() 