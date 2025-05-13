import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import flwr as fl
from flwr.common import Metrics, Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from typing import List, Tuple, Dict, Optional, Union
import time
import logging
import warnings
warnings.filterwarnings('ignore')

# Logging ayarları (klasik modeller için)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Data Loading and Preprocessing ---
def load_all_data(train_path='data/datatraining.txt', test1_path='data/datatest.txt', test2_path='data/datatest2.txt'):
    """Ayrı eğitim ve iki ayrı test dosyasını yükle, işle ve ölçekle."""
    logger.info("Eğitim ve test verileri yükleniyor...")
    df_train = pd.read_csv(train_path)
    df_test1 = pd.read_csv(test1_path)
    df_test2 = pd.read_csv(test2_path)
    
    # Tarih sütununu tüm dataframelerden kaldır
    for df in [df_train, df_test1, df_test2]:
        if 'date' in df.columns:
            df.drop('date', axis=1, inplace=True)
    
    # Özellikler (X) ve Hedef (y) ayır
    X_train = df_train.drop('Occupancy', axis=1).values
    y_train = df_train['Occupancy'].values
    X_test1 = df_test1.drop('Occupancy', axis=1).values
    y_test1 = df_test1['Occupancy'].values
    X_test2 = df_test2.drop('Occupancy', axis=1).values
    y_test2 = df_test2['Occupancy'].values
    
    # Ölçekleyiciyi sadece eğitim verisine göre eğit
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # Test verilerini aynı scaler ile transform et
    X_test1_scaled = scaler.transform(X_test1)
    X_test2_scaled = scaler.transform(X_test2)
    
    logger.info(f"Eğitim verisi boyutu: {X_train_scaled.shape}")
    logger.info(f"Test Seti 1 (datatest.txt) boyutu: {X_test1_scaled.shape}")
    logger.info(f"Test Seti 2 (datatest2.txt) boyutu: {X_test2_scaled.shape}")
    
    return X_train_scaled, y_train, X_test1_scaled, y_test1, X_test2_scaled, y_test2

# --- Classical Models --- 
def train_classical_models(X_train, y_train, X_test1, y_test1, X_test2, y_test2):
    """
    Klasik makine öğrenmesi modellerini eğit ve değerlendir (Merkezi)
    """
    logger.info("--- Klasik Modeller Eğitiliyor (Merkezi) ---")
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }
    
    results = {}
    
    for name, model in models.items():
        logger.info(f"\n{name} Modeli Eğitiliyor...")
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Test Seti 1 üzerinde değerlendirme
        y_pred_test1 = model.predict(X_test1)
        acc_test1 = accuracy_score(y_test1, y_pred_test1)
        
        # Test Seti 2 üzerinde değerlendirme
        y_pred_test2 = model.predict(X_test2)
        acc_test2 = accuracy_score(y_test2, y_pred_test2)
        
        results[name] = {
            'Accuracy_Test1': acc_test1,
            'Accuracy_Test2': acc_test2,
            'Training Time': training_time
        }
        logger.info(f"{name} - Test Seti 1 Accuracy: {acc_test1:.4f}")
        logger.info(f"{name} - Test Seti 2 Accuracy: {acc_test2:.4f}")
        logger.info(f"{name} - Training Time: {training_time:.2f} seconds")
    
    logger.info("--- Klasik Modellerin Eğitimi Tamamlandı ---")
    return results

# --- Federated Learning Components --- 
# Neural Network model
class OccupancyNN(nn.Module):
    def __init__(self, input_size):
        super(OccupancyNN, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.layer3(x))
        return x

# Client class for federated learning
class OccupancyClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader_test1, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader_test1 = val_loader_test1 # Test Set 1 için
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.BCELoss()

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        
        round_num = config.get("round", 0)
        client_identifier = f"datasize: {len(self.train_loader.dataset) if self.train_loader and self.train_loader.dataset else 'N/A'}"
        logger.info(f"Client ({client_identifier}) - Round {round_num} - Training...")
        
        total_loss = 0
        for epoch in range(5):  # Local epochs
            epoch_loss = 0
            for batch_x, batch_y in self.train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(batch_x)
                loss = self.criterion(output, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            total_loss += epoch_loss
            logger.info(f"  Client - Epoch {epoch + 1}/5 - Loss: {epoch_loss/len(self.train_loader):.4f}")
        
        avg_epoch_loss = total_loss / (5 * len(self.train_loader)) if len(self.train_loader) > 0 else float('inf')
        logger.info(f"Client ({client_identifier}) - Round {round_num} - Average Training Loss: {avg_epoch_loss:.4f}")
        return self.get_parameters({}), len(self.train_loader.dataset), {"loss": avg_epoch_loss}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        loss = 0; correct = 0; total = 0
        round_num = config.get("round", 0)
        client_identifier = f"datasize: {len(self.train_loader.dataset) if self.train_loader and self.train_loader.dataset else 'N/A'}"
        with torch.no_grad():
            for batch_x, batch_y in self.val_loader_test1: # Test Set 1 üzerinde değerlendirme
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                output = self.model(batch_x)
                loss += self.criterion(output, batch_y).item() * batch_x.size(0) # loss'u batch size ile çarp
                predicted = (output > 0.5).float()
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        avg_loss = loss / total if total > 0 else float('inf')
        accuracy = correct / total if total > 0 else 0.0
        logger.info(f"Client ({client_identifier}) - Round {round_num} - Evaluating on Test Set 1: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}")
        return avg_loss, len(self.val_loader_test1.dataset), {"accuracy": accuracy}

# FedProx Client class for federated learning
class FedProxClient(OccupancyClient): # OccupancyClient'tan miras alır
    def __init__(self, model, train_loader, val_loader_test1, device, mu=0.01):
        super().__init__(model, train_loader, val_loader_test1, device)
        self.mu = mu
        self.global_params = None

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.global_params = [p.clone().detach() for p in self.model.parameters()]
        self.model.train()
        round_num = config.get("round", 0)
        client_identifier = f"datasize: {len(self.train_loader.dataset) if self.train_loader and self.train_loader.dataset else 'N/A'}"
        logger.info(f"FedProx Client ({client_identifier}) - Round {round_num} - Training...")
        total_loss = 0
        for epoch in range(5):
            epoch_loss = 0
            for batch_x, batch_y in self.train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(batch_x)
                loss = self.criterion(output, batch_y)
                prox_reg = 0.0
                if self.global_params:
                    for w, w0 in zip(self.model.parameters(), self.global_params):
                        prox_reg += ((w - w0) ** 2).sum()
                    loss += (self.mu / 2) * prox_reg
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            total_loss += epoch_loss
            logger.info(f"  FedProx Client - Epoch {epoch + 1}/5 - Loss: {epoch_loss/len(self.train_loader):.4f}")
        
        avg_epoch_loss = total_loss / (5 * len(self.train_loader)) if len(self.train_loader) > 0 else float('inf')
        logger.info(f"FedProx Client ({client_identifier}) - Round {round_num} - Average Training Loss: {avg_epoch_loss:.4f}")
        return self.get_parameters({}), len(self.train_loader.dataset), {"loss": avg_epoch_loss}

def prepare_federated_data(X_train, y_train, n_clients=3):
    """Federated Learning için eğitim verisini client'lara böl"""
    X_train_splits = np.array_split(X_train, n_clients)
    y_train_splits = np.array_split(y_train, n_clients)
    return X_train_splits, y_train_splits

def main():
    # --- 1. Veri Yükleme ve Hazırlama --- 
    X_train_scaled, y_train, X_test1_scaled, y_test1, X_test2_scaled, y_test2 = load_all_data()
    
    # --- 2. Merkezi Eğitim (Klasik Modeller) --- 
    classical_results = train_classical_models(X_train_scaled, y_train, X_test1_scaled, y_test1, X_test2_scaled, y_test2)
    
    # --- 3. Federated Learning Hazırlık --- 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train_splits, y_train_splits = prepare_federated_data(X_train_scaled, y_train, n_clients=3)
    num_clients = len(X_train_splits)
    input_size = X_train_scaled.shape[1]
    
    # --- Federated Learning Client Fonksiyonları (Val Loader güncellendi) --- 
    def client_fn_fedavg(cid: str) -> fl.client.Client:
        client_id = int(cid)
        X_train_client = X_train_splits[client_id]
        y_train_client = y_train_splits[client_id]
        
        # Eğitim verisi için DataLoader
        train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train_client), torch.FloatTensor(y_train_client).reshape(-1, 1)), batch_size=32, shuffle=True)
        
        # Değerlendirme için global test setini kullanan DataLoader
        val_loader_test1 = DataLoader(TensorDataset(torch.FloatTensor(X_test1_scaled), torch.FloatTensor(y_test1).reshape(-1, 1)), batch_size=32)
        
        model = OccupancyNN(input_size).to(device)
        return OccupancyClient(model, train_loader, val_loader_test1, device)
    
    def client_fn_fedprox(cid: str) -> fl.client.Client:
        client_id = int(cid)
        X_train_client = X_train_splits[client_id]
        y_train_client = y_train_splits[client_id]

        # Eğitim verisi için DataLoader
        train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train_client), torch.FloatTensor(y_train_client).reshape(-1, 1)), batch_size=32, shuffle=True)
        
        # Değerlendirme için global test setini kullanan DataLoader
        val_loader_test1 = DataLoader(TensorDataset(torch.FloatTensor(X_test1_scaled), torch.FloatTensor(y_test1).reshape(-1, 1)), batch_size=32)
        
        model = OccupancyNN(input_size).to(device)
        return FedProxClient(model, train_loader, val_loader_test1, device, mu=0.01)
    
    # --- Federated Learning Strateji ve Simülasyon (Aynı kalır) --- 
    def fit_config(server_round: int):
        return {"epochs": 5, "round": server_round}
    
    def evaluate_config(server_round: int):
        return {"round": server_round}
    
    def aggregate_client_metrics_test1(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        if not metrics: return {}
        # Gelen metriklerin yapısını kontrol et
        # logger.info(f"Raw client metrics for Test1 aggregation: {metrics}") 
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics if "accuracy" in m]
        examples = [num_examples for num_examples, m in metrics if "accuracy" in m]
        aggregated_accuracy = sum(accuracies) / sum(examples) if sum(examples) > 0 else 0.0
        # logger.info(f"Aggregated Test1 Accuracy: {aggregated_accuracy}")
        return {"accuracy_test1": aggregated_accuracy}

    # Sunucu Tarafı Değerlendirme Fonksiyonu (Test Seti 2 için)
    def get_evaluate_fn_test2(model_nn: torch.nn.Module, test_data: torch.Tensor, test_labels: torch.Tensor, device: torch.device):
        def evaluate(server_round: int, parameters: Union[Parameters, List[np.ndarray]], config: Dict[str, Union[int, float, str]]) -> Optional[Tuple[float, Dict[str, float]]]:
            # Gelen parametrelerin formatını kontrol et
            if isinstance(parameters, list):
                ndarrays = parameters # Zaten ndarray listesi ise doğrudan kullan
            elif hasattr(parameters, 'tensors'): # Parameters nesnesi ise dönüştür
                ndarrays = parameters_to_ndarrays(parameters)
            else:
                logger.error(f"evaluate_fn_test2 bilinmeyen parametre türü aldı: {type(parameters)}")
                return None, {}

            # ndarrays'ın bir numpy.ndarray listesi olduğundan emin ol
            if not isinstance(ndarrays, list) or not all(isinstance(arr, np.ndarray) for arr in ndarrays):
                logger.error(f"evaluate_fn_test2: ndarrays bir numpy.ndarray listesi değil. Tür: {type(ndarrays)}")
                if isinstance(ndarrays, list) and ndarrays:
                    logger.error(f"İlk elemanın türü: {type(ndarrays[0])}")
                return None, {}

            model_state_dict_keys = list(model_nn.state_dict().keys())
            if len(ndarrays) != len(model_state_dict_keys):
                logger.error(
                    f"Katman sayılarında uyuşmazlık: parametreler ({len(ndarrays)}) vs model ({len(model_state_dict_keys)})"
                )
                return None, {}

            params_dict = zip(model_state_dict_keys, ndarrays)
            try:
                state_dict = {k: torch.tensor(v) for k, v in params_dict}
                model_nn.load_state_dict(state_dict)
            except Exception as e:
                logger.error(f"model_nn.load_state_dict sırasında hata: {e}")
                logger.error(f"  Model anahtarları: {model_state_dict_keys}")
                logger.error(f"  Sağlanan parametre sayısı: {len(ndarrays)}")
                return None, {}

            model_nn.to(device)
            model_nn.eval()
            test_loader = DataLoader(TensorDataset(test_data, test_labels.reshape(-1,1)), batch_size=32)
            loss = 0; correct = 0; total = 0
            criterion = nn.BCELoss()
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model_nn(images)
                    loss += criterion(outputs, labels).item() * images.size(0)
                    predicted = (outputs > 0.5).float()
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            avg_loss = loss / total if total > 0 else float('inf')
            accuracy = correct / total if total > 0 else 0.0
            logger.info(f"Sunucu Değerlendirmesi (Test Seti 2) - Round {server_round}: Accuracy={accuracy:.4f}, Loss={avg_loss:.4f}")
            return avg_loss, {"accuracy_test2": accuracy}
        return evaluate
    
    fl_results = {}
    strategies_to_run = {
        "FedAvg": (client_fn_fedavg, fl.server.strategy.FedAvg),
        "FedProx": (client_fn_fedprox, fl.server.strategy.FedAvg) # FedProx için server tarafı FedAvg
    }

    # Başlangıç parametrelerini bir kere oluştur
    initial_model = OccupancyNN(input_size).to(device)
    initial_params_ndarrays = [val.cpu().numpy() for _, val in initial_model.state_dict().items()]
    fl_initial_parameters = ndarrays_to_parameters(initial_params_ndarrays)

    for name, (client_fn, server_strategy_class) in strategies_to_run.items():
        logger.info(f"\n--- Running {name} Simulation ---")
        temp_model_for_eval_fn = OccupancyNN(input_size) # Her simülasyon için yeni evaluate_fn modeli
        evaluate_fn_test2 = get_evaluate_fn_test2(temp_model_for_eval_fn, torch.FloatTensor(X_test2_scaled), torch.FloatTensor(y_test2), device)
        
        strategy = server_strategy_class(
            fraction_fit=1.0, min_fit_clients=num_clients,
            fraction_evaluate=1.0, min_evaluate_clients=num_clients, # Client'lar Test Set 1'de evaluate eder
            min_available_clients=num_clients,
            on_fit_config_fn=fit_config,
            evaluate_fn=evaluate_fn_test2, # Sunucu Test Set 2'de evaluate eder
            evaluate_metrics_aggregation_fn=aggregate_client_metrics_test1, # Client metriklerini (Test Set 1 accuracy) birleştirir
            initial_parameters=fl_initial_parameters # Hazırlanan başlangıç parametrelerini kullan
        )
        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=num_clients,
            config=fl.server.ServerConfig(num_rounds=10),
            strategy=strategy,
            client_resources={"num_cpus": 1, "num_gpus": 0.0} if device == torch.device("cpu") else {"num_cpus": 1, "num_gpus": 1.0}
        )
        fl_results[name] = {}
        if history.metrics_distributed and 'accuracy_test1' in history.metrics_distributed and history.metrics_distributed['accuracy_test1']:
            fl_results[name]['Accuracy_Test1'] = history.metrics_distributed['accuracy_test1'][-1][1]
        else:
            fl_results[name]['Accuracy_Test1'] = 0.0 # Eğer metrik yoksa 0 ata
            logger.warning(f"{name} için Test Set 1 dağıtık metrikleri bulunamadı veya boş.")

        if history.metrics_centralized and 'accuracy_test2' in history.metrics_centralized and history.metrics_centralized['accuracy_test2']:
            fl_results[name]['Accuracy_Test2'] = history.metrics_centralized['accuracy_test2'][-1][1]
        else:
            fl_results[name]['Accuracy_Test2'] = 0.0 # Eğer metrik yoksa 0 ata
            logger.warning(f"{name} için Test Set 2 merkezi metrikleri bulunamadı veya boş.")
        logger.info(f"--- {name} Simulation Finished ---")

    # 4. Final Sonuçları Yazdır
    print("\n--- Final Results Comparison ---")
    header = "{:<20} | {:<20} | {:<20}".format("Model", "Accuracy (TestSet1)", "Accuracy (TestSet2)")
    print(header)
    print("-" * len(header))
    print("{:<62}".format("Classical Models (Centralized):"))
    for model_name, metrics in classical_results.items():
        print("{:<20} | {:<20.4f} | {:<20.4f}".format(model_name, metrics.get('Accuracy_Test1', 0.0), metrics.get('Accuracy_Test2', 0.0)))
    
    print("{:<62}".format("\nFederated Learning Models:"))
    for strategy_name, metrics in fl_results.items():
        print("{:<20} | {:<20.4f} | {:<20.4f}".format(strategy_name, metrics.get('Accuracy_Test1', 0.0), metrics.get('Accuracy_Test2', 0.0)))
    print("-" * len(header))

if __name__ == "__main__":
    main() 