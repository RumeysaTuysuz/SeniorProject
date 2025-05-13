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
from flwr.common import Metrics
from typing import List, Tuple, Dict
import time
import logging
import warnings
warnings.filterwarnings('ignore')

# Logging ayarları (klasik modeller için)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Classical Models --- 
def train_classical_models(X_train, X_test, y_train, y_test):
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
        
        y_pred = model.predict(X_test)
        
        results[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1 Score': f1_score(y_test, y_pred, zero_division=0),
            'Confusion Matrix': confusion_matrix(y_test, y_pred),
            'Training Time': training_time
        }
        
        logger.info(f"\n{name} Model Sonuçları:")
        logger.info(f"Accuracy: {results[name]['Accuracy']:.4f}")
        logger.info(f"Precision: {results[name]['Precision']:.4f}")
        logger.info(f"Recall: {results[name]['Recall']:.4f}")
        logger.info(f"F1 Score: {results[name]['F1 Score']:.4f}")
        logger.info(f"Training Time: {results[name]['Training Time']:.2f} seconds")
        logger.info(f"Confusion Matrix:\n{results[name]['Confusion Matrix']}")
    
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
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
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
        #print(f"\nRound {round_num} - Training...") # Daha az çıktı için yorum satırı
        
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
            #print(f"  Epoch {epoch + 1}/5 - Loss: {epoch_loss/len(self.train_loader):.4f}") # Daha az çıktı için yorum satırı
        
        #print(f"Round {round_num} - Average Loss: {total_loss/(5*len(self.train_loader)):.4f}") # Daha az çıktı için yorum satırı
        return self.get_parameters({}), len(self.train_loader.dataset), {"loss": total_loss/(5*len(self.train_loader))}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        loss = 0
        correct = 0
        total = 0
        
        round_num = config.get("round", 0)
        #print(f"\nRound {round_num} - Evaluating...") # Daha az çıktı için yorum satırı
        
        with torch.no_grad():
            for batch_x, batch_y in self.val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                output = self.model(batch_x)
                loss += self.criterion(output, batch_y).item()
                predicted = (output > 0.5).float()
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        accuracy = correct / total
        #print(f"Round {round_num} - Validation Loss: {loss/len(self.val_loader):.4f}, Accuracy: {accuracy:.4f}") # Daha az çıktı için yorum satırı
        return float(loss), len(self.val_loader.dataset), {"accuracy": accuracy}

# FedProx Client class for federated learning
class FedProxClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader, device, mu=0.01):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.BCELoss()
        self.mu = mu
        self.global_params = None

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict)
        self.global_params = [p.clone().detach() for p in self.model.parameters()]

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        
        round_num = config.get("round", 0)
        #print(f"\nRound {round_num} - FedProx Training...") # Daha az çıktı için yorum satırı
        
        total_loss = 0
        for epoch in range(5):
            epoch_loss = 0
            for batch_x, batch_y in self.train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(batch_x)
                loss = self.criterion(output, batch_y)
                # Proximal term
                prox_reg = 0.0
                if self.global_params:
                    for w, w0 in zip(self.model.parameters(), self.global_params):
                        prox_reg += ((w - w0) ** 2).sum()
                    loss += (self.mu / 2) * prox_reg
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            total_loss += epoch_loss
            #print(f"  Epoch {epoch + 1}/5 - Loss: {epoch_loss/len(self.train_loader):.4f}") # Daha az çıktı için yorum satırı
        
        #print(f"Round {round_num} - FedProx Average Loss: {total_loss/(5*len(self.train_loader)):.4f}") # Daha az çıktı için yorum satırı
        return self.get_parameters({}), len(self.train_loader.dataset), {"loss": total_loss/(5*len(self.train_loader))}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        loss = 0
        correct = 0
        total = 0
        
        round_num = config.get("round", 0)
        #print(f"\nRound {round_num} - FedProx Evaluating...") # Daha az çıktı için yorum satırı
        
        with torch.no_grad():
            for batch_x, batch_y in self.val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                output = self.model(batch_x)
                loss += self.criterion(output, batch_y).item()
                predicted = (output > 0.5).float()
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        accuracy = correct / total
        #print(f"Round {round_num} - FedProx Validation Loss: {loss/len(self.val_loader):.4f}, Accuracy: {accuracy:.4f}") # Daha az çıktı için yorum satırı
        return float(loss), len(self.val_loader.dataset), {"accuracy": accuracy}

def load_data(train_path='data/datatraining.txt', test_path1='data/datatest.txt', test_path2='data/datatest2.txt'):
    """Eğitim ve birleştirilmiş test dosyalarını yükle, işle ve ölçekle."""
    logger.info("Eğitim ve test verileri yükleniyor...")
    df_train = pd.read_csv(train_path)
    df_test1 = pd.read_csv(test_path1)
    df_test2 = pd.read_csv(test_path2)
    
    # Tarih sütununu kaldır
    df_train = df_train.drop('date', axis=1)
    df_test1 = df_test1.drop('date', axis=1)
    df_test2 = df_test2.drop('date', axis=1)
    
    # Test setlerini birleştir
    df_test = pd.concat([df_test1, df_test2], ignore_index=True)
    
    # Özellikler (X) ve Hedef (y) ayır
    X_train = df_train.drop('Occupancy', axis=1).values
    y_train = df_train['Occupancy'].values
    X_test = df_test.drop('Occupancy', axis=1).values
    y_test = df_test['Occupancy'].values
    
    # Ölçekleyiciyi sadece eğitim verisine göre eğit ve uygula
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) # Test verisini aynı scaler ile transform et
    
    logger.info(f"Eğitim verisi boyutu: {X_train_scaled.shape}")
    logger.info(f"Birleştirilmiş Test verisi boyutu: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def prepare_federated_data(X_train, y_train, n_clients=3):
    """Federated Learning için eğitim verisini client'lara böl"""
    X_train_splits = np.array_split(X_train, n_clients)
    y_train_splits = np.array_split(y_train, n_clients)
    return X_train_splits, y_train_splits

def main():
    # --- 1. Veri Yükleme ve Hazırlama --- 
    X_train_scaled, X_test_scaled, y_train, y_test = load_data()
    
    # --- 2. Merkezi Eğitim (Klasik Modeller) --- 
    classical_results = train_classical_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # --- 3. Federated Learning Hazırlık --- 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train_splits, y_train_splits = prepare_federated_data(X_train_scaled, y_train, n_clients=3)
    num_clients = len(X_train_splits)
    input_size = X_train_scaled.shape[1]
    
    # --- Federated Learning Client Fonksiyonları (Val Loader birleşik test setini kullanır) --- 
    def client_fn_fedavg(cid: str) -> fl.client.Client:
        client_id = int(cid)
        X_train_client = X_train_splits[client_id]
        y_train_client = y_train_splits[client_id]
        
        train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train_client), torch.FloatTensor(y_train_client).reshape(-1, 1)), batch_size=32, shuffle=True)
        val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test_scaled), torch.FloatTensor(y_test).reshape(-1, 1)), batch_size=32)
        
        model = OccupancyNN(input_size).to(device)
        return OccupancyClient(model, train_loader, val_loader, device)
    
    def client_fn_fedprox(cid: str) -> fl.client.Client:
        client_id = int(cid)
        X_train_client = X_train_splits[client_id]
        y_train_client = y_train_splits[client_id]

        train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train_client), torch.FloatTensor(y_train_client).reshape(-1, 1)), batch_size=32, shuffle=True)
        val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test_scaled), torch.FloatTensor(y_test).reshape(-1, 1)), batch_size=32)
        
        model = OccupancyNN(input_size).to(device)
        return FedProxClient(model, train_loader, val_loader, device, mu=0.01)
    
    # --- Federated Learning Strateji ve Simülasyon (Aynı kalır) --- 
    def fit_config(server_round: int):
        return {"epochs": 5, "round": server_round}
    
    def evaluate_config(server_round: int):
        return {"round": server_round}
    
    def aggregate_evaluate_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        if not metrics: return {}
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]
        return {"accuracy": sum(accuracies) / sum(examples)}

    strategy_args = {
        "fraction_fit": 1.0,
        "min_fit_clients": num_clients,
        "fraction_evaluate": 1.0, 
        "min_evaluate_clients": num_clients,
        "min_available_clients": num_clients,
        "on_fit_config_fn": fit_config,
        "on_evaluate_config_fn": evaluate_config,
        "initial_parameters": None,
        "evaluate_metrics_aggregation_fn": aggregate_evaluate_metrics,
    }
    fedavg_strategy = fl.server.strategy.FedAvg(**strategy_args)
    fedprox_strategy = fl.server.strategy.FedAvg(**strategy_args)

    # Simülasyonları Çalıştır
    fl_results = {}
    print("\n--- Running FedAvg Simulation ---")
    history_fedavg = fl.simulation.start_simulation(
        client_fn=client_fn_fedavg,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=fedavg_strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0} if device == torch.device("cpu") else {"num_cpus": 1, "num_gpus": 1.0}
    )
    if history_fedavg.metrics_distributed and 'accuracy' in history_fedavg.metrics_distributed:
         fl_results['FedAvg'] = history_fedavg.metrics_distributed['accuracy'][-1][1] 
    print("--- FedAvg Simulation Finished ---")
    
    print("\n--- Running FedProx Simulation ---")
    history_fedprox = fl.simulation.start_simulation(
        client_fn=client_fn_fedprox,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=fedprox_strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0} if device == torch.device("cpu") else {"num_cpus": 1, "num_gpus": 1.0}
    )
    if history_fedprox.metrics_distributed and 'accuracy' in history_fedprox.metrics_distributed:
        fl_results['FedProx'] = history_fedprox.metrics_distributed['accuracy'][-1][1]
    print("--- FedProx Simulation Finished ---")
    
    # --- 4. Final Sonuçları Tablo Halinde Yazdır --- 
    print("\n--- Final Results Comparison ---")
    print("\n{:<20} | {:<15}".format("Model", "Accuracy"))
    print("-"*38)
    print("{:<38}".format("Classical Models (Centralized):"))
    for model_name, metrics in classical_results.items():
        print("{:<20} | {:<15.4f}".format(model_name, metrics['Accuracy']))
    print("{:<38}".format("\nFederated Learning Models:"))
    for strategy_name, final_accuracy in fl_results.items():
        print("{:<20} | {:<15.4f}".format(strategy_name, final_accuracy))
    print("-"*38)

if __name__ == "__main__":
    main() 