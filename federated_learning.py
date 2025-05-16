import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
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
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
import fnmatch

warnings.filterwarnings('ignore')

# Logging settings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Function to Save Results ---
def plot_confusion_matrix(cm, classes, title, filename):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(filename)
    plt.close()

# --- Helper Function to Print Table Sections ---
def print_table_section(title: str, headers: List[str], data_rows: List[List[str]], logger_instance):
    logger_instance.info(f"\n{title}")
    if not data_rows:
        logger_instance.info("  (Bu bölüm için veri yok)")
        return

    col_widths = [len(header) for header in headers]
    for row_data in data_rows:
        for i, cell in enumerate(row_data):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    # Create separator line
    separator_parts = ["-" * width for width in col_widths]
    separator = "-+-".join(separator_parts)
    if not col_widths: # Prevent error if col_widths is empty
        separator = ""

    # Print header row
    header_line_parts = [f"{headers[i]:<{col_widths[i]}}" for i in range(len(headers))]
    logger_instance.info(" | ".join(header_line_parts))
    if separator: # Only print separator if it's meaningful
        logger_instance.info(separator)

    # Print data rows
    for row_data in data_rows:
        row_line_parts = [f"{str(row_data[i]):<{col_widths[i]}}" for i in range(len(row_data))]
        logger_instance.info(" | ".join(row_line_parts))
    
    if separator: # Add bottom line only if separator is meaningful
        # Adjust length for " | " separators between columns
        logger_instance.info("-" * (sum(col_widths) + (len(col_widths) -1) * 3 if col_widths else 0) )

# --- Data Loading and Preprocessing ---
def load_all_data(train_path='data/datatraining.txt', test1_path='data/datatest.txt', test2_path='data/datatest2.txt'):
    """Load, process, and scale separate training and two separate test files."""
    logger.info("Loading training and test data...")
    df_train = pd.read_csv(train_path)
    df_test1 = pd.read_csv(test1_path)
    df_test2 = pd.read_csv(test2_path)
    
    # Remove date column from all dataframes
    for df in [df_train, df_test1, df_test2]:
        if 'date' in df.columns:
            df.drop('date', axis=1, inplace=True)
    
    # Separate Features (X) and Target (y)
    X_train = df_train.drop('Occupancy', axis=1).values
    y_train = df_train['Occupancy'].values
    X_test1 = df_test1.drop('Occupancy', axis=1).values
    y_test1 = df_test1['Occupancy'].values
    X_test2 = df_test2.drop('Occupancy', axis=1).values
    y_test2 = df_test2['Occupancy'].values
    
    # Fit scaler only on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # Transform test data with the same scaler
    X_test1_scaled = scaler.transform(X_test1)
    X_test2_scaled = scaler.transform(X_test2)
    
    logger.info(f"Training data size: {X_train_scaled.shape}")
    logger.info(f"Test Set 1 (datatest.txt) size: {X_test1_scaled.shape}")
    logger.info(f"Test Set 2 (datatest2.txt) size: {X_test2_scaled.shape}")
    
    return X_train_scaled, y_train, X_test1_scaled, y_test1, X_test2_scaled, y_test2

# --- Classical Models --- 
def train_classical_models(X_train, y_train, X_test1, y_test1, X_test2, y_test2):
    """
    Train and evaluate classical machine learning models (Centralized)
    """
    logger.info("--- Training Classical Models (Centralized) ---")
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=20),
        'XGBoost': XGBClassifier(n_estimators=10, max_depth=2, learning_rate=0.3, random_state=42, use_label_encoder=False, eval_metric='logloss'),
        'Naive Bayes': GaussianNB(),
        'Perceptron': Perceptron(max_iter=50, eta0=0.5, random_state=42),
        'SVM': SVC(kernel='rbf', C=1.0, random_state=42, probability=True),
        'Decision Tree (CART)': DecisionTreeClassifier(random_state=42)
    }
    
    results = {}
    class_names = ['Not Occupied', 'Occupied']
    
    for name, model in models.items():
        logger.info(f"\nTraining {name} Model...")
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Evaluate on Test Set 1 (datatest.txt - now validation, CM removed)
        y_pred_test1 = model.predict(X_test1)
        acc_test1 = accuracy_score(y_test1, y_pred_test1)
        # CM for Test Set 1 removed as per user request
        
        # Evaluate on Test Set 2 (datatest2.txt - this is our Test Set)
        y_pred_test2 = model.predict(X_test2)
        acc_test2 = accuracy_score(y_test2, y_pred_test2)
        cm_test2 = confusion_matrix(y_test2, y_pred_test2)
        plot_confusion_matrix(cm_test2, classes=class_names,
                              title=f'{name} - CM (Test Set - datatest2.txt)', # Back to English
                              filename=os.path.join('results', f'cm_{name.replace(" ", "_")}_test_set.png')) # General test set naming
        
        results[name] = {
            'Accuracy_Validation': acc_test1, # Renamed from Accuracy_Test1
            'Accuracy_Test': acc_test2,       # Renamed from Accuracy_Test2
            'Training Time': training_time
        }
        logger.info(f"{name} - Validation Set (datatest.txt) Accuracy: {acc_test1:.4f}") # Updated log message
        logger.info(f"{name} - Test Set (datatest2.txt) Accuracy: {acc_test2:.4f}")       # Updated log message
    
    logger.info("--- Classical Models Training Completed ---")
    return results

# --- Federated Learning Components --- 
# Neural Network model
class OccupancyNN(nn.Module):
    def __init__(self, input_size):
        super(OccupancyNN, self).__init__()
        self.layer1 = nn.Linear(input_size, 8)
        self.layer2 = nn.Linear(8, 4)
        self.layer3 = nn.Linear(4, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.layer3(x))
        return x

# Client class for federated learning
class OccupancyClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader_test1, device, pos_weight_tensor=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader_test1 = val_loader_test1 # For Test Set 1
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001) # Adjusted learning rate
        self.criterion = nn.BCELoss() # Instantiate first
        if pos_weight_tensor is not None:
            self.criterion.pos_weight = pos_weight_tensor # Set pos_weight as an attribute
        self.client_id_str = "UnknownClient"
        if self.train_loader and self.train_loader.dataset:
            self.client_id_str = f"datasize_{len(self.train_loader.dataset)}"

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        client_identifier = self.client_id_str
        try:
            self.set_parameters(parameters)
            self.model.train()
            
            round_num = config.get("round", 0)
            logger.info(f"Client ({client_identifier}) - Round {round_num} - Training...")
            
            total_loss = 0
            # Use the number of epochs from fit_config
            epochs = config.get("epochs", 1) # Get epoch from fit_config, else 1

            if not self.train_loader or len(self.train_loader.dataset) == 0:
                logger.warning(f"Client ({client_identifier}) - Round {round_num} - Training data empty or DataLoader missing. Skipping fit.")
                return self.get_parameters({}), 0, {"loss": float('inf'), "fit_error": "empty_train_data"}

            for epoch in range(epochs):
                epoch_loss = 0
                batches_processed = 0
                for batch_x, batch_y in self.train_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    self.optimizer.zero_grad()
                    output = self.model(batch_x)
                    loss = self.criterion(output, batch_y)
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss.item()
                    batches_processed += 1
                if batches_processed > 0:
                    logger.info(f"  Client ({client_identifier}) - Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss/batches_processed:.4f}")
                else:
                    logger.warning(f"  Client ({client_identifier}) - Epoch {epoch + 1}/{epochs} - No batches processed.")
                    total_loss += float('inf') # If no batch processed, set loss to infinity
                    break # Exit this epoch
                total_loss += epoch_loss / batches_processed if batches_processed > 0 else float('inf')
            avg_epoch_loss = total_loss / epochs if epochs > 0 and batches_processed > 0 else float('inf')
            logger.info(f"Client ({client_identifier}) - Round {round_num} - Average Training Loss: {avg_epoch_loss:.4f}")
            return self.get_parameters({}), len(self.train_loader.dataset), {"loss": avg_epoch_loss}
        except Exception as e:
            logger.error(f"CLIENT FIT ERROR ({client_identifier}) - Round {config.get('round', 0)}: {e}", exc_info=True)
            # Return empty parameters and 0 samples with an error metric in case of error
            return [np.array([]) for _ in self.model.state_dict().values()], 0, {"loss": float('inf'), "fit_error": str(e)}

    def evaluate(self, parameters, config):
        client_identifier = self.client_id_str
        try:
            self.set_parameters(parameters)
            self.model.eval()
            loss = 0; correct = 0; total = 0
            round_num = config.get("round", 0)
            
            if not self.val_loader_test1 or len(self.val_loader_test1.dataset) == 0:
                logger.warning(f"Client ({client_identifier}) - Round {round_num} - Evaluation data (Test Set 1) empty or DataLoader missing. Skipping evaluate.")
                return float('inf'), 0, {"accuracy": 0.0, "loss": float('inf'), "eval_error": "empty_val_data"}

            with torch.no_grad():
                batches_processed = 0
                for batch_x, batch_y in self.val_loader_test1: # Evaluate on Test Set 1
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    output = self.model(batch_x)
                    batch_loss = self.criterion(output, batch_y).item()
                    loss += batch_loss * batch_x.size(0)
                    predicted = (output > 0.5).float()
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
                    batches_processed += 1

            if batches_processed == 0 or total == 0:
                logger.warning(f"Client ({client_identifier}) - Round {round_num} - No batches processed or total samples zero during evaluation.")
                return float('inf'), 0, {"accuracy": 0.0, "loss": float('inf'), "eval_error": "no_samples_evaluated"}

            avg_loss = loss / total
            accuracy = correct / total
            logger.info(f"Client ({client_identifier}) - Round {round_num} - Evaluating on Test Set 1: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}")
            return avg_loss, len(self.val_loader_test1.dataset), {"accuracy": accuracy, "loss": avg_loss}
        except Exception as e:
            logger.error(f"CLIENT EVALUATE ERROR ({client_identifier}) - Round {config.get('round', 0)}: {e}", exc_info=True)
            return float('inf'), 0, {"accuracy": 0.0, "loss": float('inf'), "eval_error": str(e)}

# FedProx Client class for federated learning
class FedProxClient(OccupancyClient): # Inherits from OccupancyClient
    def __init__(self, model, train_loader, val_loader_test1, device, mu=0.1, pos_weight_tensor=None):
        super().__init__(model, train_loader, val_loader_test1, device, pos_weight_tensor) # Pass pos_weight_tensor to parent
        self.mu = mu
        # Optimizer and criterion are set in parent, but we override optimizer for FedProx
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005) # Different LR for FedProx
        if self.train_loader and self.train_loader.dataset: # update client_id_str
            self.client_id_str = f"FedProx_datasize_{len(self.train_loader.dataset)}"

    def fit(self, parameters, config):
        client_identifier = self.client_id_str # Inherited client_id_str is used
        try:
            self.set_parameters(parameters)
            # Store global_params as NumPy arrays, not PyTorch tensors
            self.global_params_ndarrays = [p.copy() for p in parameters]
            self.model.train()
            round_num = config.get("round", 0)
            logger.info(f"FedProx Client ({client_identifier}) - Round {round_num} - Training...")
            total_loss = 0
            epochs = config.get("epochs", 1)

            if not self.train_loader or len(self.train_loader.dataset) == 0:
                logger.warning(f"FedProx Client ({client_identifier}) - Round {round_num} - Training data empty. Skipping fit.")
                return self.get_parameters({}), 0, {"loss": float('inf'), "fit_error": "empty_train_data"}

            for epoch in range(epochs):
                epoch_loss = 0
                batches_processed = 0
                for batch_x, batch_y in self.train_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    self.optimizer.zero_grad()
                    output = self.model(batch_x)
                    base_loss = self.criterion(output, batch_y)
                    loss = base_loss
                    prox_reg = 0.0
                    if self.global_params_ndarrays:
                        for param_idx, (w, w0_nd) in enumerate(zip(self.model.parameters(), self.global_params_ndarrays)):
                            w0 = torch.tensor(w0_nd, device=self.device) # Convert NumPy to tensor
                            if w.shape == w0.shape:
                                prox_reg += ((w - w0) ** 2).sum()
                            else:
                                logger.error(f"FedProx Client ({client_identifier}) - Shape mismatch in prox_reg: param {param_idx} model {w.shape} vs global {w0.shape}")
                        loss += (self.mu / 2) * prox_reg
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss.item()
                    batches_processed +=1
                if batches_processed > 0:
                    logger.info(f"  FedProx Client ({client_identifier}) - Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss/batches_processed:.4f}")
                else:
                    logger.warning(f"  FedProx Client ({client_identifier}) - Epoch {epoch + 1}/{epochs} - No batches processed.")
                    total_loss += float('inf')
                    break
                total_loss += epoch_loss / batches_processed if batches_processed > 0 else float('inf')
            avg_epoch_loss = total_loss / epochs if epochs > 0 and batches_processed > 0 else float('inf')
            logger.info(f"FedProx Client ({client_identifier}) - Round {round_num} - Average Training Loss: {avg_epoch_loss:.4f}")
            return self.get_parameters({}), len(self.train_loader.dataset), {"loss": avg_epoch_loss}
        except Exception as e:
            logger.error(f"FEDPROX CLIENT FIT ERROR ({client_identifier}) - Round {config.get('round',0)}: {e}", exc_info=True)
            return [np.array([]) for _ in self.model.state_dict().values()], 0, {"loss": float('inf'), "fit_error": str(e)}

def prepare_federated_data(X_train, y_train, n_clients=3):
    """Split training data for Federated Learning among clients"""
    X_train_splits = np.array_split(X_train, n_clients)
    y_train_splits = np.array_split(y_train, n_clients)
    return X_train_splits, y_train_splits

def main():
    # Create results directory
    if not os.path.exists('results'):
        os.makedirs('results')
        logger.info("'results' directory created.")
    else:
        logger.info("'results' directory already exists.")

    # Delete old confusion matrix files
    logger.info("Deleting old confusion matrix files from 'results' directory...")
    deleted_files_count = 0
    patterns_to_delete = ["cm_*_test1.png", "cm_*_test2.png", "cm_*_validation.png"]
    try:
        for filename in os.listdir("results"):
            for pattern in patterns_to_delete:
                if fnmatch.fnmatch(filename, pattern):
                    try:
                        filepath = os.path.join("results", filename)
                        os.remove(filepath)
                        logger.info(f"  Deleted: {filename}")
                        deleted_files_count += 1
                        break # Move to next file in directory once matched and deleted
                    except OSError as e:
                        logger.error(f"  Error deleting {filename}: {e}")
        if deleted_files_count == 0:
            logger.info("  No old confusion matrix files matching patterns found to delete.")
        else:
            logger.info(f"  Successfully deleted {deleted_files_count} old file(s).")
    except Exception as e:
        logger.error(f"Error listing or processing files in 'results' directory for deletion: {e}")


    # --- 1. Data Loading and Preparation --- 
    X_train_scaled, y_train, X_test1_scaled, y_test1, X_test2_scaled, y_test2 = load_all_data()
    
    # --- 2. Centralized Training (Classical Models) --- 
    classical_results = train_classical_models(X_train_scaled, y_train, X_test1_scaled, y_test1, X_test2_scaled, y_test2)
    
    # --- 3. Federated Learning Preparation --- 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = X_train_scaled.shape[1]
    class_names_fl = ['Not Occupied', 'Occupied'] # Class names for FL CM

    # Calculate pos_weight for BCELoss to handle class imbalance
    num_positives = np.sum(y_train)
    num_negatives = len(y_train) - num_positives
    pos_weight_value = num_negatives / num_positives if num_positives > 0 else 1.0 # Avoid division by zero
    pos_weight_tensor = torch.tensor([pos_weight_value], device=device)
    logger.info(f"Calculated pos_weight for BCELoss: {pos_weight_value:.4f}")

    # Create initial parameters once (same for all FL simulations)
    initial_model_for_fl = OccupancyNN(input_size).to(device)
    initial_params_ndarrays = [val.cpu().numpy() for _, val in initial_model_for_fl.state_dict().items()]
    fl_initial_parameters = ndarrays_to_parameters(initial_params_ndarrays)

    client_counts_to_test = [2, 3, 5, 10] # 10 istemci eklendi
    all_fl_results_by_client_count = {}
    # Update fit_config to include number of epochs
    num_local_epochs = 5 # Adjusted local epoch count

    for n_c in client_counts_to_test:
        logger.info(f"\n--- Starting Federated Learning Simulation: {n_c} Clients ---")
        
        X_train_splits, y_train_splits = prepare_federated_data(X_train_scaled, y_train, n_clients=n_c)
        current_num_clients = len(X_train_splits)

        def current_client_fn_fedavg(cid: str) -> fl.client.Client:
            client_id = int(cid)
            X_train_client = X_train_splits[client_id]
            y_train_client = y_train_splits[client_id]
            train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train_client), torch.FloatTensor(y_train_client).reshape(-1, 1)), batch_size=32, shuffle=True)
            val_loader_test1 = DataLoader(TensorDataset(torch.FloatTensor(X_test1_scaled), torch.FloatTensor(y_test1).reshape(-1, 1)), batch_size=32)
            model = OccupancyNN(input_size).to(device)
            return OccupancyClient(model, train_loader, val_loader_test1, device, pos_weight_tensor=pos_weight_tensor)

        def current_client_fn_fedprox(cid: str) -> fl.client.Client:
            client_id = int(cid)
            X_train_client = X_train_splits[client_id]
            y_train_client = y_train_splits[client_id]
            train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train_client), torch.FloatTensor(y_train_client).reshape(-1, 1)), batch_size=32, shuffle=True)
            val_loader_test1 = DataLoader(TensorDataset(torch.FloatTensor(X_test1_scaled), torch.FloatTensor(y_test1).reshape(-1, 1)), batch_size=32)
            model = OccupancyNN(input_size).to(device)
            return FedProxClient(model, train_loader, val_loader_test1, device, mu=0.1, pos_weight_tensor=pos_weight_tensor)

        current_strategies_to_run = {
            "FedAvg": (current_client_fn_fedavg, fl.server.strategy.FedAvg),
            "FedProx": (current_client_fn_fedprox, fl.server.strategy.FedAvg)
        }
        
        current_run_fl_results = {}
        for name, (client_fn_for_strategy, server_strategy_class) in current_strategies_to_run.items():
            logger.info(f"\n--- Running {name} Simulation with {current_num_clients} clients ---")
            
            fl_start_time = time.time() # Record start time for FL strategy

            # Recreate evaluate_fn and get_last_params_fn for each strategy
            temp_model_for_eval_fn = OccupancyNN(input_size)
            evaluate_fn_test2, get_last_params_fn = get_evaluate_fn_test2_and_param_capture(
                temp_model_for_eval_fn, torch.FloatTensor(X_test2_scaled), 
                torch.FloatTensor(y_test2), device, name, current_num_clients
            )
            
            # Define on_fit_config_fn with correct epoch count for each strategy
            def current_fit_config(server_round: int):
                return {"epochs": num_local_epochs, "round": server_round}

            strategy = server_strategy_class(
                fraction_fit=1.0, min_fit_clients=current_num_clients,
                fraction_evaluate=1.0, min_evaluate_clients=current_num_clients,
                min_available_clients=current_num_clients,
                on_fit_config_fn=current_fit_config, # Updated fit_config
                evaluate_fn=evaluate_fn_test2, 
                evaluate_metrics_aggregation_fn=aggregate_client_metrics_test1,
                initial_parameters=fl_initial_parameters 
            )
            
            # Define ray_init_args to set object_store_memory
            ray_init_args_config = {
                "object_store_memory": 100 * 1024 * 1024,  # 100 MB
                "ignore_reinit_error": True # Add this to avoid issues if Ray is already initialized
            }

            history = fl.simulation.start_simulation(
                client_fn=client_fn_for_strategy,
                num_clients=current_num_clients,
                config=fl.server.ServerConfig(num_rounds=10), # Adjusted num_rounds
                strategy=strategy, 
                client_resources={"num_cpus": 1, "num_gpus": 0.0}, # CPU only for simplicity
                ray_init_args=ray_init_args_config
            )
            
            fl_end_time = time.time() # Record end time for FL strategy
            fl_training_time = fl_end_time - fl_start_time # Calculate training time

            current_run_fl_results[name] = {}
            # Use new metric keys for validation (client-side on datatest.txt)
            if history.metrics_distributed and 'accuracy_validation' in history.metrics_distributed and history.metrics_distributed['accuracy_validation']:
                current_run_fl_results[name]['Accuracy_Validation'] = history.metrics_distributed['accuracy_validation'][-1][1]
            else:
                current_run_fl_results[name]['Accuracy_Validation'] = 0.0 
                logger.warning(f"Validation accuracy (datatest.txt) distributed metrics not found or empty for {name} ({current_num_clients} clients).")

            if history.metrics_distributed and 'loss_validation' in history.metrics_distributed and history.metrics_distributed['loss_validation']:
                current_run_fl_results[name]['Loss_Validation'] = history.metrics_distributed['loss_validation'][-1][1]
            else:
                current_run_fl_results[name]['Loss_Validation'] = float('inf')
                logger.warning(f"Validation loss (datatest.txt) distributed metrics not found or empty for {name} ({current_num_clients} clients).")

            # Use new metric key for test (server-side on datatest2.txt)
            if history.metrics_centralized and 'accuracy_test' in history.metrics_centralized and history.metrics_centralized['accuracy_test']:
                current_run_fl_results[name]['Accuracy_Test'] = history.metrics_centralized['accuracy_test'][-1][1]
            else:
                current_run_fl_results[name]['Accuracy_Test'] = 0.0
                logger.warning(f"Test accuracy (datatest2.txt) centralized metrics not found or empty for {name} ({current_num_clients} clients).")
            
            # Test loss is available in history.losses_centralized
            if history.losses_centralized and history.losses_centralized:
                 current_run_fl_results[name]['Loss_Test'] = history.losses_centralized[-1][1] # Store last test loss
            else:
                current_run_fl_results[name]['Loss_Test'] = float('inf')
                logger.warning(f"Test loss (datatest2.txt) centralized metrics not found or empty for {name} ({current_num_clients} clients).")

            current_run_fl_results[name]['Training Time'] = fl_training_time # Store training time
            logger.info(f"--- {name} Simulation with {current_num_clients} clients Finished ---")

            # --- Confusion Matrices and Other Metrics for FL Model (with updated parameter retrieval) ---
            final_global_params_for_cm = get_last_params_fn()
            if final_global_params_for_cm:
                final_global_params_ndarrays = parameters_to_ndarrays(final_global_params_for_cm)
                final_global_model = OccupancyNN(input_size).to(device)
                f_params_dict = zip(final_global_model.state_dict().keys(), final_global_params_ndarrays)
                f_state_dict = {k: torch.tensor(v) for k, v in f_params_dict}
                final_global_model.load_state_dict(f_state_dict)
                final_global_model.eval()

                # CM on Validation Set (datatest.txt) - REMOVED
                # y_pred_fl_test1_list = []
                # with torch.no_grad():
                #     test_loader1 = DataLoader(TensorDataset(torch.FloatTensor(X_test1_scaled), torch.FloatTensor(y_test1).reshape(-1,1)), batch_size=32)
                #     for batch_x, _ in test_loader1:
                #         outputs = final_global_model(batch_x.to(device))
                #         predicted_labels = (outputs > 0.5).float().cpu().numpy()
                #         y_pred_fl_test1_list.extend(predicted_labels.flatten())
                # 
                # y_pred_fl_test1_array = np.array(y_pred_fl_test1_list)
                # 
                # cm_fl_test1 = confusion_matrix(y_test1, y_pred_fl_test1_array)
                # plot_confusion_matrix(cm_fl_test1, classes=class_names_fl,
                #                       title=f'{name} ({current_num_clients} clients) - CM (Validation - datatest.txt)',
                #                       filename=os.path.join('results', f'cm_{name}_{current_num_clients}clients_validation.png'))
                logger.info(f"{name} ({current_num_clients} clients) - Validation Set (datatest.txt) Final Accuracy (from history): "
                            f"{current_run_fl_results[name].get('Accuracy_Validation', 0.0):.4f}")
                # Log validation loss as well
                logger.info(f"{name} ({current_num_clients} clients) - Validation Set (datatest.txt) Final Loss (from history): "
                            f"{current_run_fl_results[name].get('Loss_Validation', float('inf')):.4f}")

                # CM on Test Set (datatest2.txt) - RETAINED
                y_pred_fl_test2_list = []
                with torch.no_grad():
                    test_loader2 = DataLoader(TensorDataset(torch.FloatTensor(X_test2_scaled), torch.FloatTensor(y_test2).reshape(-1,1)), batch_size=32)
                    for batch_x, _ in test_loader2:
                        outputs = final_global_model(batch_x.to(device))
                        predicted_labels = (outputs > 0.5).float().cpu().numpy()
                        y_pred_fl_test2_list.extend(predicted_labels.flatten())
                
                y_pred_fl_test2_array = np.array(y_pred_fl_test2_list)

                cm_fl_test2 = confusion_matrix(y_test2, y_pred_fl_test2_array)
                plot_confusion_matrix(cm_fl_test2, classes=class_names_fl,
                                      title=f'{name} ({current_num_clients} clients) - CM (Test Set - datatest2.txt)', # Back to English
                                      filename=os.path.join('results', f'cm_{name}_{current_num_clients}clients_test_set.png')) # Standardized filename
                logger.info(f"{name} ({current_num_clients} clients) - Test Set (datatest2.txt) Final Accuracy (from history): "
                            f"{current_run_fl_results[name].get('Accuracy_Test', 0.0):.4f}")
            else:
                logger.warning(f"FL CM & Metrics: Could not capture final parameters - {name} ({current_num_clients} clients).")

            # --- FL Learning Curves (Separated for Validation and Test) ---
            
            # Plot 1: Validation Metrics (Accuracy & Loss on datatest.txt)
            plt.figure(figsize=(10, 6))
            plot_title_val_lc = f'Validation Metrics - {name} ({current_num_clients} clients) - datatest.txt' # English
            plotted_val_something = False

            if history.metrics_distributed and 'accuracy_validation' in history.metrics_distributed and history.metrics_distributed['accuracy_validation']:
                rounds_val_acc, acc_val_lc = zip(*history.metrics_distributed['accuracy_validation'])
                if rounds_val_acc and acc_val_lc:
                    plt.plot(rounds_val_acc, acc_val_lc, marker='o', linestyle='-', label='Validation Accuracy (Client Avg.)') # English
                    plotted_val_something = True
            
            if history.metrics_distributed and 'loss_validation' in history.metrics_distributed and history.metrics_distributed['loss_validation']:
                rounds_val_loss, loss_val_lc = zip(*history.metrics_distributed['loss_validation'])
                if rounds_val_loss and loss_val_lc:
                    plt.plot(rounds_val_loss, loss_val_lc, marker='s', linestyle='--', label='Validation Loss (Client Avg.)') # English
                    plotted_val_something = True

            if plotted_val_something:
                plt.title(plot_title_val_lc)
                plt.xlabel('Federated Round') # English
                plt.ylabel('Accuracy / Loss') # English
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join('results', f'lc_validation_{name}_{current_num_clients}clients.png'))
            plt.close() # Close validation plot figure

            # Plot 2: Test Metrics (Accuracy & Loss on datatest2.txt)
            plt.figure(figsize=(10, 6))
            plot_title_test_lc = f'Test Metrics - {name} ({current_num_clients} clients) - datatest2.txt' # English
            plotted_test_something = False

            if history.metrics_centralized and 'accuracy_test' in history.metrics_centralized:
                acc_test_data = history.metrics_centralized.get('accuracy_test')
                if acc_test_data:
                    rounds_test_acc, acc_test_server = zip(*acc_test_data)
                    if rounds_test_acc and acc_test_server:
                        plt.plot(rounds_test_acc, acc_test_server, marker='x', linestyle='-', label='Test Accuracy (Server)') # English
                        plotted_test_something = True

            if history.losses_centralized: # Server-side losses are test losses on datatest2.txt
                rounds_test_loss_server, loss_test_server = zip(*history.losses_centralized)
                if rounds_test_loss_server and loss_test_server:
                    plt.plot(rounds_test_loss_server, loss_test_server, marker='p', linestyle='--', label='Test Loss (Server)') # English
                    plotted_test_something = True
            
            if plotted_test_something:
                plt.title(plot_title_test_lc)
                plt.xlabel('Federated Round') # English
                plt.ylabel('Accuracy / Loss') # English
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join('results', f'lc_test_{name}_{current_num_clients}clients.png'))
            plt.close() # Close test plot figure

        all_fl_results_by_client_count[n_c] = current_run_fl_results

    # --- 4. Print Final Results Table ---
    logger.info("\n\n--- COMPARATIVE RESULTS OF ALL MODELS (TABLE) ---")
    
    table_data_for_print = []
    table_headers = [
        "Model Name", 
        "Acc (Validation)", # Updated header
        "Acc (Test)",       # Updated header
        "Training Time (s)"
    ]
    
    # Classical model results (validation on test1, test on test2)
    for name_classical, metrics_classical in classical_results.items():
        row = [name_classical]
        row.extend([
            f"{metrics_classical.get('Accuracy_Validation', 0.0):.4f}", # Test1 is now Validation
            f"{metrics_classical.get('Accuracy_Test', 0.0):.4f}",       # Test2 is Test
            f"{metrics_classical.get('Training Time', 0.0):.2f}"
        ])
        table_data_for_print.append(row)

    # FL model results
    for n_c_fl, fl_results_for_count_fl in all_fl_results_by_client_count.items():
        for strategy_name_fl, metrics_fl in fl_results_for_count_fl.items():
            model_display_name_fl = f"{strategy_name_fl} ({n_c_fl} clients)"
            row = [model_display_name_fl]
            row.extend([
                f"{metrics_fl.get('Accuracy_Validation', 0.0):.4f}", # Use new key
                f"{metrics_fl.get('Accuracy_Test', 0.0):.4f}",       # Use new key
                f"{metrics_fl.get('Training Time', 0.0):.2f}"
            ])
            table_data_for_print.append(row)

    # Calculate column widths
    col_widths = [len(header) for header in table_headers]
    for row_data in table_data_for_print:
        for i, cell in enumerate(row_data):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    
    # Create separator line (more table-like)
    # '-' character for each column width, with '+-+' as separator
    separator_parts = ["-" * width for width in col_widths]
    separator = "-+-".join(separator_parts)
    # Prevent error if col_widths is empty or None
    if not col_widths:
        separator = ""

    # Print header row
    header_line_parts = [f"{table_headers[i]:<{col_widths[i]}}" for i in range(len(table_headers))]
    logger.info(" | ".join(header_line_parts))
    if separator: # Only print separator if it's meaningful
      logger.info(separator)


    # Print data rows
    for row_data in table_data_for_print:
        row_line_parts = [f"{str(row_data[i]):<{col_widths[i]}}" for i in range(len(row_data))]
        logger.info(" | ".join(row_line_parts))
    
    if separator: # Add bottom line only if separator is meaningful
        logger.info("-" * (sum(col_widths) + (len(col_widths) -1) * 3 if col_widths else 0) ) # Adjust length for " | "


    # --- Overall Accuracy Comparison Graph ---
    model_names_plot = []
    accuracies_validation_plot = [] # Renamed for clarity
    accuracies_test_plot = []       # Renamed for clarity

    for model_name_plot, metrics_plot in classical_results.items():
        model_names_plot.append(model_name_plot)
        accuracies_validation_plot.append(metrics_plot.get('Accuracy_Validation', 0.0)) # Test1 is Validation
        accuracies_test_plot.append(metrics_plot.get('Accuracy_Test', 0.0))   # Test2 is Test

    for n_c_plot, fl_results_for_count_plot in all_fl_results_by_client_count.items():
        for strategy_name_plot, metrics_plot_fl in fl_results_for_count_plot.items(): # Renamed
            model_names_plot.append(f'{strategy_name_plot} ({n_c_plot} clients)')
            accuracies_validation_plot.append(metrics_plot_fl.get('Accuracy_Validation', 0.0)) # Use new key
            accuracies_test_plot.append(metrics_plot_fl.get('Accuracy_Test', 0.0))       # Use new key

    x_indices = np.arange(len(model_names_plot))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(max(12, len(model_names_plot) * 0.8), 7))
    rects1 = ax.bar(x_indices - bar_width/2, accuracies_validation_plot, bar_width, label='Validation Accuracy (datatest.txt)')
    rects2 = ax.bar(x_indices + bar_width/2, accuracies_test_plot, bar_width, label='Test Accuracy (datatest2.txt)')

    ax.set_ylabel('Accuracy')
    ax.set_title('Overall Model Accuracy Comparison') # English
    ax.set_xticks(x_indices)
    ax.set_xticklabels(model_names_plot, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis='y', linestyle='--')

    def autolabel(rects_list):
        for rects in rects_list:
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{:.4f}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8, rotation=0)

    autolabel([rects1, rects2])
    fig.tight_layout()
    plt.savefig(os.path.join('results', 'overall_accuracy_comparison.png'))
    plt.close()

    logger.info("All graphs and confusion matrices saved to 'results' directory.")

# fit_config, aggregate_client_metrics_test1 and get_evaluate_fn_test2 functions should remain in global scope.
def fit_config(server_round: int):
    return {"epochs": 1, "round": server_round} # Epoch count was reduced to 1, can be changed if needed.

def aggregate_client_metrics_test1(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    if not metrics: return {}
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics if "accuracy" in m and isinstance(m, dict)]
    losses = [num_examples * m["loss"] for num_examples, m in metrics if "loss" in m and isinstance(m, dict)] # Add loss aggregation
    examples = [num_examples for num_examples, m in metrics if "accuracy" in m and "loss" in m and isinstance(m, dict)] # Ensure both are present
    
    aggregated_accuracy = sum(accuracies) / sum(examples) if sum(examples) > 0 else 0.0
    aggregated_loss = sum(losses) / sum(examples) if sum(examples) > 0 else float('inf') # Aggregate loss
    
    logger.info(f"Aggregated client metrics (validation on datatest.txt): Accuracy={aggregated_accuracy:.4f}, Loss={aggregated_loss:.4f}")
    return {"accuracy_validation": aggregated_accuracy, "loss_validation": aggregated_loss} # Rename keys

# Server-Side Evaluation Function and Parameter Capture
def get_evaluate_fn_test2_and_param_capture(model_nn: torch.nn.Module, test_data: torch.Tensor, test_labels: torch.Tensor, device: torch.device, strategy_name_for_log: str, client_count_for_log: int):
    last_params_store = {"params": None, "round": -1}

    def evaluate(server_round: int, parameters: Union[Parameters, List[np.ndarray]], config: Dict[str, Union[int, float, str]]) -> Optional[Tuple[float, Dict[str, float]]]:
        nonlocal last_params_store # To modify variable in enclosing scope
        try:
            ndarrays: Optional[List[np.ndarray]] = None
            current_params_for_storage: Optional[Parameters] = None

            if isinstance(parameters, list):
                ndarrays = parameters # If already list of ndarray, use directly
            elif hasattr(parameters, 'tensors'): 
                ndarrays = parameters_to_ndarrays(parameters)
                current_params_for_storage = parameters # Store only if Parameters type
            else:
                logger.error(f"evaluate_fn_test2 ({strategy_name_for_log}, {client_count_for_log} clients) received unknown parameter type: {type(parameters)} round: {server_round}")
                return float('inf'), {"accuracy_test": 0.0, "server_eval_error": "unknown_param_type"}

            if ndarrays is None or not all(isinstance(arr, np.ndarray) for arr in ndarrays):
                logger.error(f"evaluate_fn_test2 ({strategy_name_for_log}, {client_count_for_log} clients): ndarrays is not a list of np.ndarray or is None. Type: {type(ndarrays)} round: {server_round}")
                return float('inf'), {"accuracy_test": 0.0, "server_eval_error": "invalid_ndarrays"}
            
            # Before loading parameters, check model state dict keys and number of ndarrays
            model_state_dict_keys = list(model_nn.state_dict().keys())
            if len(ndarrays) != len(model_state_dict_keys):
                logger.error(
                    f"evaluate_fn_test2 ({strategy_name_for_log}, {client_count_for_log} clients) - Layer count mismatch: parameters ({len(ndarrays)}) vs model ({len(model_state_dict_keys)}) round: {server_round}"
                )
                return float('inf'), {"accuracy_test": 0.0, "server_eval_error": "layer_mismatch"}

            params_dict = zip(model_state_dict_keys, ndarrays)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            model_nn.load_state_dict(state_dict, strict=True)
            model_nn.to(device)
            model_nn.eval()
            
            # Store last parameters (only if Parameters type)
            if current_params_for_storage and server_round >= last_params_store["round"]:
                last_params_store["params"] = current_params_for_storage
                last_params_store["round"] = server_round

            test_loader = DataLoader(TensorDataset(test_data, test_labels.reshape(-1,1)), batch_size=32)
            loss = 0; correct = 0; total = 0
            criterion = nn.BCELoss()
            batches_processed = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model_nn(images)
                    loss += criterion(outputs, labels).item() * images.size(0)
                    predicted = (outputs > 0.5).float()
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    batches_processed +=1
            
            if batches_processed == 0 or total == 0:
                logger.warning(f"Server Evaluation ({strategy_name_for_log}, {client_count_for_log} clients, Test Set 2) - Round {server_round}: No batches processed or total samples zero.")
                return float('inf'), {"accuracy_test": 0.0, "server_eval_error": "no_samples_evaluated_server"}

            avg_loss = loss / total
            accuracy = correct / total
            logger.info(f"Server Evaluation ({strategy_name_for_log}, {client_count_for_log} clients, Test Set 2) - Round {server_round}: Accuracy={accuracy:.4f}, Loss={avg_loss:.4f}")
            return avg_loss, {"accuracy_test": accuracy} # Rename key
        except Exception as e:
            logger.error(f"SERVER EVALUATE FN ERROR ({strategy_name_for_log}, {client_count_for_log} clients) - Round {server_round}: {e}", exc_info=True)
            return float('inf'), {"accuracy_test": 0.0, "server_eval_error": str(e)} # Rename key
    return evaluate, lambda: last_params_store["params"]

if __name__ == "__main__":
    main() 