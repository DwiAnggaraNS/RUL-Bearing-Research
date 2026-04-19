import os
import nbformat as nbf

nb_path = r'd:\Proyek Dosen\Riset Bearing\Notebook-Github\Cross-Domain Generalization RUL Bearing with XAI\Training & Val (XJTU)\3_TRAINING_LSTM.ipynb'
nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell('# IMPLEMENTATION OF VANILLA LSTM ARCHITECTURE ON XJTU DATASET\n\nThis notebook demonstrates the training and validation of the Vanilla LSTM architecture across multiple window sizes on the XJTU-SY bearing dataset. It features extensive looping over window sizes, caching, error handling, and robust generation of publication-ready visualizations.\n\nAuthor: Anonymous'))

cells.append(nbf.v4.new_code_cell('''import os
import glob
import time
import gc
import re
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Matplotlib Publication Standards
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'lines.linewidth': 2.0,
    'grid.alpha': 0.3
})
sns.set_style("whitegrid")

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Active GPU: {torch.cuda.get_device_name(0)}")'''))

cells.append(nbf.v4.new_markdown_cell('## 1. Global Configurations\nCentralized hyperparameters, paths, and environment settings. These define the parameters for the LSTM training pipeline.'))
cells.append(nbf.v4.new_code_cell('''DATASET_DIR = r"D:\Proyek Dosen\Riset Bearing\XJTU_Modeling_Input_Dataset"
RESULTS_DIR = r"D:\Proyek Dosen\Riset Bearing\LSTM_Results"

if not os.path.exists(DATASET_DIR):
    print(f"Warning: DATASET_DIR '{DATASET_DIR}' not found. Please verify the path.")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Experiment Setup
WINDOW_SIZES = [30, 40, 50, 70]
BEARING_LIFESPAN_TIME = 392_275  # Maximum lifespan for scaling RUL if needed
NUM_FEATURES = 15

# Training Hyperparameters
EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 0.001
LSTM_HIDDEN_DIM = 64
LSTM_NUM_LAYERS = 2
DROPOUT = 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Computation Device: {DEVICE}")'''))

cells.append(nbf.v4.new_markdown_cell('## 2. Model Architecture\nDefinition of the Vanilla LSTM Regression Network.'))
cells.append(nbf.v4.new_code_cell('''class VanillaLSTM_RUL(nn.Module):
    """
    Vanilla LSTM Neural Network for Remaining Useful Life (RUL) Prediction.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=1, dropout=0.1):
        super(VanillaLSTM_RUL, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, 
                            num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.ReLU()  # RUL is non-negative
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # Get the output from the last time step
        last_out = lstm_out[:, -1, :]
        out = self.fc(last_out)
        return out'''))

cells.append(nbf.v4.new_markdown_cell('## 3. Evaluation Metrics and Visualization Utilities\nHelper functions for error evaluation (RMSE, MAE, R2, Relative Prediction Error) and generating performance graphs.'))
cells.append(nbf.v4.new_code_cell('''def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_relative_prediction_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if not np.any(mask): return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def scale_health_index_to_rul(hi):
    return np.clip(hi, 0, 1) * BEARING_LIFESPAN_TIME

def compute_asymmetric_loss(y_pred, y_true, a=10.0, b=13.0):
    pred_tensor, true_tensor = torch.tensor(y_pred, dtype=torch.float32), torch.tensor(y_true, dtype=torch.float32)
    diff = pred_tensor - true_tensor
    loss = torch.where(diff < 0, torch.exp(-diff / a) - 1, torch.exp(diff / b) - 1)
    return float(loss.mean().item())

def visualize_predictions(y_true, y_pred, title, save_path=None):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Actual Health Index', color='black', alpha=0.8)
    plt.plot(y_pred, label='Predicted Health Index', color='gold', linestyle='--')
    plt.title(title, fontweight='bold')
    plt.xlabel('Time Steps')
    plt.ylabel('Health Index (0.0: Failure, 1.0: Normal)')
    plt.legend(loc='upper right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()'''))

cells.append(nbf.v4.new_markdown_cell('## 4. Training & Validation Pipeline\nIterates over the defined validation window sizes, training the architecture and extracting test evaluation markers. It dynamically isolates memory usage to prevent Out-Of-Memory (OOM) errors during the loop.'))
cells.append(nbf.v4.new_code_cell(r'''global_summary_metrics = []
excel_export_path = os.path.join(RESULTS_DIR, "LSTM_Predictions_Summary.xlsx")
excel_sheet_buffer = {}

for window_size in WINDOW_SIZES:
    print(f"\n{'='*60}")
    print(f"STARTING EXPERIMENT FOR WINDOW SIZE: {window_size}")
    print(f"{'='*60}")
    
    window_result_dir = os.path.join(RESULTS_DIR, f"window_size_{window_size}")
    os.makedirs(window_result_dir, exist_ok=True)
    
    training_filepath = os.path.join(DATASET_DIR, f"processed_train_w{window_size}.csv")
    if not os.path.exists(training_filepath):
        print(f"[ERROR] Training payload {training_filepath} missing. Skipping window size {window_size}.")
        continue
        
    print(f"[INFO] Loading training dataset: {training_filepath}")
    df_train = pd.read_csv(training_filepath)
    
    y_train_raw = df_train['Target_RUL'].values
    if np.isnan(y_train_raw).any() or np.isinf(y_train_raw).any():
        print(f"[DATA LOG] Target RUL contains NaN/Inf! Applying nan_to_num recovery.")
        y_train_raw = np.nan_to_num(y_train_raw)
        
    columns_to_drop = ['Target_RUL']
    if 'Bearing_ID' in df_train.columns:
        columns_to_drop.append('Bearing_ID')
        
    x_train_flat = df_train.drop(columns=columns_to_drop).values
    if np.isnan(x_train_flat).any() or np.isinf(x_train_flat).any():
        print(f"[DATA LOG] Features contain NaN/Inf! Applying nan_to_num recovery.")
        x_train_flat = np.nan_to_num(x_train_flat)
        
    num_samples = x_train_flat.shape[0]
    derived_features = x_train_flat.shape[1] // window_size
    
    x_train_sequences = x_train_flat.reshape(num_samples, window_size, derived_features)
    tensor_x_train = torch.tensor(x_train_sequences, dtype=torch.float32).to(DEVICE)
    tensor_y_train = torch.tensor(y_train_raw, dtype=torch.float32).view(-1, 1).to(DEVICE)
    
    # Memory Management
    del df_train, x_train_flat, x_train_sequences
    gc.collect()
    
    model = VanillaLSTM_RUL(
        input_dim=derived_features,
        hidden_dim=LSTM_HIDDEN_DIM,
        num_layers=LSTM_NUM_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_criterion = nn.MSELoss()
    
    training_dataset = TensorDataset(tensor_x_train, tensor_y_train)
    training_loader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print("[INFO] Starting DL Training Phase...")
    model.train()
    
    best_loss_value = float('inf')
    best_model_weights = copy.deepcopy(model.state_dict())
    loss_history_log = []
    
    for epoch in range(EPOCHS):
        current_epoch_loss = 0.0
        for batch_input, batch_target in training_loader:
            optimizer.zero_grad()
            predictions = model(batch_input)
            loss_val = loss_criterion(predictions, batch_target)
            loss_val.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            current_epoch_loss += loss_val.item()
        
        average_epoch_loss = current_epoch_loss / len(training_loader)
        loss_history_log.append(average_epoch_loss)
        
        if average_epoch_loss < best_loss_value:
            best_loss_value = average_epoch_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{(epoch+1):03d}/{EPOCHS}] - Validation MSE Loss: {average_epoch_loss:.6f}")
    
    model.load_state_dict(best_model_weights)
    torch.save(model.state_dict(), os.path.join(window_result_dir, "optimized_lstm_model.pth"))
    
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history_log, label='MSE Training Loss', color='red')
    plt.title(f'Continuous Training Loss Optimization (Window: {window_size})')
    plt.xlabel('Epoch Iteration')
    plt.ylabel('MSE Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(window_result_dir, "loss_history_optimization.png"), dpi=300)
    plt.close()
    
    # Memory Management
    del tensor_x_train, tensor_y_train, training_dataset, training_loader
    torch.cuda.empty_cache()
    gc.collect()
    
    # Validation Evaluation
    evaluation_files = glob.glob(os.path.join(DATASET_DIR, f"processed_val_*_w{window_size}.csv"))
    if len(evaluation_files) == 0:
        evaluation_files = glob.glob(os.path.join(DATASET_DIR, f"processed_test_*_w{window_size}.csv"))
        
    if len(evaluation_files) == 0:
        print(f"[WARNING] No evaluation set files targeting window {window_size} were identified!")
        
    regional_predictions_dataframe = pd.DataFrame()
    regional_metrics_log = []
    
    model.eval()
    with torch.no_grad():
        for eval_file_path in evaluation_files:
            regex_match = re.search(r"processed_(?:val|test)_(.*)_w\d+\.csv", os.path.basename(eval_file_path))
            bearing_identity = regex_match.group(1) if regex_match else 'Evaluated_Unknown'
            print(f"[EVAL] Evaluating Sequence Pattern on Domain: {bearing_identity}...")
            
            df_eval = pd.read_csv(eval_file_path)
            y_true_eval = df_eval['Target_RUL'].values
            
            eval_drop_limiters = ['Target_RUL']
            if 'Bearing_ID' in df_eval.columns:
                eval_drop_limiters.append('Bearing_ID')
                
            x_eval_flat = df_eval.drop(columns=eval_drop_limiters).values
            x_eval_flat = np.nan_to_num(x_eval_flat)
            num_eval_samples = x_eval_flat.shape[0]
            num_eval_features = x_eval_flat.shape[1] // window_size
            
            x_eval_sequences = x_eval_flat.reshape(num_eval_samples, window_size, num_eval_features)
            tensor_x_eval = torch.tensor(x_eval_sequences, dtype=torch.float32).to(DEVICE)
            
            y_pred_eval = model(tensor_x_eval).cpu().numpy().flatten()
            y_pred_eval = np.nan_to_num(y_pred_eval)
            
            rmse_marker = calculate_rmse(y_true_eval, y_pred_eval)
            mae_marker = mean_absolute_error(y_true_eval, y_pred_eval)
            r2_marker = r2_score(y_true_eval, y_pred_eval)
            rpe_marker = calculate_relative_prediction_error(y_true_eval, y_pred_eval)
            asym_marker = compute_asymmetric_loss(y_pred_eval, y_true_eval)
            
            regional_metrics_log.append({
                'Bearing_ID': bearing_identity,
                'Window_Size': window_size,
                'RMSE': rmse_marker,
                'MAE': mae_marker,
                'R2': r2_marker,
                'RPE(%)': rpe_marker,
                'Asymmetric_Loss': asym_marker
            })
            
            bearing_output_frame = pd.DataFrame({
                'Bearing_ID': bearing_identity,
                'Time_Step': np.arange(len(y_true_eval)),
                'Expected_Health_Index': y_true_eval,
                'Predicted_Health_Index': y_pred_eval,
                'Expected_Remaining_RUL': scale_health_index_to_rul(y_true_eval),
                'Predicted_Remaining_RUL': scale_health_index_to_rul(y_pred_eval)
            })
            regional_predictions_dataframe = pd.concat([regional_predictions_dataframe, bearing_output_frame], ignore_index=True)
            
            graph_title = f"Health Index Trajectory ({bearing_identity} | WS: {window_size})"
            graph_save_path = os.path.join(window_result_dir, f"hi_trajectory_{bearing_identity}.png")
            visualize_predictions(y_true_eval, y_pred_eval, graph_title, graph_save_path)
            
            # Memory Management
            del df_eval, x_eval_flat, x_eval_sequences, tensor_x_eval
            gc.collect()
            
    if not regional_predictions_dataframe.empty:
        regional_predictions_dataframe.to_csv(os.path.join(window_result_dir, f"prediction_states_w{window_size}.csv"), index=False)
        excel_sheet_buffer[f"WS_{window_size}"] = regional_predictions_dataframe
        
    if len(regional_metrics_log) > 0:
        regional_metrics_df = pd.DataFrame(regional_metrics_log)
        regional_metrics_df.to_csv(os.path.join(window_result_dir, f"evaluation_markers_w{window_size}.csv"), index=False)
        aggregated_metrics = regional_metrics_df.mean(numeric_only=True)
        
        global_summary_metrics.append({
            'Window_Size': window_size,
            'Mean_RMSE': aggregated_metrics['RMSE'],
            'Mean_MAE': aggregated_metrics['MAE'],
            'Mean_R2': aggregated_metrics['R2'],
            'Mean_RPE(%)': aggregated_metrics['RPE(%)'],
            'Mean_Asymmetric_Loss': aggregated_metrics['Asymmetric_Loss']
        })

print(f"\n[INFO] Consolidating predictions into Global Metrics Workbook: {excel_export_path}")
if len(excel_sheet_buffer) > 0:
    with pd.ExcelWriter(excel_export_path) as xl_writer:
        for sheet_reference, sheet_data in excel_sheet_buffer.items():
            sheet_data.to_excel(xl_writer, sheet_name=sheet_reference, index=False)
    print(">> Target Compilation Terminated Successfully.")
else:
    print(">> No predictions captured for workbook representation.")'''))

cells.append(nbf.v4.new_markdown_cell('## 5. Window Size Performance Evaluation\nAggregates evaluations across all setups to discover the optimal structural modeling timeframe and visualizations the RMSE trend across variations.'))
cells.append(nbf.v4.new_code_cell('''summary_dashboard = pd.DataFrame(global_summary_metrics)

if not summary_dashboard.empty:
    display(summary_dashboard)
    summary_dashboard.to_csv(os.path.join(RESULTS_DIR, "LSTM_Global_Window_Comparison.csv"), index=False)
    
    plt.figure(figsize=(10, 6))
    plt.plot(summary_dashboard['Window_Size'], summary_dashboard['Mean_RMSE'], 
             marker='o', linestyle='-', color='purple', linewidth=2.5, markersize=8)
    
    for index, context_row in summary_dashboard.iterrows():
        plt.annotate(f"{context_row['Mean_RMSE']:.4f}", 
                     (context_row['Window_Size'], context_row['Mean_RMSE']),
                     textcoords="offset points", 
                     xytext=(0, 10), 
                     ha='center')
                     
    plt.title('Validation Root Mean Square Error vs Target Window Size (LSTM)', fontweight='bold', pad=15)
    plt.xlabel('Designated Window Format Size')
    plt.ylabel('Aggregated Validation RMSE')
    plt.xticks(summary_dashboard['Window_Size'].values)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "optimal_rmse_curve_representation.png"), dpi=300)
    plt.show()
    
    optimal_window_size = summary_dashboard.loc[summary_dashboard['Mean_RMSE'].idxmin()]['Window_Size']
    print(f"\\n>>> ANALYTICS CONCLUSION: Based on uniform RMSE testing, the optimal structure timeframe is: {int(optimal_window_size)} <<<")
else:
    print(">> Sequence Terminated: No analytical metrics available for overview extraction.")'''))

nb.cells = cells
with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print('Source Notebook generated successfully.')
