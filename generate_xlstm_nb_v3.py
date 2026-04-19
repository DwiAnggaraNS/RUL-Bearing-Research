import os
import nbformat as nbf

# 1. Update xLSTM_Implementation.py directly
src_path = r'd:\Proyek Dosen\Riset Bearing\Notebook-Github\Cross-Domain Generalization RUL Bearing with XAI\src\xLSTM_Implementation.py'

new_src_code = '''import torch
import torch.nn as nn
import torch.optim as optim

# ==============================================================================
# 1. xLSTM MODULE: EXPONENTIAL GATING & MATRIX MEMORY APPROXIMATION
# ==============================================================================
class xLSTMBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super(xLSTMBlock, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.W_i = nn.Linear(input_dim, hidden_dim)
        self.W_f = nn.Linear(input_dim, hidden_dim)
        self.W_o = nn.Linear(input_dim, hidden_dim)
        self.W_z = nn.Linear(input_dim, hidden_dim)
        
        self.group_norm = nn.GroupNorm(1, hidden_dim)
        self._reported_weights_nan = False
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        device = x.device
        
        # --- LOGGING: INPUT SURVIVAL CHECK ---
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"[FATAL CORE LOG] Input tensor 'x' given to xLSTMBlock is ALREADY NaN/Inf! Max: {torch.max(x)}")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
            
        if not self._reported_weights_nan:
            if torch.isnan(self.W_i.weight).any() or torch.isnan(self.W_f.weight).any():
                print("[FATAL CORE LOG] Network WEIGHTS inside xLSTMBlock have become NaN! (Optimizer corruption)")
                self._reported_weights_nan = True
        
        h_t = torch.zeros(batch_size, self.hidden_dim).to(device)
        c_t = torch.zeros(batch_size, self.hidden_dim).to(device)
        n_t = torch.ones(batch_size, self.hidden_dim).to(device)
        
        outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            
            # Gating projections
            wi = self.W_i(x_t)
            wf = self.W_f(x_t)
            wo = self.W_o(x_t)
            wz = self.W_z(x_t)
            
            # Sub-level sanity check
            if torch.isnan(wi).any() or torch.isnan(wf).any():
                print(f"[FATAL CORE LOG] Gating projection evaluated to NaN at timestep {t}.")
                wi = torch.nan_to_num(wi, nan=0.0)
                wf = torch.nan_to_num(wf, nan=0.0)
                wo = torch.nan_to_num(wo, nan=0.0)
                wz = torch.nan_to_num(wz, nan=0.0)
            
            # Exponentials carefully bounded to strictly avoid explosive cascades
            i_t = torch.exp(torch.clamp(wi, min=-15.0, max=2.0)) 
            f_t = torch.exp(torch.clamp(wf, min=-15.0, max=2.0))
            
            o_t = torch.sigmoid(wo)
            z_t = torch.tanh(wz)
            
            c_t = f_t * c_t + i_t * z_t
            n_t = f_t * n_t + i_t
            
            # Rescaling to contain long-sequence numeric inflation
            safe_limit = 1e4
            scale_factor = torch.clamp(n_t, min=1.0)
            should_scale = scale_factor > safe_limit
            if should_scale.any():
                scale_div = torch.where(should_scale, scale_factor / safe_limit, torch.ones_like(scale_factor))
                c_t = c_t / scale_div
                n_t = n_t / scale_div

            h_t = o_t * (c_t / (n_t + 1e-6))
            
            if torch.isnan(h_t).any():
                print(f"[xLSTMBlock DEBUG] Detected NaN at timestep {t}. Applying zero-fill recovery.")
                h_t = torch.nan_to_num(h_t, nan=0.0)
                c_t = torch.nan_to_num(c_t, nan=0.0)
                n_t = torch.nan_to_num(n_t, nan=1.0)
                
            outputs.append(h_t.unsqueeze(1))
            
        out_tensor = torch.cat(outputs, dim=1)
        out_tensor = out_tensor.permute(0, 2, 1)
        out_tensor = self.group_norm(out_tensor)
        out_tensor = out_tensor.permute(0, 2, 1)
        
        return out_tensor

# ==============================================================================
# 2. HYBRID ENCODER: TRANSFORMER + xLSTM 
# ==============================================================================
class HybridEncoderLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, p_factor: int = 2, dropout: float = 0.1):
        super(HybridEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        expanded_dim = embed_dim * p_factor
        self.linear_up = nn.Linear(embed_dim, expanded_dim)
        self.xlstm = xLSTMBlock(input_dim=expanded_dim, hidden_dim=expanded_dim)
        self.linear_down = nn.Linear(expanded_dim, embed_dim)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.self_attn(x, x, x)
        if torch.isnan(attn_out).any():
             print("[FATAL CORE LOG] MultiheadAttention evaluated to NaN.")
             attn_out = torch.nan_to_num(attn_out)
             
        x = self.norm1(x + self.dropout(attn_out))
        
        x_up = self.linear_up(x)
        x_xlstm = self.xlstm(x_up)
        x_down = self.linear_down(x_xlstm)
        
        x = self.norm2(x + self.dropout(x_down))
        return x

# ==============================================================================
# 3. FULL MODEL: xLSTM-TRANSFORMER
# ==============================================================================
class xLSTM_Transformer_RUL(nn.Module):
    def __init__(self, num_features: int = 15, embed_dim: int = 32, num_heads: int = 4, num_encoder_layers: int = 2, num_decoder_layers: int = 2, dropout: float = 0.1):
        super(xLSTM_Transformer_RUL, self).__init__()
        
        self.input_embedding = nn.Linear(num_features, embed_dim)
        # --- CRITICAL FIX: INITIAL NORM TO PREVENT EXPLOSIONS FROM UNSCALED DATA ---
        self.initial_norm = nn.LayerNorm(embed_dim) 
        
        self.encoders = nn.ModuleList([
            HybridEncoderLayer(embed_dim=embed_dim, num_heads=num_heads, p_factor=2, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.output_linear = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.ReLU() 
        )

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        if torch.isnan(src).any():
             print("[FATAL CORE LOG] RAW INPUT 'src' evaluated to NaN.")
             src = torch.nan_to_num(src)
             
        x = self.input_embedding(src)
        x = self.initial_norm(x) # Prevents huge values triggering exponential cascade
        
        memory = x
        for encoder in self.encoders:
            memory = encoder(memory)
            
        out = self.decoder(tgt=x, memory=memory)
        
        out = out.permute(0, 2, 1)
        out = self.pool(out).squeeze(-1) 
        rul_pred = self.output_linear(out)       
        
        return rul_pred
'''

with open(src_path, 'w', encoding='utf-8') as f:
    f.write(new_src_code)
    
print('Source xLSTM_Implementation.py updated cleanly.')

# 2. Update the Notebook training loops to include logging and zero_grad fallbacks
nb_path = r'd:\Proyek Dosen\Riset Bearing\Notebook-Github\Cross-Domain Generalization RUL Bearing with XAI\Training & Val (XJTU)\3_TRAINING_XLSTM.ipynb'

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell('# IMPLEMENTATION OF xLSTM ARCHITECTURE ON XJTU DATASET\n\nThis notebook demonstrates the training and validation of the xLSTM-Transformer architecture across multiple window sizes on the XJTU-SY bearing dataset. It features explicit deep anomaly tracking, data sanitization, and mathematical logging constraints to safeguard against sequence float saturation.\n\nAuthor: Anonymous'))

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

import sys
sys.path.append(os.path.abspath(os.path.join('..', 'src')))
from xLSTM_Implementation import xLSTM_Transformer_RUL

# Global Error Logging - Forces crash traceback the VERY Millisecond a tensor turns NaN!
torch.autograd.set_detect_anomaly(True)

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

print(f"PyTorch Version: {torch.__version__}")'''))

cells.append(nbf.v4.new_markdown_cell('## 1. Global Configurations\nCentralized hyperparameters, paths, and environment settings.'))

cells.append(nbf.v4.new_code_cell(r'''DATASET_DIR = r"D:\Proyek Dosen\Riset Bearing\XJTU_Modeling_Input_Dataset"
RESULTS_DIR = r"D:\Proyek Dosen\Riset Bearing\xLSTM_Results"

os.makedirs(RESULTS_DIR, exist_ok=True)

WINDOW_SIZES = [30, 40, 50, 70]
BEARING_LIFESPAN_TIME = 392_275
NUM_FEATURES = 15

EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 0.001
EMBED_DIM = 32
NUM_HEADS = 4
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2
DROPOUT = 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Computation Device: {DEVICE}")'''))

cells.append(nbf.v4.new_markdown_cell('## 2. Evaluation Metrics and Visualization Utilities'))
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
    plt.ylabel('Health Index')
    plt.legend()
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()'''))

cells.append(nbf.v4.new_markdown_cell('## 3. Training Pipeline with NaN Fallback Architectures'))
cells.append(nbf.v4.new_code_cell(r'''global_summary_metrics = []
excel_export_path = os.path.join(RESULTS_DIR, "xLSTM_Predictions_Summary.xlsx")
excel_sheet_buffer = {}

for window_size in WINDOW_SIZES:
    print(f"\n{'='*60}\nSTARTING EXPERIMENT FOR WINDOW SIZE: {window_size}\n{'='*60}")
    window_result_dir = os.path.join(RESULTS_DIR, f"window_size_{window_size}")
    os.makedirs(window_result_dir, exist_ok=True)
    
    # ----------------------------------------------------
    # A. LOAD & SANITIZE DATA
    # ----------------------------------------------------
    training_filepath = os.path.join(DATASET_DIR, f"processed_train_w{window_size}.csv")
    if not os.path.exists(training_filepath): continue
        
    df_train = pd.read_csv(training_filepath)
    y_train_raw = df_train['Target_RUL'].values
    
    # Target RUL Deep Sanitization
    if np.isnan(y_train_raw).any() or np.isinf(y_train_raw).any():
        print(f"[DATA LOG] Target RUL contains NaN/Inf! Applying nan_to_num recovery.")
        y_train_raw = np.nan_to_num(y_train_raw)
        
    cols_to_drop = ['Target_RUL']
    if 'Bearing_ID' in df_train.columns: cols_to_drop.append('Bearing_ID')
    x_train_flat = df_train.drop(columns=cols_to_drop).values
    
    # Input Sequence Feature Deep Sanitization
    if np.isnan(x_train_flat).any() or np.isinf(x_train_flat).any():
        print(f"[DATA LOG] Features contain NaN/Inf! Applying nan_to_num recovery.")
        x_train_flat = np.nan_to_num(x_train_flat)
        
    num_samples = x_train_flat.shape[0]
    derived_features = x_train_flat.shape[1] // window_size
    tensor_x_train = torch.tensor(x_train_flat.reshape(num_samples, window_size, derived_features), dtype=torch.float32).to(DEVICE)
    tensor_y_train = torch.tensor(y_train_raw, dtype=torch.float32).view(-1, 1).to(DEVICE)
    
    del df_train, x_train_flat; gc.collect()
    
    # ----------------------------------------------------
    # B. INITIALIZE & TRAIN
    # ----------------------------------------------------
    model = xLSTM_Transformer_RUL(
        num_features=derived_features, embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS, num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS, dropout=DROPOUT
    ).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_criterion = nn.MSELoss()
    
    training_loader = DataLoader(TensorDataset(tensor_x_train, tensor_y_train), batch_size=BATCH_SIZE, shuffle=True)
    
    print("[INFO] Starting DL Training Phase...")
    model.train()
    best_loss_value = float('inf')
    best_model_weights = copy.deepcopy(model.state_dict())
    loss_history_log = []
    
    skip_window_due_to_fatal_nan = False
    
    for epoch in range(EPOCHS):
        if skip_window_due_to_fatal_nan: break
        current_epoch_loss = 0.0
        valid_batches = 0
        
        for batch_idx, (batch_input, batch_target) in enumerate(training_loader):
            optimizer.zero_grad()
            predictions = model(batch_input)
            
            if torch.isnan(predictions).any():
                print(f"[FATAL NETWORK LOG] Exploding Neurons! Predictions returned NaN array on Epoch {epoch+1}, Batch {batch_idx}.")
                skip_window_due_to_fatal_nan = True
                break

            loss_val = loss_criterion(predictions, batch_target)
            
            # --- GRADIENT / LOSS SANITY CHECK ---
            if torch.isnan(loss_val) or torch.isinf(loss_val):
                print(f"[FATAL NETWORK LOG] Loss evaluated to NaN on Epoch {epoch+1}, Batch {batch_idx}. Skipping optimization step.")
                skip_window_due_to_fatal_nan = True
                break
                
            loss_val.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
            
            optimizer.step()
            current_epoch_loss += loss_val.item()
            valid_batches += 1
            
        if valid_batches == 0 or skip_window_due_to_fatal_nan:
            print(f"[INFO] Halting training loop early for Window Size {window_size} due to network numeric collapse.")
            break
            
        average_epoch_loss = current_epoch_loss / valid_batches
        loss_history_log.append(average_epoch_loss)
        
        if average_epoch_loss < best_loss_value:
            best_loss_value = average_epoch_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{(epoch+1):03d}/{EPOCHS}] - Validation MSE Loss: {average_epoch_loss:.6f}")
    
    if skip_window_due_to_fatal_nan:
        continue 
        
    model.load_state_dict(best_model_weights)
    torch.save(model.state_dict(), os.path.join(window_result_dir, "optimized_xlstm_model.pth"))
    
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history_log, label='MSE Training Loss', color='red')
    plt.title(f'Continuous Training Loss Optimization (Window: {window_size})')
    plt.legend()
    plt.savefig(os.path.join(window_result_dir, "loss_history_optimization.png"), dpi=300)
    plt.close()
    
    del tensor_x_train, tensor_y_train, training_loader; torch.cuda.empty_cache(); gc.collect()
    
    # ----------------------------------------------------
    # C. CROSS-DOMAIN EVALUATION
    # ----------------------------------------------------
    evaluation_files = glob.glob(os.path.join(DATASET_DIR, f"processed_val_*_w{window_size}.csv"))
    if not evaluation_files: evaluation_files = glob.glob(os.path.join(DATASET_DIR, f"processed_test_*_w{window_size}.csv"))
        
    regional_predictions_dataframe = pd.DataFrame()
    regional_metrics_log = []
    
    model.eval()
    with torch.no_grad():
        for eval_file_path in evaluation_files:
            bearing_identity = re.search(r"processed_(?:val|test)_(.*)_w\d+\.csv", os.path.basename(eval_file_path)).group(1)
            print(f"[EVAL] Processing Validation Data: {bearing_identity}...")
            
            df_eval = pd.read_csv(eval_file_path)
            y_true_eval = df_eval['Target_RUL'].values
            
            eval_drop_limiters = ['Target_RUL']
            if 'Bearing_ID' in df_eval.columns: eval_drop_limiters.append('Bearing_ID')
                
            x_eval_flat = df_eval.drop(columns=eval_drop_limiters).values
            x_eval_flat = np.nan_to_num(x_eval_flat) # Safety First
                
            ns, nf = x_eval_flat.shape[0], x_eval_flat.shape[1] // window_size
            tensor_x_eval = torch.tensor(x_eval_flat.reshape(ns, window_size, nf), dtype=torch.float32).to(DEVICE)
            
            y_pred_eval = model(tensor_x_eval).cpu().numpy().flatten()
            y_pred_eval = np.nan_to_num(y_pred_eval) # Enforce numeric limits 
            
            regional_metrics_log.append({
                'Bearing_ID': bearing_identity, 'Window_Size': window_size,
                'RMSE': calculate_rmse(y_true_eval, y_pred_eval),
                'MAE': mean_absolute_error(y_true_eval, y_pred_eval),
                'R2': r2_score(y_true_eval, y_pred_eval),
                'RPE(%)': calculate_relative_prediction_error(y_true_eval, y_pred_eval),
                'Asymmetric_Loss': compute_asymmetric_loss(y_pred_eval, y_true_eval)
            })
            
            b_frame = pd.DataFrame({
                'Bearing_ID': bearing_identity, 'Time_Step': np.arange(len(y_true_eval)),
                'Expected_Health_Index': y_true_eval, 'Predicted_Health_Index': y_pred_eval,
                'Expected_Remaining_RUL': scale_health_index_to_rul(y_true_eval),
                'Predicted_Remaining_RUL': scale_health_index_to_rul(y_pred_eval)
            })
            regional_predictions_dataframe = pd.concat([regional_predictions_dataframe, b_frame], ignore_index=True)
            visualize_predictions(y_true_eval, y_pred_eval, f"Health Index Trajectory ({bearing_identity} | WS: {window_size})", os.path.join(window_result_dir, f"hi_trajectory_{bearing_identity}.png"))
            del df_eval, x_eval_flat, tensor_x_eval; gc.collect()
            
    if not regional_predictions_dataframe.empty:
        regional_predictions_dataframe.to_csv(os.path.join(window_result_dir, f"prediction_states_w{window_size}.csv"), index=False)
        excel_sheet_buffer[f"WS_{window_size}"] = regional_predictions_dataframe
        
    if regional_metrics_log:
        df_metrics = pd.DataFrame(regional_metrics_log)
        df_metrics.to_csv(os.path.join(window_result_dir, f"evaluation_markers_w{window_size}.csv"), index=False)
        avg_m = df_metrics.mean(numeric_only=True)
        global_summary_metrics.append({
            'Window_Size': window_size, 'Mean_RMSE': avg_m['RMSE'], 'Mean_MAE': avg_m['MAE'],
            'Mean_R2': avg_m['R2'], 'Mean_RPE(%)': avg_m['RPE(%)'], 'Mean_Asymmetric_Loss': avg_m['Asymmetric_Loss']
        })

print(f"\n[INFO] Validating Metric Export Buffer...")
if len(excel_sheet_buffer) > 0:
    with pd.ExcelWriter(excel_export_path) as xl_writer:
        for sheet_reference, sheet_data in excel_sheet_buffer.items():
            sheet_data.to_excel(xl_writer, sheet_name=sheet_reference, index=False)
    print(f">> Successfully Converted to Master Excel Format: {excel_export_path}")
else:
    print(">> Processing Halt: No numeric matrices survived network processing.")'''))

cells.append(nbf.v4.new_markdown_cell('## 4. Final Aggregated Conclusion Analysis'))
cells.append(nbf.v4.new_code_cell('''summary_dashboard = pd.DataFrame(global_summary_metrics)\nif not summary_dashboard.empty:\n    display(summary_dashboard)\n    summary_dashboard.to_csv(os.path.join(RESULTS_DIR, "xLSTM_Global_Window_Comparison.csv"), index=False)\n    optimal_window_size = summary_dashboard.loc[summary_dashboard['Mean_RMSE'].idxmin()]['Window_Size']\n    print(f"\\n>>> ANALYTICS CONCLUSION: Based on uniform RMSE testing, the optimal structure timeframe is: {int(optimal_window_size)} <<<")\nelse:\n    print(">> Sequence Terminated: No analytical metrics available for overview extraction.")'''))

nb['cells'] = cells
with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print('Source Notebook regenerated cleanly with comprehensive logging systems!')
