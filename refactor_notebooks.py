import nbformat as nbf
import os

notebooks = ['3_TRAINING_LSTM.ipynb', '3_TRAINING_GRU.ipynb', '3_TRAINING_TCN_BILSTM.ipynb', '3_TRAINING_XLSTM.ipynb']
base_dir = r"d:\Proyek Dosen\Riset Bearing\Notebook-Github\Cross-Domain Generalization RUL Bearing with XAI\Training & Val (XJTU)"

common_imports = """import os
import glob
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
"""

common_config = """DATASET_DIR = r"D:\Proyek Dosen\Riset Bearing\XJTU-SY_Bearing_Datasets\Processed_Data\LSTM_Inputs"
RESULTS_DIR = r"D:\Proyek Dosen\Riset Bearing\Training_Results"
os.makedirs(RESULTS_DIR, exist_ok=True)

WINDOW_SIZES = [10, 20, 30]
FOLDS = 5
NUM_FEATURES = 14

# Anti-Overfitting Hyperparameters
EPOCHS = 60
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-3
HIDDEN_DIM = 16
NUM_LAYERS = 1
NOISE_STD = 0.05
DROPOUT = 0.2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Computation Device: {DEVICE}")
"""

utils_code = """def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def scale_health_index_to_rul(hi, max_rul):
    return np.clip(hi, 0, 1) * max_rul
"""

for nb_name in notebooks:
    model_name = nb_name.split('_')[2].split('.')[0]
    
    nb = nbf.v4.new_notebook()
    
    nb.cells.append(nbf.v4.new_markdown_cell(f"# {model_name} Training with 5-Fold CV and Extreme Anti-Overfitting"))
    nb.cells.append(nbf.v4.new_code_cell(common_imports))
    nb.cells.append(nbf.v4.new_code_cell(common_config))
    nb.cells.append(nbf.v4.new_code_cell(utils_code))
    
    # Model definition
    if model_name == "GRU":
        model_code = """class RobustModel(nn.Module):
    def __init__(self, input_size=14, hidden_dim=16, num_layers=1, output_size=1, dropout=0.2, noise_std=0.05):
        super(RobustModel, self).__init__()
        self.noise_std = noise_std
        self.rnn = nn.GRU(input_size, hidden_dim, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        if self.training and self.noise_std > 0.0:
            x = x + torch.randn_like(x) * self.noise_std
        out, _ = self.rnn(x)
        out = self.dropout(out[:, -1, :])
        return self.linear(out) # NO RELU
"""
    elif model_name == "LSTM":
        model_code = """class RobustModel(nn.Module):
    def __init__(self, input_size=14, hidden_dim=16, num_layers=1, output_size=1, dropout=0.2, noise_std=0.05):
        super(RobustModel, self).__init__()
        self.noise_std = noise_std
        self.rnn = nn.LSTM(input_size, hidden_dim, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        if self.training and self.noise_std > 0.0:
            x = x + torch.randn_like(x) * self.noise_std
        out, _ = self.rnn(x)
        out = self.dropout(out[:, -1, :])
        return self.linear(out) # NO RELU
"""
    else:
        model_code = """class RobustModel(nn.Module):
    def __init__(self, input_size=14, hidden_dim=16, num_layers=1, output_size=1, dropout=0.2, noise_std=0.05):
        super(RobustModel, self).__init__()
        self.noise_std = noise_std
        self.rnn = nn.GRU(input_size, hidden_dim, num_layers, batch_first=True) # Fallback generic
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        if self.training and self.noise_std > 0.0:
            x = x + torch.randn_like(x) * self.noise_std
        out, _ = self.rnn(x)
        out = self.dropout(out[:, -1, :])
        return self.linear(out)
"""
    nb.cells.append(nbf.v4.new_code_cell(model_code))
    
    training_loop = """
summary_metrics = []

for ws in WINDOW_SIZES:
    print(f"\\n{'='*40}\\nEvaluating Window Size: {ws}\\n{'='*40}")
    ws_dir = os.path.join(DATASET_DIR, f"window_size_{ws}")
    
    fold_metrics = []
    
    for fold in range(1, FOLDS + 1):
        print(f"--- Fold {fold} ---")
        train_file = os.path.join(ws_dir, f"processed_train_fold{fold}.csv")
        val_file = os.path.join(ws_dir, f"processed_val_fold{fold}.csv")
        
        if not os.path.exists(train_file) or not os.path.exists(val_file):
            print(f"Missing data for Fold {fold}. Skipping.")
            continue
            
        df_train = pd.read_csv(train_file)
        df_val = pd.read_csv(val_file)
        
        # Scrambling Prevention Pipeline
        # Extract targets
        y_train = df_train['Health_Index'].values
        y_val = df_val['Health_Index'].values
        
        # Drop non-feature columns
        drop_cols = [c for c in ['Health_Index', 'Target_RUL', 'Bearing_ID', 'Change_Point', 'Original_Minute'] if c in df_train.columns]
        x_train_flat = df_train.drop(columns=drop_cols).values
        x_val_flat = df_val.drop(columns=drop_cols).values
        
        # FIX SCRAMBLING BUG: .reshape(N, F, W).transpose(0, 2, 1)
        x_train_3d = x_train_flat.reshape(-1, NUM_FEATURES, ws).transpose(0, 2, 1)
        x_val_3d = x_val_flat.reshape(-1, NUM_FEATURES, ws).transpose(0, 2, 1)
        
        # To Tensors
        tensor_x_train = torch.tensor(x_train_3d, dtype=torch.float32).to(DEVICE)
        tensor_y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(DEVICE)
        tensor_x_val = torch.tensor(x_val_3d, dtype=torch.float32).to(DEVICE)
        tensor_y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(DEVICE)
        
        train_loader = DataLoader(TensorDataset(tensor_x_train, tensor_y_train), batch_size=BATCH_SIZE, shuffle=True)
        
        model = RobustModel(input_size=NUM_FEATURES, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, noise_std=NOISE_STD, dropout=DROPOUT).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        criterion = nn.MSELoss()
        
        # Train
        model.train()
        for epoch in range(EPOCHS):
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                
        # Evaluate
        model.eval()
        with torch.no_grad():
            pred_hi = model(tensor_x_val).cpu().numpy().flatten()
            pred_hi = np.clip(pred_hi, 0, 1) # Clipping outside network
            rmse = calculate_rmse(y_val, pred_hi)
            fold_metrics.append(rmse)
            print(f"Fold {fold} RMSE (HI): {rmse:.4f}")
            
    if fold_metrics:
        avg_rmse = np.mean(fold_metrics)
        print(f"==> Average RMSE for Window Size {ws}: {avg_rmse:.4f}")
        summary_metrics.append({'Window_Size': ws, 'Avg_RMSE_HI': avg_rmse})

summary_df = pd.DataFrame(summary_metrics)
print("\\nFinal Evaluation Across Window Sizes:")
print(summary_df)
summary_df.to_csv(os.path.join(RESULTS_DIR, f"{model_name}_CV_Results.csv"), index=False)
"""
    nb.cells.append(nbf.v4.new_code_cell(training_loop))
    
    with open(os.path.join(base_dir, nb_name), 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
        
print("Notebooks refactored successfully.")
