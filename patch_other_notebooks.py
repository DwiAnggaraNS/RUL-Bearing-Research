import nbformat as nbf
import os

notebooks = ['3_TRAINING_GRU.ipynb', '3_TRAINING_TCN_BILSTM.ipynb', '3_TRAINING_XLSTM.ipynb']
base_dir = r"d:\Proyek Dosen\Riset Bearing\Notebook-Github\Cross-Domain Generalization RUL Bearing with XAI\Training & Val (XJTU)"

for nb_name in notebooks:
    file_path = os.path.join(base_dir, nb_name)
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            nb = nbf.read(f, as_version=4)
            
        for cell in nb.cells:
            if cell.cell_type == 'code' and "x_train_3d = x_train_flat.reshape" in cell.source:
                # Replace the hardcoded NUM_FEATURES to dynamic ones in the other notebooks too
                cell.source = cell.source.replace("x_train_3d = x_train_flat.reshape(-1, NUM_FEATURES, ws).transpose(0, 2, 1)", 
                "actual_num_features = x_train_flat.shape[1] // ws\n        x_train_3d = x_train_flat.reshape(-1, actual_num_features, ws).transpose(0, 2, 1)")
                
                cell.source = cell.source.replace("x_val_3d = x_val_flat.reshape(-1, NUM_FEATURES, ws).transpose(0, 2, 1)", 
                "x_val_3d = x_val_flat.reshape(-1, actual_num_features, ws).transpose(0, 2, 1)")
                
                cell.source = cell.source.replace("input_size=NUM_FEATURES", "input_size=actual_num_features")
                
        with open(file_path, 'w', encoding='utf-8') as f:
            nbf.write(nb, f)
            
print("Other notebooks patched dynamically.")
