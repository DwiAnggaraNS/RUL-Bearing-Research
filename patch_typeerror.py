import nbformat as nbf
import os

notebooks = ['3_TRAINING_LSTM.ipynb', '3_TRAINING_GRU.ipynb', '3_TRAINING_TCN_BILSTM.ipynb', '3_TRAINING_XLSTM.ipynb']
base_dir = r"d:\Proyek Dosen\Riset Bearing\Notebook-Github\Cross-Domain Generalization RUL Bearing with XAI\Training & Val (XJTU)"

for nb_name in notebooks:
    file_path = os.path.join(base_dir, nb_name)
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            nb = nbf.read(f, as_version=4)
            
        for cell in nb.cells:
            if cell.cell_type == 'code' and "x_train_flat = df_train.drop(columns=drop_cols).values" in cell.source:
                old_code = """        drop_cols = [c for c in ['Health_Index', 'Target_RUL', 'Bearing_ID', 'Change_Point', 'Original_Minute'] if c in df_train.columns]
        x_train_flat = df_train.drop(columns=drop_cols).values
        x_val_flat = df_val.drop(columns=drop_cols).values"""
                
                new_code = """        drop_cols = [c for c in ['Health_Index', 'Target_RUL', 'Bearing_ID', 'Change_Point', 'Original_Minute'] if c in df_train.columns]
        
        # Prevent TypeError (numpy.object_): filter strictly numeric to avoid accidental metadata strings like 'Normalized_Bearing_ID'
        df_train_num = df_train.select_dtypes(include=[np.number])
        df_val_num = df_val.select_dtypes(include=[np.number])
        x_train_flat = df_train_num.drop(columns=[c for c in drop_cols if c in df_train_num.columns]).values.astype(np.float32)
        x_val_flat = df_val_num.drop(columns=[c for c in drop_cols if c in df_val_num.columns]).values.astype(np.float32)"""
                
                cell.source = cell.source.replace(old_code, new_code)
                
        with open(file_path, 'w', encoding='utf-8') as f:
            nbf.write(nb, f)
            
print("Successfully patched TypeError across all 4 notebooks.")
