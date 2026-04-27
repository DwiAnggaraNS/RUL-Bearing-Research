import torch
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
