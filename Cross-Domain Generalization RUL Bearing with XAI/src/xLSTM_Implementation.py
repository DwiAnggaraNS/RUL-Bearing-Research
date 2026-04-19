import torch
import torch.nn as nn
import torch.optim as optim
import math
from typing import Tuple

# ==============================================================================
# 1. xLSTM MODULE: EXPONENTIAL GATING & MATRIX MEMORY APPROXIMATION
# Based on Section 2.2 and Eq. (1)-(10) of the paper
# 
# ==============================================================================
class xLSTMBlock(nn.Module):
    """
    Simplified implementation of the xLSTM module (incorporating sLSTM and mLSTM traits).
    Uses exponential gating to prevent saturation during late degradation phases, 
    as described in the xLSTM-Transformer paper.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super(xLSTMBlock, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Linear projections for gates (similar to sLSTM/mLSTM architecture)
        # Using exponential activation for the input and forget gates
        self.W_i = nn.Linear(input_dim, hidden_dim)
        self.W_f = nn.Linear(input_dim, hidden_dim)
        self.W_o = nn.Linear(input_dim, hidden_dim)
        self.W_z = nn.Linear(input_dim, hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the xLSTM block over a sequence.
        Args:
            x: Input tensor of shape (Batch_Size, Sequence_Length, Input_Dim)
        Returns:
            Output tensor of shape (Batch_Size, Sequence_Length, Hidden_Dim)
        """
        batch_size, seq_len, _ = x.size()
        device = x.device
        
        # Initialize hidden state and cell state (matrix/scalar approximations)
        h_t = torch.zeros(batch_size, self.hidden_dim).to(device)
        c_t = torch.zeros(batch_size, self.hidden_dim).to(device)
        n_t = torch.ones(batch_size, self.hidden_dim).to(device) # Normalizer state
        
        outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            
            # Exponential Gating (Eq. 1 and Eq. 2) with strict upper-bounding to prevent float32 overflow (NaNs)
            # Max clamped to 0.0 so exp(x) stays within stable range <= 1.0, preserving long sequences.
            i_t = torch.exp(torch.clamp(self.W_i(x_t), min=-20.0, max=0.0)) 
            f_t = torch.exp(torch.clamp(self.W_f(x_t), min=-20.0, max=0.0))
            
            # Sigmoid for output gate (Eq. 3)
            o_t = torch.sigmoid(self.W_o(x_t))
            
            # Input transformation
            z_t = torch.tanh(self.W_z(x_t))
            
            # Update states (Eq. 4 and Eq. 5)
            c_t = f_t * c_t + i_t * z_t
            n_t = f_t * n_t + i_t
            
            # Output computation with normalizer (Eq. 6)
            h_t = o_t * (c_t / (n_t + 1e-6))
            outputs.append(h_t.unsqueeze(1))
            
        return torch.cat(outputs, dim=1)

# ==============================================================================
# 2. HYBRID ENCODER: TRANSFORMER + xLSTM 
# Based on Section 2.4 and Eq. (15)-(18)
# ==============================================================================
class HybridEncoderLayer(nn.Module):
    """
    Custom Transformer Encoder layer with an embedded xLSTM module.
    Replaces standard Feed-Forward block partly with xLSTM for dynamic temporal modeling.
    """
    def __init__(self, embed_dim: int, num_heads: int, p_factor: int = 2, dropout: float = 0.1):
        super(HybridEncoderLayer, self).__init__()
        
        # 1. Multi-Head Attention
        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # 2. xLSTM Projection Block (Eq. 16 - 18)
        expanded_dim = embed_dim * p_factor
        self.linear_up = nn.Linear(embed_dim, expanded_dim)
        self.xlstm = xLSTMBlock(input_dim=expanded_dim, hidden_dim=expanded_dim)
        self.linear_down = nn.Linear(expanded_dim, embed_dim)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Multi-Head Attention (Eq. 15)
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # xLSTM Module Integration (Eq. 16 - 18)
        # Up-projection
        x_up = self.linear_up(x)
        # xLSTM nonlinear processing
        x_xlstm = self.xlstm(x_up)
        # Down-projection
        x_down = self.linear_down(x_xlstm)
        
        # Residual and Normalization
        x = self.norm2(x + self.dropout(x_down))
        return x

# ==============================================================================
# 3. FULL MODEL: xLSTM-TRANSFORMER
# ==============================================================================
class xLSTM_Transformer_RUL(nn.Module):
    """
    Complete xLSTM-Transformer Neural Network for Bearing RUL Prediction.
    Follows the Encoder-Decoder framework proposed by Jiang et al. (2026).
    """
    def __init__(self, 
                 num_features: int = 15, 
                 embed_dim: int = 32, 
                 num_heads: int = 4, 
                 num_encoder_layers: int = 2,
                 num_decoder_layers: int = 2,
                 dropout: float = 0.1):
        """
        Initializes the model architecture.
        
        Args:
            num_features (int): Number of extracted physical features (default 15).
            embed_dim (int): Hidden dimension size (paper specifies 16-32).
            num_heads (int): Number of attention heads.
            num_encoder_layers (int): Number of Hybrid Encoder blocks.
            num_decoder_layers (int): Number of Standard Decoder blocks.
        """
        super(xLSTM_Transformer_RUL, self).__init__()
        
        # Input Embedding: Projects the 15 features into the model's hidden dimension
        self.input_embedding = nn.Linear(num_features, embed_dim)
        
        # Hybrid Encoders (Transformer + xLSTM)
        self.encoders = nn.ModuleList([
            HybridEncoderLayer(embed_dim=embed_dim, num_heads=num_heads, p_factor=2, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # Standard Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # Output Regression Block (Flatten + Linear)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.output_linear = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.ReLU() # RUL is strictly non-negative
        )

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            src: Input tensor of shape (Batch_Size, Window_Size, 15)
        Returns:
            rul_pred: Predicted RUL value of shape (Batch_Size, 1)
        """
        # 1. Input Embedding
        x = self.input_embedding(src)
        
        # 2. Pass through Hybrid Encoders
        memory = x
        for encoder in self.encoders:
            memory = encoder(memory)
            
        # 3. Pass through Decoder 
        # (Using the encoded memory as both target and memory for self-reconstruction)
        out = self.decoder(tgt=x, memory=memory)
        
        # 4. Regress to RUL Value
        # Permute for AdaptiveAvgPool: (Batch, Seq_Len, Embed_Dim) -> (Batch, Embed_Dim, Seq_Len)
        out = out.permute(0, 2, 1)
        out = self.pool(out).squeeze(-1) # Output shape: (Batch, Embed_Dim)
        rul_pred = self.output_linear(out)       # Output shape: (Batch, 1)
        
        return rul_pred

# ==============================================================================
# 4. TRAINING FUNCTION IMPLEMENTATION
# ==============================================================================
def train_xlstm_transformer(model: nn.Module, 
                            train_loader, 
                            epochs: int = 50, 
                            lr: float = 0.001):
    """
    Standard training loop for the xLSTM-Transformer model.
    Matches the paper's hyperparameter settings (Adam optimizer, MSE Loss).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Optimizer and Loss function as defined in Table 3 of the paper
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    print(f"Starting Training on device: {device}...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(batch_x)
            
            # Loss calculation
            loss = criterion(predictions, batch_y)
            
            # Backward pass & Optimization
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] | MSE Loss: {avg_loss:.4f}")

# Example Configuration (Table 3 parameters)
# WINDOW_SIZE = 30 (or 50, dynamic)
# NUM_FEATURES = 15
# BATCH_SIZE = 32
# HIDDEN_DIMENSIONS = 32 (embed_dim)