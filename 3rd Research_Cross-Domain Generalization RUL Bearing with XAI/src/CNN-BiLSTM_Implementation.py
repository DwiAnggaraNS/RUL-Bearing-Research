import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Function
import numpy as np
from typing import Tuple

# ==============================================================================
# 1. GRADIENT REVERSAL LAYER (GRL) FOR ADVERSARIAL DOMAIN ADAPTATION
# ==============================================================================
class GradientReversalLayer(Function):
    """
    Gradient Reversal Layer (GRL).
    Acts as an identity mapping during the forward pass.
    Multiplies the gradient by a negative constant (alpha) during the backward pass.
    This forces the feature extractor to learn domain-invariant features.
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        # Reverse the gradient and scale by alpha
        return grad_output.neg() * ctx.alpha, None

def grad_reverse(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    return GradientReversalLayer.apply(x, alpha)

# ==============================================================================
# 2. MODEL ARCHITECTURE: CNN-BiLSTM-DA
# ==============================================================================
class CNN_BiLSTM_DomainAdaptation(nn.Module):
    """
    CNN-Bi-LSTM-Based Domain Adaptation Model for RUL Prediction.
    Based on the architecture by Li et al. (Sensors 2024).
    """
    def __init__(self, 
                 num_features: int = 15, 
                 bilstm_hidden_size: int = 64):
        """
        Initializes the model architecture.

        Args:
            num_features (int): Number of extracted physical features (default 15).
            bilstm_hidden_size (int): Hidden units for the BiLSTM layer (paper uses 64).
        """
        super(CNN_BiLSTM_DomainAdaptation, self).__init__()

        # --- MODULE 1: CNN Feature Extractor ---
        # Adjusted kernels from 8/4 to 3 to prevent aggressive shrinkage on small window sizes.
        self.cnn = nn.Sequential(
            # Layer 1
            nn.Conv1d(in_channels=num_features, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), # Halves the sequence length
            
            # Layer 2
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            # Layer 3
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        # --- MODULE 2: Temporal Extractor (Bi-LSTM) ---
        # Input size is 64 (from the last CNN out_channels)
        # Hidden size is 64, num_layers is 2, dropout is 0.5 (as per paper)
        self.bilstm = nn.LSTM(input_size=64, 
                              hidden_size=bilstm_hidden_size, 
                              num_layers=2, 
                              batch_first=True, 
                              dropout=0.5, 
                              bidirectional=True)

        # BiLSTM outputs hidden states of size: hidden_size * 2 (because it's bidirectional)
        self.lstm_output_dim = bilstm_hidden_size * 2  # 64 * 2 = 128

        # --- MODULE 3: RUL Regression Predictor ---
        # The paper specifically uses 1D Conv instead of Dense layers for final regression
        # Input channel: 1, Output channel: 1, Kernel: 128, Stride: 128
        self.rul_regressor = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=128, stride=128),
            nn.ReLU() # RUL cannot be negative
        )

        # --- MODULE 4: Domain Adaptation Classifier ---
        # 3 Linear layers for binary domain classification (Source = 0, Target = 1)
        self.domain_classifier = nn.Sequential(
            nn.Linear(self.lstm_output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2) # 2 Output classes for Source and Target
        )

    def forward(self, x: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the network.
        
        Args:
            x: Input tensor of shape (Batch_Size, Features=15, Window_Size)
            alpha: Weight for the Gradient Reversal Layer during training.
        
        Returns:
            rul_prediction: Shape (Batch_Size, 1)
            domain_prediction: Shape (Batch_Size, 2)
        """
        # 1. Spatial Feature Extraction (CNN)
        # Input: (Batch, 15, Window_Size) -> Output: (Batch, 64, Reduced_Window_Size)
        features = self.cnn(x)

        # 2. Temporal Feature Extraction (BiLSTM)
        # LSTM expects input shape: (Batch, Seq_Len, Features)
        # Permute to: (Batch, Reduced_Window_Size, 64)
        features = features.permute(0, 2, 1)
        lstm_out, _ = self.bilstm(features)

        # Take the output of the last time step
        # final_feature shape: (Batch, 128)
        final_feature = lstm_out[:, -1, :] 

        # 3. RUL Prediction (Conv1D)
        # Conv1D expects shape: (Batch, Channels, Length). So we add a channel dimension.
        # Shape becomes: (Batch, 1, 128)
        rul_input = final_feature.unsqueeze(1)
        rul_pred = self.rul_regressor(rul_input)
        # Flatten back to (Batch, 1)
        rul_pred = rul_pred.view(rul_pred.size(0), -1) 

        # 4. Domain Classification (with GRL)
        # Reverse gradients from the domain classifier to the feature extractor
        reversed_feature = grad_reverse(final_feature, alpha)
        domain_pred = self.domain_classifier(reversed_feature)

        return rul_pred, domain_pred

# ==============================================================================
# 3. TRAINING LOOP FOR DOMAIN ADAPTATION
# ==============================================================================
def train_domain_adaptation(model: nn.Module, 
                            source_loader, 
                            target_loader, 
                            epochs: int = 100, 
                            lr: float = 0.0001):
    """
    Training loop implementing Joint Loss: RUL Prediction Loss + Domain Classification Loss.
    Source: XJTU Dataset (Has RUL labels).
    Target: Lecturer's Misalignment Dataset (No RUL labels, used for domain alignment).
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Loss functions defined in the paper
    rul_criterion = nn.MSELoss()
    domain_criterion = nn.CrossEntropyLoss()
    
    model.train()
    
    for epoch in range(epochs):
        # Calculate dynamic alpha for Gradient Reversal Layer (optional but recommended)
        # Allows the model to learn features first before applying heavy adversarial penalties
        p = float(epoch) / epochs
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        
        # Iterate over source and target batches simultaneously
        for (source_x, source_rul), (target_x, _) in zip(source_loader, target_loader):
            
            optimizer.zero_grad()
            
            # Create domain labels: Source = 0, Target = 1
            batch_size = source_x.size(0)
            domain_label_source = torch.zeros(batch_size, dtype=torch.long)
            domain_label_target = torch.ones(target_x.size(0), dtype=torch.long)
            
            # --- 1. Forward Pass on Source Data (XJTU) ---
            rul_pred_source, domain_pred_source = model(source_x, alpha=alpha)
            
            # Calculate Source Losses
            loss_rul = rul_criterion(rul_pred_source, source_rul)
            loss_domain_source = domain_criterion(domain_pred_source, domain_label_source)
            
            # --- 2. Forward Pass on Target Data (Lecturer Data) ---
            _, domain_pred_target = model(target_x, alpha=alpha)
            loss_domain_target = domain_criterion(domain_pred_target, domain_label_target)
            
            # --- 3. Combined Loss & Backpropagation ---
            # Total Loss = MSE (Prediction) + CrossEntropy (Domain Classification)
            total_loss = loss_rul + loss_domain_source + loss_domain_target
            
            total_loss.backward()
            optimizer.step()
            
        print(f"Epoch [{epoch+1}/{epochs}] | Total Loss: {total_loss.item():.4f} | RUL Loss: {loss_rul.item():.4f}")

# Usage Example Notes:
# Make sure dataloaders return tensors of shape (Batch, 15, Window_Size)
# You can achieve this using torch.permute(0, 2, 1) on sliding window outputs.