import tensorflow as tf
from tensorflow.keras import layers, models, Model
import numpy as np
from typing import Tuple

class TCN_BiLSTM_Attention:
    """
    TCN-BiLSTM-Multi-Head Attention Hybrid Model for RUL Prediction.
    Based on the architecture proposed by Guo et al. (IEEE Access).
    """

    def __init__(self, 
                 window_size: int, 
                 num_features: int = 15, 
                 num_heads: int = 4):
        """
        Initializes the model architecture parameters.

        Args:
            window_size (int): The sequence length of the sliding window (e.g., 30, 40, 50).
            num_features (int): The number of extracted physical features (default is 15).
            num_heads (int): Number of attention heads for the MHA layer.
        """
        self.window_size = window_size
        self.num_features = num_features
        self.num_heads = num_heads
        self.model = self._build_model()

    def _residual_tcn_block(self, 
                            x: tf.Tensor, 
                            filters: int, 
                            kernel_size: int, 
                            dilation_rate: int) -> tf.Tensor:
        """
        Constructs a single Temporal Convolutional Network (TCN) residual block 
        with dilated causal convolution.

        Args:
            x (tf.Tensor): Input tensor.
            filters (int): Number of convolutional filters.
            kernel_size (int): Size of the convolutional kernel.
            dilation_rate (int): Dilation rate for the causal convolution.

        Returns:
            tf.Tensor: Output tensor of the TCN block.
        """
        # Save input for residual connection
        res = x
        
        # Dilated Causal Convolution (causal padding prevents data leakage from the future)
        x = layers.Conv1D(filters=filters, 
                          kernel_size=kernel_size, 
                          padding='causal', 
                          dilation_rate=dilation_rate, 
                          activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        # 1x1 Conv to match dimensions if filters are different for the residual connection
        if res.shape[-1] != filters:
            res = layers.Conv1D(filters=filters, kernel_size=1, padding='same')(res)
            
        # Residual Addition
        x = layers.Add()([x, res])
        return layers.ReLU()(x)

    def _build_model(self) -> Model:
        """
        Builds the complete TCN-BiLSTM-MHA model.

        Returns:
            tf.keras.Model: The compiled Keras model.
        """
        # 1. Input Layer
        inputs = layers.Input(shape=(self.window_size, self.num_features), name="Input_Features")

        # 2. Initial Convolution (Adjusted strides=1 for pre-extracted features)
        # Based on paper: filters=16, kernel=12
        x = layers.Conv1D(filters=16, kernel_size=12, padding='causal', activation='relu')(inputs)
        # Removed aggressive pooling to preserve time steps for the window size

        # 3. TCN Modules (Parameters based on Table 1 of the paper)
        # Block 1: filters=12, kernel=3, dilation=1
        x = self._residual_tcn_block(x, filters=12, kernel_size=3, dilation_rate=1)
        # Block 2: filters=6, kernel=5, dilation=2
        x = self._residual_tcn_block(x, filters=6, kernel_size=5, dilation_rate=2)
        # Block 3: filters=4, kernel=7, dilation=4
        x = self._residual_tcn_block(x, filters=4, kernel_size=7, dilation_rate=4)

        # 4. Bidirectional LSTM Layer
        # Based on paper: 32 neurons
        x = layers.Bidirectional(layers.LSTM(units=32, return_sequences=True))(x)

        # 5. Multi-Head Attention Mechanism
        # Query, Key, and Value are all derived from the BiLSTM output (Self-Attention)
        attention_output = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=32)(x, x)
        
        # Add & Norm (Standard Transformer block practice for stability)
        x = layers.Add()([x, attention_output])
        x = layers.LayerNormalization()(x)

        # 6. Global Features Extraction & Output Layer
        # Flatten the temporal dimension to feed into the dense layer
        x = layers.GlobalAveragePooling1D()(x)
        
        # Output layer for regression (1 neuron, ReLU activation as specified in paper)
        # ReLU ensures RUL predictions are strictly non-negative
        outputs = layers.Dense(units=1, activation='relu', name="RUL_Prediction")(x)

        model = models.Model(inputs=inputs, outputs=outputs, name="TCN_BiLSTM_MHA")
        
        # Compile model
        # Using Adam optimizer and MSE loss (standard for regression tasks)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss='mse',
                      metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')])
        return model

    def get_summary(self):
        """Prints the architectural summary of the model."""
        self.model.summary()

    def train(self, 
              X_train: np.ndarray, 
              y_train: np.ndarray, 
              epochs: int = 100, 
              batch_size: int = 64, 
              validation_split: float = 0.2):
        """
        Trains the model on the provided dataset.
        """
        print("Starting Model Training...")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
            ]
        )
        return history

# ==========================================
# USAGE EXAMPLE
# ==========================================
if __name__ == "__main__":
    # 1. Configuration matching preprocessing pipeline
    WINDOW_SIZE = 30
    NUM_FEATURES = 15
    
    # 2. Instantiate the Model
    rul_model = TCN_BiLSTM_Attention(window_size=WINDOW_SIZE, num_features=NUM_FEATURES)
    rul_model.get_summary()
    
    # 3. Dummy Data Simulation (e.g., 1000 sliding window samples)
    # Shape: (Samples, Window Size, Features) -> (1000, 50, 15)
    dummy_X_train = np.random.normal(0, 1, (1000, WINDOW_SIZE, NUM_FEATURES))
    # Shape: (Samples, 1) -> Target RUL normalized between [4]
    dummy_y_train = np.random.uniform(0, 1, (1000, 1))
    
    # 4. Train the Model
    rul_model.train(dummy_X_train, dummy_y_train, epochs=50)