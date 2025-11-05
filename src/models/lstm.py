import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class LSTMVol(nn.Module):
    """
    LSTM model for volatility forecasting

    Architecture:
    - Multi-layer LSTM for sequence processing
    - Linear head with Softplus activation (ensures σ ≥ 0)
    """

    def __init__(self, n_feats: int, hidden: int = 64, layers: int = 2, dropout: float = 0.1):
        """
        Parameters:
        -----------
        n_feats : int
            Number of input features
        hidden : int
            Hidden state dimension
        layers : int
            Number of LSTM layers
        dropout : float
            Dropout rate between LSTM layers
        """
        super().__init__()
        self.n_feats = n_feats
        self.hidden = hidden
        self.layers = layers

        self.lstm = nn.LSTM(
            input_size=n_feats,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, 1),
            nn.Softplus()  # enforce σ ≥ 0
        )

    def forward(self, x):
        """
        Forward pass

        Parameters:
        -----------
        x : torch.Tensor
            Input sequences (batch, seq_len, features)

        Returns:
        --------
        torch.Tensor
            Volatility predictions (batch,)
        """
        out, _ = self.lstm(x)
        # Use last time step output
        return self.head(out[:, -1, :]).squeeze(-1)


class LSTMVolTrainer:
    """
    Trainer class for LSTM volatility forecasting
    """

    def __init__(
        self,
        model: LSTMVol,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        device: Optional[str] = None
    ):
        """
        Parameters:
        -----------
        model : LSTMVol
            LSTM model to train
        lr : float
            Learning rate
        weight_decay : float
            L2 regularization
        device : str, optional
            Device to use ('cuda' or 'cpu')
        """
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=False
        )

        self.history = {
            'train_loss': [],
            'val_loss': []
        }

    def qlike_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Quasi-likelihood loss for volatility forecasting

        QLIKE = log(σ²) + y² / σ²
        """
        eps = 1e-8
        return torch.mean(
            torch.log(y_pred**2 + eps) + (y_true**2) / (y_pred**2 + eps)
        )

    def mse_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """MSE loss"""
        return torch.mean((y_pred - y_true)**2)

    def train_epoch(
        self,
        train_loader: DataLoader,
        loss_fn: str = 'qlike'
    ) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # Forward pass
            y_pred = self.model(X_batch)

            # Compute loss
            if loss_fn == 'qlike':
                loss = self.qlike_loss(y_pred, y_batch)
            else:
                loss = self.mse_loss(y_pred, y_batch)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
        loss_fn: str = 'qlike'
    ) -> float:
        """Validate on validation set"""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            y_pred = self.model(X_batch)

            if loss_fn == 'qlike':
                loss = self.qlike_loss(y_pred, y_batch)
            else:
                loss = self.mse_loss(y_pred, y_batch)

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        loss_fn: str = 'qlike',
        early_stopping_patience: int = 20,
        verbose: bool = True
    ) -> Dict:
        """
        Train the model

        Parameters:
        -----------
        train_loader : DataLoader
            Training data loader
        val_loader : DataLoader, optional
            Validation data loader
        epochs : int
            Number of epochs
        loss_fn : str
            Loss function ('qlike' or 'mse')
        early_stopping_patience : int
            Patience for early stopping
        verbose : bool
            Print training progress

        Returns:
        --------
        dict
            Training history
        """
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader, loss_fn)
            self.history['train_loss'].append(train_loss)

            # Validate
            if val_loader is not None:
                val_loss = self.validate(val_loader, loss_fn)
                self.history['val_loss'].append(val_loss)

                # Learning rate scheduling
                self.scheduler.step(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    self.best_state = {
                        'epoch': epoch,
                        'model_state': self.model.state_dict(),
                        'optimizer_state': self.optimizer.state_dict(),
                        'val_loss': val_loss
                    }
                else:
                    patience_counter += 1

                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {train_loss:.4f}, "
                          f"Val Loss: {val_loss:.4f}")

                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}")

        # Load best model if validation was used
        if val_loader is not None and hasattr(self, 'best_state'):
            self.model.load_state_dict(self.best_state['model_state'])
            if verbose:
                print(f"\nBest model from epoch {self.best_state['epoch']+1} "
                      f"with val loss: {self.best_state['val_loss']:.4f}")

        return self.history

    @torch.no_grad()
    def predict(self, X: torch.Tensor) -> np.ndarray:
        """
        Make predictions

        Parameters:
        -----------
        X : torch.Tensor
            Input sequences (batch, seq_len, features)

        Returns:
        --------
        np.ndarray
            Predictions
        """
        self.model.eval()
        X = X.to(self.device)
        predictions = self.model(X)
        return predictions.cpu().numpy()


def create_sequences(
    data: pd.DataFrame,
    target_col: str,
    feature_cols: list,
    seq_len: int = 30,
    forecast_horizon: int = 1
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Create sequences for LSTM training

    Parameters:
    -----------
    data : pd.DataFrame
        Input data
    target_col : str
        Target column name (realized volatility)
    feature_cols : list
        Feature column names
    seq_len : int
        Sequence length (lookback window)
    forecast_horizon : int
        Forecast horizon (days ahead)

    Returns:
    --------
    X : np.ndarray
        Feature sequences (n_samples, seq_len, n_features)
    y : np.ndarray
        Target values (n_samples,)
    dates : pd.DatetimeIndex
        Corresponding dates
    """
    n = len(data)
    n_feats = len(feature_cols)

    X_list = []
    y_list = []
    date_list = []

    for i in range(seq_len, n - forecast_horizon + 1):
        # Features: [i-seq_len, i)
        X_seq = data[feature_cols].iloc[i-seq_len:i].values

        # Target: i + forecast_horizon - 1
        target_idx = i + forecast_horizon - 1
        y_val = data[target_col].iloc[target_idx]

        X_list.append(X_seq)
        y_list.append(y_val)
        date_list.append(data.index[target_idx])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    dates = pd.DatetimeIndex(date_list)

    return X, y, dates


def prepare_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    batch_size: int = 32,
    shuffle_train: bool = True
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Prepare PyTorch DataLoaders

    Parameters:
    -----------
    X_train, y_train : np.ndarray
        Training data
    X_val, y_val : np.ndarray, optional
        Validation data
    batch_size : int
        Batch size
    shuffle_train : bool
        Shuffle training data

    Returns:
    --------
    train_loader : DataLoader
        Training data loader
    val_loader : DataLoader, optional
        Validation data loader
    """
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train
    )

    val_loader = None
    if X_val is not None and y_val is not None:
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val)
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        )

    return train_loader, val_loader


def rolling_lstm_forecast(
    data: pd.DataFrame,
    target_col: str,
    feature_cols: list,
    seq_len: int = 30,
    train_window: int = 756,  # 3 years
    refit_freq: int = 20,  # Monthly
    lstm_hidden: int = 64,
    lstm_layers: int = 2,
    epochs: int = 50,
    batch_size: int = 32,
    verbose: bool = False
) -> pd.Series:
    """
    Generate rolling LSTM forecasts with periodic refitting

    Parameters:
    -----------
    data : pd.DataFrame
        Input data with features and target
    target_col : str
        Target column name
    feature_cols : list
        Feature column names
    seq_len : int
        Sequence length
    train_window : int
        Training window size (days)
    refit_freq : int
        Refit frequency (days)
    lstm_hidden : int
        LSTM hidden size
    lstm_layers : int
        Number of LSTM layers
    epochs : int
        Training epochs per fit
    batch_size : int
        Batch size
    verbose : bool
        Print progress

    Returns:
    --------
    pd.Series
        Rolling forecasts
    """
    forecasts = []
    dates = []

    start_idx = train_window + seq_len
    last_fit_idx = None
    trained_model = None

    if verbose:
        print(f"Starting rolling LSTM forecasts...")
        print(f"Train window: {train_window} days, Refit every: {refit_freq} days")

    for i in range(start_idx, len(data)):
        # Check if we need to refit
        should_refit = (
            trained_model is None or
            last_fit_idx is None or
            (i - last_fit_idx) >= refit_freq
        )

        if should_refit:
            # Get training data
            train_start = max(0, i - train_window)
            train_data = data.iloc[train_start:i]

            # Create sequences
            X_train, y_train, _ = create_sequences(
                train_data,
                target_col=target_col,
                feature_cols=feature_cols,
                seq_len=seq_len,
                forecast_horizon=1
            )

            if len(X_train) < 100:  # Need minimum samples
                continue

            # Split train/val (80/20)
            n_train = int(0.8 * len(X_train))
            X_tr, y_tr = X_train[:n_train], y_train[:n_train]
            X_val, y_val = X_train[n_train:], y_train[n_train:]

            # Prepare dataloaders
            train_loader, val_loader = prepare_dataloaders(
                X_tr, y_tr, X_val, y_val,
                batch_size=batch_size
            )

            # Create and train model
            model = LSTMVol(
                n_feats=len(feature_cols),
                hidden=lstm_hidden,
                layers=lstm_layers,
                dropout=0.2
            )

            trainer = LSTMVolTrainer(model, lr=1e-3, weight_decay=1e-5)

            try:
                trainer.fit(
                    train_loader,
                    val_loader,
                    epochs=epochs,
                    early_stopping_patience=15,
                    verbose=False
                )
                trained_model = trainer
                last_fit_idx = i

                if verbose and i % 100 == 0:
                    print(f"Refitted model at index {i} ({data.index[i].date()})")
            except Exception as e:
                if verbose:
                    print(f"Training failed at index {i}: {e}")
                continue

        # Make prediction
        if trained_model is not None:
            try:
                # Get last seq_len observations
                test_seq = data[feature_cols].iloc[i-seq_len:i].values
                test_seq = test_seq.reshape(1, seq_len, -1)  # Add batch dimension

                pred = trained_model.predict(torch.FloatTensor(test_seq))[0]

                forecasts.append(pred)
                dates.append(data.index[i])
            except Exception as e:
                if verbose:
                    print(f"Prediction failed at index {i}: {e}")
                continue

    if verbose:
        print(f"Generated {len(forecasts)} forecasts")

    return pd.Series(forecasts, index=dates, name='lstm_forecast')


if __name__ == "__main__":
    print("LSTM Volatility Forecasting Module")
    print("=" * 50)
    print("\nAvailable components:")
    print("- LSTMVol: LSTM model architecture")
    print("- LSTMVolTrainer: Training and evaluation")
    print("- create_sequences: Prepare time series data")
    print("- rolling_lstm_forecast: Rolling forecast generation")