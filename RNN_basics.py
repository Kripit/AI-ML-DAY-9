import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import logging
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import yfinance as yf
import pandas as pd
import os
import warnings
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s ",
    handlers=[logging.StreamHandler(), logging.FileHandler("stock_predictions.log")],
)
logger = logging.getLogger(__name__)

# Configuration class
class Config:
    def __init__(self):
        # Data parameters
        self.data_dir = "./stock_data"  # Directory to save raw and split data
        self.plot_dir = "./plots"  # Added for saving plots
        self.sequence_length = 60  # 60 days of historical data for predicton
        self.predict_steps = 1  # single-step prediction (easier to validate )

        # training parameters - designed to prevent overfitting
        self.epochs = 100
        self.batch_size = 32  # larger batch size for stable gradient
        self.patience = 15  # early stopping patience

        # Model architecture parameters
        self.hidden_size = 64  # 128 might cause overfitting so 128
        self.num_layers = 2
        self.dropout = 0.3

        # Optimizations parameters
        self.lr = 0.001
        self.weight_decay = 1e-4
        self.clip_grad_norm = 1.0

        # file paths
        self.model_path = "best_Stock_rnn_model.pt"
        self.scaler_path = "stock_scaler.pkl"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
        for split in ["train", "val", "test"]:
            os.makedirs(os.path.join(self.data_dir, split), exist_ok=True)

# Downloading the dataset , stock dataset for train test val
class StockDataDownloader:
    """
    Downloads and preprocesses stock data with proper train , test , val splits
    includes data normalization and feature engineering
    """

    def __init__(self, config):
        self.config = config
        self.scaler = MinMaxScaler(feature_range=(-1, 1))

    def download_stock_data(self, symbol="AAPL", years=5):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)

        logger.info(f"Downloading {symbol} data from {start_date.date()} to {end_date.date()}")

        try:
            stock_data = yf.download(
                symbol,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                progress=False,
            )
            if stock_data.empty:
                raise ValueError(f"No data found for symbol {symbol}")

            # Handle MultiIndex columns from yfinance
            if isinstance(stock_data.columns, pd.MultiIndex):
                # Flatten MultiIndex columns - take the first level (the actual column names)
                stock_data.columns = [col[0] if col[0] != '' else col[1] for col in stock_data.columns.values]
                logger.info(f"Flattened MultiIndex columns to: {list(stock_data.columns)}")

            # Reset index to make date a column
            stock_data.reset_index(inplace=True)
            
            # Ensure we have the expected columns
            expected_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in expected_columns if col not in stock_data.columns]
            if missing_columns:
                logger.error(f"Missing expected columns: {missing_columns}")
                logger.error(f"Available columns: {list(stock_data.columns)}")
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Clean the data immediately after download
            stock_data = self.clean_raw_data(stock_data, symbol)
            
            # Add technical indicators
            stock_data = self.add_technical_indicators(stock_data)

            raw_data_path = os.path.join(self.config.data_dir, f"{symbol}_raw.csv")
            stock_data.to_csv(raw_data_path, index=False)

            logger.info(f"Downloaded and cleaned {len(stock_data)} days of data for {symbol}")
            return stock_data

        except Exception as e:
            logger.error(f"Error downloading data for {symbol}: {str(e)}")
            raise

    def clean_raw_data(self, df, symbol):
        """
        Clean the raw data to remove any duplicate headers or symbol rows
        """
        logger.info(f"Cleaning raw data for {symbol}")
        logger.info(f"Original data shape: {df.shape}")
        logger.info(f"Original columns: {list(df.columns)}")
        
        # Remove any rows where numeric columns contain non-numeric values or the symbol
        if len(df) > 1:
            rows_to_drop = []
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            for idx, row in df.iterrows():
                # Check if any numeric column contains the symbol as a string
                is_symbol_row = False
                for col in numeric_columns:
                    if col in df.columns:
                        val = str(row[col]).upper()
                        if val == symbol.upper() or val == symbol.upper() + '.0':
                            is_symbol_row = True
                            break
                
                if is_symbol_row:
                    rows_to_drop.append(idx)
                    logger.info(f"Found symbol row at index {idx}")
            
            if rows_to_drop:
                df = df.drop(rows_to_drop)
                df = df.reset_index(drop=True)
                logger.info(f"Removed {len(rows_to_drop)} symbol rows")
        
        # Ensure numeric columns are actually numeric
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            if col in df.columns:
                # Convert to numeric, coercing errors to NaN
                original_dtype = df[col].dtype
                df[col] = pd.to_numeric(df[col], errors='coerce')
                logger.info(f"Converted {col} from {original_dtype} to numeric")
        
        # Remove any rows with NaN values in critical columns
        original_len = len(df)
        df = df.dropna(subset=['Close', 'Volume'])
        if len(df) < original_len:
            logger.info(f"Removed {original_len - len(df)} rows with NaN values")
        
        # Ensure Date column is datetime
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        
        # Sort by date to ensure proper chronological order
        if 'Date' in df.columns:
            df = df.sort_values('Date').reset_index(drop=True)
        
        # Final validation - check data ranges
        for col in ['Close', 'Volume']:
            if col in df.columns:
                min_val, max_val = df[col].min(), df[col].max()
                logger.info(f"{col} range: {min_val:.2f} to {max_val:.2f}")
        
        logger.info(f"Cleaned data shape: {df.shape}")
        return df

    def add_technical_indicators(self, df):
        """
        Adding technical indicators to improve prediction accuracy.
        These features help the model understand market trends.
        """
        logger.info("Adding technical indicators...")
        
        # Ensure we have the required columns
        required_cols = ['Close', 'High', 'Low', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Moving averages
        df["MA_5"] = df["Close"].rolling(window=5).mean()
        df["MA_10"] = df["Close"].rolling(window=10).mean()
        df["MA_20"] = df["Close"].rolling(window=20).mean()

        # Returns (percentage change)
        df["Returns"] = df["Close"].pct_change()

        # Volatility (standard deviation of returns)
        df["Volatility"] = df["Returns"].rolling(window=10).std()

        # Price range indicator
        df["Price_Range"] = (df["Close"] - df["Low"]) / (df["High"] - df["Low"] + 1e-8)  # Add small epsilon to avoid division by zero

        # Volume moving average
        df["Volume_MA"] = df["Volume"].rolling(window=10).mean()

        # Fill NaN values using forward and backward fill
        df.fillna(method="bfill", inplace=True)
        df.fillna(method="ffill", inplace=True)
        
        # Remove any remaining NaN values
        df.dropna(inplace=True)
        
        logger.info(f"Technical indicators added. Final shape: {df.shape}")
        return df

    def prepare_data_splits(self, data, symbol):
        """
        Split data into train/validation/test sets with proper temporal ordering.
        This is crucial for time series data - we can't randomly shuffle.
        """
        logger.info(f"Preparing data splits for {symbol}")
        
        # Sort by date to ensure proper temporal ordering
        data = data.sort_values("Date").reset_index(drop=True)
        
        # Log data info before splitting
        logger.info(f"Data shape before splitting: {data.shape}")
        logger.info(f"Date range: {data['Date'].min()} to {data['Date'].max()}")

        total_len = len(data)
        train_len = int(total_len * 0.7)
        val_len = int(total_len * 0.15)

        train_data = data[:train_len].copy()
        val_data = data[train_len : train_len + val_len].copy()
        test_data = data[train_len + val_len :].copy()

        feature_columns = [
            "Date",
            "Close",
            "Volume",
            "MA_5",
            "MA_10",
            "MA_20",
            "Volatility",
            "Price_Range",
            "Volume_MA",
        ]

        # Double-check for any remaining symbol rows in each split
        for split_name, df in [("train", train_data), ("val", val_data), ("test", test_data)]:
            original_len = len(df)
            # Remove any rows where non-Date columns contain the symbol
            numeric_cols = [col for col in feature_columns if col != "Date"]
            
            # Check for rows where any numeric column contains the symbol as a string
            mask = ~df[numeric_cols].apply(lambda row: row.astype(str).str.upper().eq(symbol.upper()).any(), axis=1)
            df = df[mask].copy()
            
            if len(df) < original_len:
                logger.info(f"Removed {original_len - len(df)} symbol rows from {split_name} split")
                
            # Update the split data
            if split_name == "train":
                train_data = df
            elif split_name == "val":
                val_data = df
            else:
                test_data = df

        # Ensure all numeric columns are actually numeric
        numeric_columns = [col for col in feature_columns if col != "Date"]
        for split_name, df in [("train", train_data), ("val", val_data), ("test", test_data)]:
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Drop rows with NaN values
            before_drop = len(df)
            df.dropna(subset=numeric_columns, inplace=True)
            after_drop = len(df)
            
            if before_drop != after_drop:
                logger.info(f"Dropped {before_drop - after_drop} rows with NaN from {split_name}")

        # Fit scaler only on training data
        logger.info("Fitting scaler on training data...")
        scaler_data = train_data[numeric_columns].values
        logger.info(f"Scaler input shape: {scaler_data.shape}")
        
        self.scaler.fit(scaler_data)

        # Transform all splits
        train_data[numeric_columns] = self.scaler.transform(train_data[numeric_columns].values)
        val_data[numeric_columns] = self.scaler.transform(val_data[numeric_columns].values)
        test_data[numeric_columns] = self.scaler.transform(test_data[numeric_columns].values)

        # Save the splits
        train_data[feature_columns].to_csv(os.path.join(self.config.data_dir, "train", f"{symbol}_train.csv"), index=False)
        val_data[feature_columns].to_csv(os.path.join(self.config.data_dir, "val", f"{symbol}_val.csv"), index=False)
        test_data[feature_columns].to_csv(os.path.join(self.config.data_dir, "test", f"{symbol}_test.csv"), index=False)

        # Save the scaler
        import joblib
        joblib.dump(self.scaler, os.path.join(self.config.data_dir, f"{symbol}_scaler.pkl"))

        logger.info(
            f"Data split completed: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}"
        )

        return train_data, val_data, test_data

class StockDataset(Dataset):
    """
    PyTorch Dataset class for handling stock data sequences.
    Creates sequences of historical data to predict future prices.
    """

    def __init__(self, data_dir, split_type="train", sequence_length=60):
        self.split_type = split_type
        self.sequence_length = sequence_length
        self.sequences = []
        self.targets = []

        split_dir = os.path.join(data_dir, split_type)
        
        # Check if directory exists
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Directory {split_dir} does not exist. Make sure data processing completed successfully.")
        
        # Check if directory has any CSV files
        csv_files = [f for f in os.listdir(split_dir) if f.endswith('.csv')]
        if not csv_files:
            raise ValueError(f"No CSV files found in {split_dir}")
        
        logger.info(f"Found {len(csv_files)} CSV files in {split_dir}: {csv_files}")
        
        feature_columns = [
            "Close",
            "Volume",
            "MA_5",
            "MA_10",
            "MA_20",
            "Volatility",
            "Price_Range",
            "Volume_MA",
        ]

        for filename in csv_files:
            filepath = os.path.join(split_dir, filename)
            try:
                df = pd.read_csv(filepath)
                logger.info(f"Processing {filename} - Shape: {df.shape}")
                
                # Validate and clean the dataframe
                df = self.validate_and_clean_dataframe(df, filename, feature_columns)
                
                if df is None or len(df) < self.sequence_length + 1:
                    logger.warning(f"Skipping {filename} - insufficient data after cleaning")
                    continue
                
                sequences, targets = self._create_sequences(df[feature_columns])
                if sequences and targets:
                    self.sequences.extend(sequences)
                    self.targets.extend(targets)
                    logger.info(f"Added {len(sequences)} sequences from {filename}")
                
            except Exception as e:
                logger.error(f"Error processing {filename}: {str(e)}")
                continue

        if not self.sequences:
            raise ValueError(f"No valid sequences found for {split_type} set")

        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)

        logger.info(f"Created {len(self.sequences)} sequences for {split_type} set")

    def validate_and_clean_dataframe(self, df, filename, feature_columns):
        """
        Validate and clean the dataframe to ensure it contains valid data
        """
        # Check if all required columns exist
        missing_cols = [col for col in feature_columns if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing columns {missing_cols} in {filename}")
            return None
        
        # Select only the required columns
        df = df[feature_columns + ["Date"] if "Date" in df.columns else feature_columns].copy()
        
        # Convert all feature columns to numeric
        for col in feature_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Check for and log any non-finite values
        for col in feature_columns:
            non_finite_count = (~np.isfinite(df[col])).sum()
            if non_finite_count > 0:
                logger.warning(f"Found {non_finite_count} non-finite values in {col} of {filename}")
        
        # Remove rows with any non-finite values
        original_len = len(df)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        
        if len(df) < original_len:
            logger.info(f"Removed {original_len - len(df)} rows with invalid values from {filename}")
        
        return df if len(df) > 0 else None

    def _create_sequences(self, df):
        """
        Create input sequences and corresponding targets from dataframe.
        Each sequence contains 'sequence_length' days of historical data.
        """
        sequences = []
        targets = []

        for i in range(len(df) - self.sequence_length):
            sequence = df.iloc[i : i + self.sequence_length].values
            target = df["Close"].iloc[i + self.sequence_length]
            
            # Validate sequence and target
            if not np.isfinite(sequence).all() or not np.isfinite(target):
                continue
                
            sequences.append(sequence)
            targets.append(target)

        return sequences, targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.float32), torch.tensor(self.targets[idx], dtype=torch.float32)

class AdvancedStockRNN(nn.Module):
    """
    Advanced RNN model combining RNN, GRU, and Bidirectional RNN with attention.
    Includes multiple regularization techniques to prevent overfitting.
    """

    def __init__(self, input_size=8, hidden_size=64, num_layers=2, dropout=0.3):
        super(AdvancedStockRNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # layer 1: Standar RNN for basic sequential processing
        self.rnn = nn.RNN(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Layer 2: GRU for better gradient flow ane memory
        self.gru = nn.GRU(
            hidden_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Bidriectional RNN for capturing both past and future context
        self.bi_rnn = nn.GRU(
            hidden_size,
            hidden_size // 2,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Attention mechanism for focusing on important time steps
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_size)  # to normalize every mini - batch's variance activations ( mean = 0 , variance = 1)

        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 1),
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Output tensor of shape (batch_size, 1)
        """
        batch_size = x.size(0)

        # Initialize hidden state
        h0_rnn = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        h0_gru = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        h0_bi = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size // 2).to(x.device)

        # Pass through the rnn layers
        rnn_out, _ = self.rnn(x, h0_rnn)  # (batch, seq_len, hidden_size)
        gru_out, _ = self.gru(rnn_out, h0_gru)
        bi_out, _ = self.bi_rnn(gru_out, h0_bi)

        attention_scores = self.attention(bi_out)  # (batch, seq_len, 1)
        attention_weights = torch.softmax(attention_scores, dim=1)  # (batch, seq_len, 1)

        # Weughted some of hidden states
        context_vector = torch.sum(attention_weights * bi_out, dim=1)  # (batch, hidden_size)

        context_vector = self.dropout(context_vector)

        if context_vector.size(0) > 1:
            context_vector = self.batch_norm(context_vector)

        output = self.fc_layers(context_vector)

        return output

class ModelTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []

        self.criterion = nn.MSELoss()

        self.optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

        self.scheduler_cosine = CosineAnnealingLR(self.optimizer, T_max=config.epochs)
        self.scheduler_plateau = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=7
        )

        self.scaler = GradScaler()

    def train_epoch(self, train_loader):
        """Train the model for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for sequences, targets in train_loader:
            sequences, targets = sequences.to(self.config.device), targets.to(self.config.device)

            self.optimizer.zero_grad()

            with autocast():
                predictions = self.model(sequences)
                loss = self.criterion(predictions.squeeze(), targets)

            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()

            # Gradient clipping to prevent exploding gradients
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def validate_epoch(self, val_loader):
        """Validate the model for one epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences, targets = sequences.to(self.config.device), targets.to(self.config.device)

                with autocast():
                    predictions = self.model(sequences)
                    loss = self.criterion(predictions.squeeze(), targets)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def train(self, train_loader, val_loader):
        logger.info("Starting model training")

        for epoch in range(self.config.epochs):
            train_loss = self.train_epoch(train_loader)

            val_loss = self.validate_epoch(val_loader)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            self.scheduler_cosine.step()
            self.scheduler_plateau.step(val_loss)

            # early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0

                # save best model
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "train_loss": train_loss,
                    },
                    self.config.model_path,
                )

                logger.info(f"New best model saved at epoch {epoch + 1}")

            else:
                self.patience_counter += 1

            # Log progress
            if (epoch + 1) % 10 == 0 or self.patience_counter == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{self.config.epochs}, "
                    f"Train Loss: {train_loss:.6f}, "
                    f"Val Loss: {val_loss:.6f}, "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.7f}"
                )

            # Early stopping
            if self.patience_counter >= self.config.patience:
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

        logger.info("Training completed!")

    def plot_training_history(self):
        """Plot training and validation loss curves."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label="Training Loss", color="blue")
        plt.plot(self.val_losses, label="Validation Loss", color="red")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Model Training History")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.config.plot_dir, "training_history.png"), dpi=300, bbox_inches="tight")
        plt.close()

def evaluate_model(model, test_loader, config, scaler):
    """
    Comprehensive model evaluation with multiple metrics.
    """
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for sequences, targets in test_loader:
            sequences, targets = sequences.to(config.device), targets.to(config.device)

            with autocast():
                pred = model(sequences)
                predictions.extend(pred.cpu().numpy())
                actuals.extend(targets.cpu().numpy())

    predictions = np.array(predictions).flatten()  # flatten used to conver multi dimensional array into single dimesnional array
    actuals = np.array(actuals).flatten()

    # Denormalize predictions and actuals for meaningful metrics
    predictions_denorm = scaler.inverse_transform(
        np.column_stack([predictions] + [np.zeros((len(predictions), 7))])
    )[:, 0]
    actuals_denorm = scaler.inverse_transform(
        np.column_stack([actuals] + [np.zeros((len(actuals), 7))])
    )[:, 0]

    # Calculate metrics
    mse = mean_squared_error(actuals_denorm, predictions_denorm)
    mae = mean_absolute_error(actuals_denorm, predictions_denorm)
    rmse = np.sqrt(mse)
    r2 = r2_score(actuals_denorm, predictions_denorm)

    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((actuals_denorm - predictions_denorm) / actuals_denorm)) * 100

    logger.info("=== Model Evaluation Results ===")
    logger.info(f"Mean Squared Error (MSE): {mse:.4f}")
    logger.info(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    logger.info(f"Mean Absolute Error (MAE): {mae:.4f}")
    logger.info(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    logger.info(f"R-squared Score: {r2:.4f}")

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "r2": r2,
        "predictions": predictions_denorm,
        "actuals": actuals_denorm,
    }

def test_yfinance_download():
    """
    Test function to debug yfinance download issues
    """
    logger.info("Testing yfinance download...")
    
    try:
        # Simple test download
        stock_data = yf.download("AAPL", start="2024-01-01", end="2024-12-31", progress=False)
        
        logger.info(f"Downloaded data shape: {stock_data.shape}")
        logger.info(f"Columns: {stock_data.columns.tolist()}")
        logger.info(f"Column types: {type(stock_data.columns)}")
        logger.info(f"Index: {stock_data.index}")
        logger.info(f"First 3 rows:\n{stock_data.head(3)}")
        
        # Handle MultiIndex if present
        if isinstance(stock_data.columns, pd.MultiIndex):
            logger.info("MultiIndex detected - flattening...")
            stock_data.columns = [col[0] if col[0] != '' else col[1] for col in stock_data.columns.values]
            logger.info(f"Flattened columns: {stock_data.columns.tolist()}")
        
        # Reset index
        stock_data.reset_index(inplace=True)
        logger.info(f"After reset_index - columns: {stock_data.columns.tolist()}")
        logger.info(f"After reset_index - shape: {stock_data.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"yfinance test failed: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def main():
    """
    Main execution function that orchestrates the entire pipeline.
    """
    # First, test yfinance
    logger.info("Running yfinance test...")
    if not test_yfinance_download():
        logger.error("yfinance test failed. Please check your internet connection and yfinance installation.")
        return
    
    # Initialize configuration
    config = Config()

    # Download and prepare data
    logger.info("Step 1: Downloading stock data...")
    downloader = StockDataDownloader(config)

    # Download multiple stocks for diversified training
    stocks = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
    successful_stocks = []

    for stock in stocks:
        try:
            stock_data = downloader.download_stock_data(stock, years=5)
            downloader.prepare_data_splits(stock_data, stock)
            successful_stocks.append(stock)
            logger.info(f"Successfully processed {stock}")
        except Exception as e:
            logger.error(f"Failed to process {stock}: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            continue

    # Check if any stocks were processed successfully
    if not successful_stocks:
        logger.error("No stocks were processed successfully. Exiting.")
        return

    logger.info(f"Successfully processed {len(successful_stocks)} stocks: {successful_stocks}")

    # Verify that data directories exist and contain files
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(config.data_dir, split)
        if not os.path.exists(split_dir):
            logger.error(f"Directory {split_dir} does not exist")
            return
        
        csv_files = [f for f in os.listdir(split_dir) if f.endswith('.csv')]
        if not csv_files:
            logger.error(f"No CSV files found in {split_dir}")
            return
        
        logger.info(f"Found {len(csv_files)} files in {split}: {csv_files}")

    # Create datasets and data loaders
    logger.info("Step 2: Creating datasets...")
    try:
        train_dataset = StockDataset(config.data_dir, "train", config.sequence_length)
        val_dataset = StockDataset(config.data_dir, "val", config.sequence_length)
        test_dataset = StockDataset(config.data_dir, "test", config.sequence_length)
    except ValueError as e:
        logger.error(f"Failed to create datasets: {str(e)}")
        return
    except FileNotFoundError as e:
        logger.error(f"Directory not found: {str(e)}")
        logger.error("Make sure the data processing completed successfully")
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Initialize model
    logger.info("Step 3: Initializing model...")
    model = AdvancedStockRNN(
        input_size=8,  # Number of features
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(config.device)

    # Log model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Train model
    logger.info("Step 4: Training model...")
    trainer = ModelTrainer(model, config)
    trainer.train(train_loader, val_loader)

    # Plot training history
    trainer.plot_training_history()

    # Load best model for evaluation
    logger.info("Step 5: Evaluating model...")
    try:
        checkpoint = torch.load(config.model_path, map_location=config.device)
        model.load_state_dict(checkpoint["model_state_dict"])
    except FileNotFoundError:
        logger.error("No trained model found. Skipping evaluation.")
        return

    # Evaluate on test set
    import joblib
    scaler_path = os.path.join(config.data_dir, f"{stocks[0]}_scaler.pkl")
    try:
        scaler = joblib.load(scaler_path)
    except FileNotFoundError:
        logger.error(f"Scaler file {scaler_path} not found. Skipping evaluation.")
        return

    results = evaluate_model(model, test_loader, config, scaler)

    # Plot predictions vs actuals
    plt.figure(figsize=(15, 8))
    plt.plot(results["actuals"][:100], label="Actual Prices", color="blue", alpha=0.7)
    plt.plot(results["predictions"][:100], label="Predicted Prices", color="red", alpha=0.7)
    plt.xlabel("Time Steps")
    plt.ylabel("Stock Price ($)")
    plt.title("Stock Price Predictions vs Actual (First 100 Test Samples)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(config.plot_dir, "predictions_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("Pipeline completed successfully!")

if __name__ == "__main__":
    main()