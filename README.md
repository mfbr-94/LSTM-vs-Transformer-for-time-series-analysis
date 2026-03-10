LSTM vs Transformer for Time Series Analysis
A comparative deep learning study benchmarking LSTM and Transformer architectures for financial time series anomaly detection and forecasting. Implemented in PyTorch, this project evaluates model performance on historical stock data (MSFT) using reconstruction-based anomaly detection and multi-horizon forecasting.
🚀 Features
Dual Architecture Comparison: LSTM Autoencoder vs. Transformer Autoencoder.
Anomaly Detection: Unsupervised learning via reconstruction error thresholding.
Multi-Horizon Forecasting: Predicts 1-day, 5-day, and 21-day log returns with prediction intervals.
Automated Pipeline: Data downloading (yfinance), feature engineering, training, and evaluation.
Visualization: Generates interactive HTML (Plotly) and static PNG (Matplotlib) reports.
📦 Requirements
Ensure you have Python 3.8+ installed. Install dependencies via pip:
pip install torch numpy pandas yfinance matplotlib scikit-learn plotly
⚡ Usage
Clone the repository.
Run the main script:
python main.py
Configuration:
Ticker: Default MSFT (editable in script).
Sequence Length: 30 trading days.
Train/Test Split: 70% / 30% (chronological).
Device: Auto-detects CUDA or CPU.

Model
Type
Mechanism
Objective
LSTM Autoencoder
Recurrent
Encoder compresses sequence to latent vector; Decoder reconstructs input.
Minimize Reconstruction Error
Transformer Autoencoder
Attention
Self-attention captures global dependencies; MLP reconstructs input.
Minimize Reconstruction Error
Transformer Forecaster
Attention
Encoder extracts context from final token; Linear head predicts future return.
Minimize Forecast MSE

🧠 Methodology
Feature Engineering: Log returns, rolling volatility, momentum, volume z-scores, and drawdown.
Anomaly Labeling: Weak supervision using the top 1% of absolute log-returns as proxy ground truth.
Normalization: StandardScaler fit on training data only to prevent look-ahead bias.
Evaluation Metrics: ROC-AUC, PR-AUC (critical for imbalanced data), and MSE.
Uncertainty Quantification: Prediction intervals derived from test set residual standard deviation.
📊 Outputs
Upon completion, the script generates the following artifacts:
File
Description
model_performance_comparison.png
Bar chart comparing ROC-AUC & PR-AUC scores.
anomaly_scores.png
Timeline of reconstruction errors vs. proxy anomalies.
price_with_anomalies.png
Stock price chart with detected anomaly markers.
*.html
Interactive versions of the above plots (if Plotly installed).
Console
Forecast summaries with 50%, 78%, and 92% prediction intervals.

⚠️ Disclaimer
This project is for educational and research purposes only.
Not financial advice.
Past performance does not guarantee future results.
Transaction costs, slippage, and market impact are not modeled.
Do not use for live trading without extensive validation.
📄 License
MIT License
