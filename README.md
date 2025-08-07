# High-Performance Real-Time Trade Simulator for Cryptocurrency Markets

## Project Overview

This project, developed as an assignment for GoQuant, is a high-performance trade simulator designed to estimate transaction costs and market impact for cryptocurrency spot trades. It connects to a real-time WebSocket feed providing L2 order book data for the OKX exchange (specifically for BTC-USDT-SWAP via a GoQuant provided endpoint: `wss://ws.gomarket-cpp.goquant.io/ws/l2-orderbook/okx/BTC-USDT-SWAP`).

The application features a Python-based backend for data processing and financial modeling, and a Tkinter-based graphical user interface (GUI) for user input and real-time display of calculated metrics. It incorporates several models to estimate costs, including a dynamically trained linear regression model for slippage.

**Developed by**: `Aarya Gupta`  
**Date**: May 2024

![image](https://github.com/user-attachments/assets/eef1d9ab-1250-48f7-a2d2-eb78b0d16dec)
Figure 1: High-Level System Architecture Diagram - When the Regression-based Slippage Model is used to calculate Transaction Cost and the Model's Performance Metrics.

---

## Features

* **Real-Time Data Processing:** Connects to a WebSocket for live L2 order book updates (asks and bids).
* **Interactive UI:**
  * **Input Panel:** Allows users to specify trade parameters (Exchange, Asset, Order Type, Quantity in USD, Volatility, Fee Tier).
  * **Output Panel:** Dynamically displays calculated metrics in real-time.
* **Comprehensive Cost Estimation:**
  * **Expected Slippage:** Estimated using a Linear Regression model trained on-the-fly with synthetically generated data from live order book snapshots. The "walk-the-book" method is used for data generation and as a direct simulation baseline.
  * **Expected Fees:** Calculated using a rule-based model based on OKX fee structures and user-selected fee tier.
  * **Expected Market Impact:** Estimated using a simplified Almgren-Chriss model formulation.
  * **Net Cost:** Aggregated total of slippage, fees, and market impact.
* **Additional Metrics:**
  * **Maker/Taker Proportion:** Determined for market orders (100% Taker).
  * **Latency Monitoring:** Displays various internal processing latencies:
    * WebSocket data processing latency.
    * Financial calculation latency.
    * UI update (StringVar setting) latency.
    * End-to-end tick processing latency.
* **Machine Learning Integration:**
  * On-the-fly training of a `scikit-learn` Linear Regression model for slippage.
  * Real-time collection of probe data and periodic model retraining.
  * Display of model performance metrics (MSE, R2 Score, Training Samples) in the UI.
  * Logging of training data and model performance for offline analysis.
* **Data Analysis & Visualization:**
  * Includes a separate Python script (`analyze_slippage_data.py`) to perform offline analysis of logged data.
  * Generates plots for model performance evolution, feature vs. slippage relationships, and slippage distribution.
* **Robust Architecture:**
  * Multi-threaded design to separate network I/O from the UI, ensuring responsiveness.
  * Modular codebase for maintainability.
  * Comprehensive error handling and logging.

---

## System Architecture

The application is structured into several Python modules within the `src/` directory:

* **`main_app.py`**: The main application entry point. Manages the Tkinter UI, orchestrates other modules, and handles the primary application logic.
* **`websocket_handler.py`**: Responsible for establishing and maintaining the WebSocket connection, receiving messages, and passing them for processing. Runs in a separate thread.
* **`order_book_manager.py`**: Manages the L2 order book data structure (asks and bids), updating it with new data from the WebSocket and providing access to the current book state.
* **`financial_calculations.py`**: Contains all financial models and calculation logic for:
  * Fee calculation.
  * Slippage estimation (walk-the-book).
  * Market impact estimation (simplified Almgren-Chriss).
  * Slippage Regression Model (`SlippageRegressionModel` class for training and prediction).
* **`config.py`**: Stores configuration values like WebSocket URLs, fee rates, and default model parameters.
* **`analyze_slippage_data.py`** (in project root): Script for offline analysis and plot generation from logged data.

**Data Flow:**
1.`websocket_handler.py` connects to the WebSocket and receives L2 order book messages.
2. For each message, it records an arrival timestamp and updates the shared `OrderBookManager` instance. It also calculates the L1 latency (WS message parsing + book update).
3. It then calls a UI update callback (`schedule_ui_update` in `main_app.py`) with the updated book manager, status, and timing information.
4. `main_app.py`, via `_update_ui_from_websocket` (run in the main UI thread):
  * Updates live market data display (best bid/ask, spread, timestamp).
  * Generates probe data points using the current order book for the slippage regression model.
  * Periodically retrains the slippage regression model.
  * Calls `_recalculate_all_outputs()`.
5.  `_recalculate_all_outputs()`:
  * Reads current user inputs from the UI.
  * Calls functions in `financial_calculations.py` to get:
    * Slippage (from the trained regression model).
    * Fees.
    * Market Impact.
  * Calculates Net Cost.
  * Determines Maker/Taker proportion.
  * Updates all corresponding UI `StringVar`s.
  * Measures and updates calculation latency, UI update latency, and end-to-end latency.
  * Updates regression model performance metrics in the UI.
6.  Data for regression training/analysis and model performance is logged to CSV files.

---

## Setup and Execution

### Prerequisites

* Python 3.8+
* A VPN may be required to access the OKX data endpoint, depending on your location. (The specific GoQuant endpoint `wss://ws.gomarket-cpp.goquant.io/...` might be globally accessible).

### Installation

1.  **Clone the repository (if applicable, or extract the archive):**
    ```bash
    # git clone https://github.com/Aarya-Gupta/Trade_Simulator_OKX_L2.git
    # cd Trade_Simulator_OKX_L2
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    This will install `websockets`, `numpy`, `scikit-learn`, `pandas`, `matplotlib`, and `seaborn`.

### Running the Application

1.  **Navigate to the project root directory.**
2.  **Run the main application:**
    ```bash
    python -m src.main_app
    ```
    The GUI should appear. Ensure your internet connection (and VPN if needed) is active. The application will attempt to connect to the WebSocket and start displaying data.

### Running the Offline Analysis Script

After running the main application for a period (e.g., 15-30 minutes) to generate log files (`slippage_regression_log.csv`, `model_performance_log.csv`):

1.  **Ensure the log files are present in the project root.**
2.  **Run the analysis script:**
    ```bash
    python analyze_slippage_data.py
    ```
3.  Generated plots will be saved in the `output_plots/` directory.

---

## Models and Algorithms Implemented

### 1. Expected Fees

* **Type:** Rule-based model.
* **Logic:** Fees are calculated as `OrderValue_USD * TakerFeeRate`.
* **Taker Fee Rate:** Determined based on the user-selected "Fee Tier" from the UI. A mapping of fee tiers to taker rates is stored in `src/config.py` (based on example OKX spot trading fee structures). Market orders are assumed to be taker orders.
* **Example Tiers:** "Regular User LV1" (0.10%), "VIP 1" (0.08%), "VIP 8" (0.01%), etc.

### 2. Expected Slippage

Two approaches are used:

#### a. Walk-the-Book (Direct Simulation - for probe data generation)

* **Logic**: For a given target USD amount (for probes), this method simulates a market BUY order by iterating through the ask side of the live L2 order book, level by level, from best (lowest) price upwards, accumulating asset quantity until the target USD is spent or liquidity is exhausted.

* **Calculation**:
  1. `MidPrice_Snapshot = (BestAskPrice + BestBidPrice) / 2` (taken before the simulated trade).
  2. `AverageExecutionPrice = TotalActualUSD_Spent / TotalAsset_Acquired`.
  3. `SlippageValue = AverageExecutionPrice - MidPrice_Snapshot`.
  4. `SlippagePercentage = (SlippageValue / MidPrice_Snapshot) * 100`.

* This method provides the "true" slippage for the synthetic probe orders used to train the regression model.

#### b. Linear Regression Model (for UI display)

* **Objective**: To predict slippage percentage based on current market conditions and order size.
* **Model Type**: `sklearn.linear_model.LinearRegression`.
* **Features Used**:
  1. `OrderSize_USD`: The user-inputted order quantity in USD.
  2. `MarketSpread_bps`: Current bid-ask spread in basis points `((BestAsk - BestBid) / MidPrice) * 10000`.
  3. `MarketDepth_BestAsk_USD`: Total liquidity (USD value) available at the best ask price.
* **Target Variable**: `SlippagePercentage` (obtained from walk-the-book simulations of probe orders).
* **Training**:
  * **Data Generation**: On each WebSocket update (if the book is not crossed), several "probe" orders of predefined USD sizes (e.g., $1k, $10k, $100k, $1M) are simulated using the walk-the-book method. The resulting features and true slippage percentages are stored.
  * **On-the-Fly Training**: The model is trained initially after collecting `min_samples_to_train` (e.g., 1000) data points. It is then periodically retrained (e.g., every 200 WebSocket ticks) with all accumulated valid data. An 80/20 train-test split is used internally during training for evaluation.
* **Prediction**: For the user's specified order size, the current market spread and depth are extracted, and the trained linear model predicts the slippage percentage.
* **Output Capping**: Predicted negative slippage for BUY orders is capped at 0% for UI display.
* **Performance Metrics**: Test MSE and R2 Score are calculated after each training and displayed in the UI and logged.
  * *Observed R2 Scores*: Typically in the range of **[Your Observed R2 Range, e.g., 0.1 - 0.6+]** after sufficient training, indicating the model captures some, but not all, variance due to the simplicity of the linear model and inherent market noise.
  * *Observed MSE*: Very low (e.g., **[Your Observed MSE, e.g., 1e-8 to 1e-29]**), reflecting the predominance of near-zero true slippage in the probe data from the tight-spread feed.

### 3. Expected Market Impact

* **Model Type**: Simplified Almgren-Chriss like formulation for a single market order.
* **Objective**: Estimate the additional cost incurred due to the price movement caused by the trade itself.
* **Formula Used**:
  `MarketImpactCost_USD = C * Volatility * (OrderQuantity_USD / DailyVolume_USD) * OrderQuantity_USD`
  
  Where:
  * `C`: `MARKET_IMPACT_COEFFICIENT` (a configurable constant, e.g., 0.5, from `src/config.py`).
  * `Volatility`: User-inputted asset volatility (e.g., daily, as a decimal like 0.02 for 2%).
  * `OrderQuantity_USD`: The target USD value of the trade.
  * `DailyVolume_USD`: Assumed daily trading volume of the asset in USD (e.g., 5 Billion USD for BTC-USDT-SWAP, from `src/config.py`).
* **Assumptions**: This formula assumes impact cost is proportional to volatility, the square of the order size, and inversely proportional to daily volume. The coefficient `C` encapsulates other factors like temporary vs. permanent impact proportionality.

### 4. Net Cost

* **Calculation:** `NetCost_USD = SlippageCost_USD + FeeCost_USD + MarketImpactCost_USD`.
* **SlippageCost_USD:** Derived from the regression model's predicted slippage percentage: `(PredictedSlippagePercentage / 100.0) * OrderQuantity_USD`.

### 5. Maker/Taker Proportion

* **Logic:** For market orders, the trade is always a "Taker" order, consuming liquidity from the order book.
* **Output:** "100% Taker" (or "N/A" if order quantity is zero).
* **Note:** A logistic regression model, as mentioned in the assignment, would typically be used to predict the probability of a *limit order* being filled as a maker or taker based on its placement relative to the spread, size, etc. This was not implemented as the scope was market orders.

---

## Performance Analysis & Optimization

### Latency Benchmarking

The application measures and displays the following latencies in real-time:

1. **WS Proc. Latency (ms)**: Time taken from receiving a raw WebSocket message to parsing it (JSON) and updating the internal `OrderBookManager`.
   * *Typical Observed Value*: **[Your Value, e.g., 0.1 - 0.5 ms]**

2. **Calc. Latency (ms)**: Time taken for all financial calculations (slippage, fees, market impact, net cost, model predictions) in the `_recalculate_all_outputs` method.
   * *Typical Observed Value*: **[Your Value, e.g., 0.5 - 2.0 ms]** (may vary with regression model complexity if retrained within this path, though currently, retraining is separate).

3. **UI Update Latency (ms)**: Approximate time taken to set all `tk.StringVar` values in the UI for a single tick update. This does not include Tkinter's internal rendering time.
   * *Typical Observed Value*: **[Your Value, e.g., 0.1 - 0.3 ms]**

4. **End-to-End Latency (ms)**: Total time from WebSocket message arrival in the handler to all UI `StringVar`s being updated with new information for that tick.
   * *Typical Observed Value*: **[Your Value, e.g., 1.0 - 3.0 ms]**

**Overall System Performance:** The system generally processes data significantly faster than the typical WebSocket message arrival rate (e.g., ~100ms per message), ensuring no backlog.

### Optimization Techniques Implemented & Justified

* **Memory Management**:
  * Python's automatic garbage collection is leveraged.
  * `OrderBookManager`: For L2 snapshots, asks/bids lists are replaced. While efficient for snapshots, for very high-frequency diff-based updates, more specialized structures (e.g., sorted trees) would be considered in a production HFT system. Given the assignment's context, this is a practical balance.
  * `SlippageRegressionModel`: Training data (`data_X`, `data_y`) grows. For this assignment, this is unmanaged but would be capped in a continuously running production system to prevent unbounded memory use.

* **Network Communication**:
  * **Asynchronous Operations**: `asyncio` and the `websockets` library are used for non-blocking WebSocket communication, allowing the application to handle network I/O efficiently without freezing the UI or main logic.
  * **Ping/Pong Strategy**: Client-side pings were disabled (`ping_interval=None`) to rely on server-sent pings, potentially reducing unnecessary network traffic and improving stability with the provided endpoint.

* **Data Structure Selection**:
  * **Order Book**: Python `list` of `tuple`s `(price, quantity)` for asks and bids. These are sorted upon each update. Given that L2 updates are full snapshots, re-sorting is a clear and reasonably efficient approach. `float` conversions are performed once per data point.
  * **Regression Data**: Stored as Python `list`s and converted to `numpy` arrays for `scikit-learn` training, which is standard and efficient for this library.

* **Thread Management**:
  * A dedicated background thread is used for the WebSocket `asyncio` event loop. This isolates all network operations and initial data parsing from the main Tkinter UI thread, ensuring UI responsiveness.
  * Cross-thread UI updates are safely handled using `Tkinter.after(0, ...)` to marshal calls to the main UI thread.
  * Graceful shutdown logic attempts to stop the asyncio loop and join the WebSocket thread when the application window is closed.

* **Regression Model Efficiency**:
  * **Library Choice**: `scikit-learn`'s `LinearRegression` is implemented in C and optimized for performance.
  * **Periodic Training**: The model is not retrained on every single tick. Instead, training occurs after a configurable number of new data points (`min_samples_to_train`) are collected and then at intervals (`train_interval_ticks`), balancing model freshness with computational load.
  * **Simple Features**: The feature set for regression (order size, spread, best ask depth) is small and quick to compute.
  * **Probe Data Generation**: The `calculate_slippage_walk_book` function for generating probe data iterates through necessary book levels; its complexity is tied to the depth required to fill the probe orders.

---

## Known Issues / Observations

* **Crossed Order Books**: The provided WebSocket data feed (`wss://ws.gomarket-cpp.goquant.io/...`) occasionally (or frequently, depending on market conditions simulated by the feed) exhibits "crossed book" scenarios where the best ask price is reported as less than or equal to the best bid price.
  * The application detects this and logs a warning.
  * For slippage calculation during such events, the mid-price calculation falls back to using the best ask as the reference to avoid nonsensical results.
  * Probe data generation for the slippage regression model is **filtered** to exclude data points generated during crossed book states to improve the quality of training data.

* **Slippage Regression Model Performance**:
  * The R2 score for the on-the-fly trained linear regression model typically ranges from [Your Low R2] to [Your High R2] after sufficient data collection. This indicates that the simple linear model with the current features captures some, but not all, of the variance in slippage. The market microstructure and slippage are complex phenomena.
  * MSE values are generally very low (e.g., 1e-8 to 1e-29), primarily because the true slippage for many probe orders (especially smaller ones on the tight-spread feed) is near zero.

* **WebSocket Stability**: The WebSocket connection sometimes closes with a code 1006 (Abnormal Closure), potentially due to server-side idle timeouts or network interruptions. The application handles these disconnections and updates the UI status.

## Future Enhancements (Potential)

* Implement more sophisticated regression models (e.g., Gradient Boosting, Quantile Regression for different slippage percentiles).
* Add more features to the slippage model (e.g., historical volatility, order book imbalance).
* Allow user selection of different assets if the WebSocket supported dynamic subscriptions.
* Implement a more robust reconnection strategy for the WebSocket with exponential backoff.
* Persist the trained regression model to disk to avoid cold starts.
* Provide an option to use the "walk-the-book" slippage directly in the UI alongside the regression model's prediction for comparison.

---

This README aims to be a living document reflecting the project's state.
