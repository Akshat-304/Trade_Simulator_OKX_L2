# src/financial_calculations.py
# Run to infer standalone tests :
# python -m src.financial_calculations
# will implement pytest later.

import logging
from typing import Tuple, Optional, List, Dict  # For type hinting
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from .config import (
    OKX_FEE_RATES,
    DEFAULT_TAKER_FEE_RATE,
    ASSUMED_DAILY_VOLUME_USD,
    MARKET_IMPACT_COEFFICIENT,
)

# We'll need access to the OrderBookManager type for type hinting if not already imported
# from .order_book_manager import OrderBookManager # Assuming it's in the same directory
import numpy as np
from sklearn.linear_model import LinearRegression

# --- NEW IMPORTS for Regression Evaluation ---
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


logger = logging.getLogger(__name__)


def calculate_expected_fees(quantity_usd: float, fee_tier: str) -> float:
    """
    Calculates the expected trading fees based on the quantity in USD and fee tier.
    Assumes market orders are always Taker orders.

    Args:
        quantity_usd (float): The total value of the order in USD.
        fee_tier (str): The selected fee tier from the UI (e.g., "Regular User LV1").

    Returns:
        float: The calculated fee in USD. Returns 0.0 if quantity is invalid.
    """
    if not isinstance(quantity_usd, (int, float)) or quantity_usd < 0:
        logger.warning(f"Invalid quantity_usd for fee calculation: {quantity_usd}")
        return 0.0
    tier_info = OKX_FEE_RATES.get(fee_tier)
    if tier_info:
        taker_fee_rate = tier_info.get("taker", DEFAULT_TAKER_FEE_RATE)
    else:
        logger.warning(
            f"Fee tier '{fee_tier}' not found. Using default taker fee rate: {DEFAULT_TAKER_FEE_RATE}"
        )
        taker_fee_rate = DEFAULT_TAKER_FEE_RATE
    expected_fee = quantity_usd * taker_fee_rate
    return expected_fee


def calculate_slippage_walk_book(
    target_usd_to_spend: float,
    order_book,  # Type hint can be OrderBookManager if imported
) -> Tuple[Optional[float], Optional[float], float, float]:
    """
    Calculates slippage by simulating a BUY market order walking the ask side of the order book.
    Tries to spend `target_usd_to_spend`.

    Args:
        target_usd_to_spend (float): The amount in USD to try and spend.
        order_book (OrderBookManager): The current order book instance.

    Returns:
        Tuple[Optional[float], Optional[float], float, float]:
            - slippage_percentage (Optional[float]): Slippage in percentage. None if not calculable.
            - average_execution_price (Optional[float]): Average price at which the order was filled. None if not calculable.
            - total_asset_acquired (float): Actual amount of base asset acquired.
            - actual_usd_spent (float): Actual amount of USD spent.
    """
    if target_usd_to_spend <= 0:
        return (
            0.0,
            None,
            0.0,
            0.0,
        )  # No slippage, no price, no asset, no spend for 0 USD

    asks = order_book.asks  # List of [price, quantity]
    bids = order_book.bids

    if not asks or not bids:
        logger.warning(
            "Slippage calc: Asks or Bids are empty. Cannot calculate mid-price or execute."
        )
        return None, None, 0.0, 0.0

    initial_best_ask_price = asks[0][0]
    initial_best_bid_price = bids[0][0]

    if (
        initial_best_ask_price <= initial_best_bid_price
    ):  # Should not happen in a healthy book
        logger.warning(
            f"Slippage calc: Best ask {initial_best_ask_price} <= best bid {initial_best_bid_price}. Book crossed?"
        )
        # Fallback: use best ask as reference if mid-price is problematic
        mid_price_snapshot = initial_best_ask_price
    else:
        mid_price_snapshot = (initial_best_ask_price + initial_best_bid_price) / 2.0

    if mid_price_snapshot <= 0:  # Should not happen
        logger.error(
            "Slippage calc: Mid price is zero or negative, cannot calculate slippage."
        )
        return None, None, 0.0, 0.0

    total_asset_acquired = 0.0
    actual_usd_spent = 0.0
    remaining_usd_to_spend = target_usd_to_spend

    # logger.debug(f"Walking the book for BUY: target_usd_spend={target_usd_to_spend}, mid_snapshot={mid_price_snapshot}")
    # logger.debug(f"Available asks: {asks[:5]}") # Log first 5 ask levels

    for price_level, quantity_at_level in asks:
        if (
            remaining_usd_to_spend <= 1e-9
        ):  # Effectively zero, considering float precision
            break

        cost_to_buy_at_level = price_level * quantity_at_level

        if remaining_usd_to_spend >= cost_to_buy_at_level:
            # Can consume the entire level
            asset_bought = quantity_at_level
            usd_spent_this_level = cost_to_buy_at_level
        else:
            # Consume part of the level
            asset_bought = remaining_usd_to_spend / price_level
            usd_spent_this_level = remaining_usd_to_spend

        total_asset_acquired += asset_bought
        actual_usd_spent += usd_spent_this_level
        remaining_usd_to_spend -= usd_spent_this_level

        # logger.debug(f"Level: P={price_level}, Q={quantity_at_level}. Bought: {asset_bought}, Spent: {usd_spent_this_level}. Remaining USD: {remaining_usd_to_spend}")

    if total_asset_acquired <= 1e-9:  # Effectively zero asset acquired
        logger.warning(
            f"Slippage calc: No asset acquired. Target USD: {target_usd_to_spend}. Actual USD spent: {actual_usd_spent}. This might happen if asks are empty or prices are extremely high."
        )
        # If we spent some USD but got no asset (highly unlikely with valid prices), avg_exec_price is infinite.
        # If we spent no USD (e.g. target_usd_to_spend was too small for any level), treat as no trade.
        return 0.0 if actual_usd_spent == 0 else None, None, 0.0, actual_usd_spent

    average_execution_price = actual_usd_spent / total_asset_acquired

    # For a BUY, positive slippage is an additional cost (paid more than mid-price)
    slippage_value = (
        average_execution_price - mid_price_snapshot
    )  # For a BUY, positive slippage is bad (paid more)
    slippage_percentage = (slippage_value / mid_price_snapshot) * 100.0
    # logger.debug(f"Slippage Result: AvgExecPrice={average_execution_price}, Slippage%={slippage_percentage}, AssetAcquired={total_asset_acquired}, USDSpent={actual_usd_spent}")

    return (
        slippage_percentage,
        average_execution_price,
        total_asset_acquired,
        actual_usd_spent,
    )


def calculate_market_impact_cost(
    order_quantity_usd: float, asset_volatility: float, asset_symbol: str
) -> Optional[float]:
    """
    Calculates a simplified market impact cost.
    Formula: ImpactCost_USD = C * volatility * (OrderSizeUSD / DailyVolumeUSD) * OrderSizeUSD

    Args:
        order_quantity_usd (float): The USD value of the order.
        asset_volatility (float): The asset's volatility (e.g., daily, as decimal 0.02 for 2%).
        asset_symbol (str): The symbol of the asset (e.g., "BTC-USDT-SWAP") to fetch assumed daily volume.

    Returns:
        Optional[float]: Estimated market impact cost in USD. None if inputs are invalid.
    """
    if order_quantity_usd < 0 or asset_volatility < 0:
        logger.warning(
            f"Market Impact: Invalid inputs. Order Qty: {order_quantity_usd}, Vol: {asset_volatility}"
        )
        return None
    if order_quantity_usd == 0:
        return 0.0

    daily_volume_usd = ASSUMED_DAILY_VOLUME_USD.get(asset_symbol)
    if not daily_volume_usd or daily_volume_usd <= 0:
        logger.warning(
            f"Market Impact: Daily volume for {asset_symbol} not found or invalid in config. Using a fallback of 1B."
        )
        daily_volume_usd = 1_000_000_000.0  # Fallback large volume

    # Fraction of daily volume
    volume_fraction = order_quantity_usd / daily_volume_usd

    # Simplified impact cost calculation
    # ImpactCost_USD = C * volatility * (OrderSizeUSD / DailyVolumeUSD) * OrderSizeUSD
    # This is equivalent to: Price_Impact_Percentage_of_Order_Value = C * volatility * volume_fraction
    # And ImpactCost = Price_Impact_Percentage_of_Order_Value * OrderSizeUSD
    market_impact_cost = (
        MARKET_IMPACT_COEFFICIENT
        * asset_volatility
        * volume_fraction
        * order_quantity_usd
    )

    # logger.debug(f"Market Impact for {asset_symbol}: OrderUSD={order_quantity_usd}, Volatility={asset_volatility}, "
    #              f"DailyVolUSD={daily_volume_usd}, VolumeFraction={volume_fraction:.6f}, ImpactCostUSD={market_impact_cost:.4f}")

    return market_impact_cost


# --- CODE for Regression Model ---
class SlippageRegressionModel:
    def __init__(self, min_samples_to_train=50, features_dim=3, test_set_size=0.2):
        self.model = LinearRegression()
        self.is_trained = False
        self.data_X = []  # List of feature lists
        self.data_y = []  # List of target slippage percentages
        self.min_samples_to_train = min_samples_to_train
        self.features_dim = (
            features_dim  # order_size_usd, spread_bps, depth_best_ask_usd
        )
        self.test_set_size = test_set_size  # Proportion of data to use for testing
        self.mse = None
        self.r2 = None
        self.training_samples_count = 0

        logger.info("SlippageRegressionModel initialized.")

    def add_data_point(self, features: List[float], target_slippage_pct: float):
        if len(features) != self.features_dim:
            logger.warning(
                f"Incorrect feature dimension. Expected {self.features_dim}, got {len(features)}"
            )
            return
        self.data_X.append(features)
        self.data_y.append(target_slippage_pct)
        # Optional: Limit data size to prevent memory issues for long runs
        # MAX_DATA_POINTS = 1000
        # if len(self.data_X) > MAX_DATA_POINTS:
        #     self.data_X.pop(0)
        #     self.data_y.pop(0)

    def train(self) -> bool:
        logger.debug(
            f"Train called. Total data points available: {len(self.data_X)}"
        )  # Add this
        if len(self.data_X) < self.min_samples_to_train:
            logger.debug(
                f"Not enough samples to train. Have {len(self.data_X)}, need {self.min_samples_to_train}."
            )  # Add this
            self.is_trained = False
            return False

        try:
            X = np.array(self.data_X)
            y = np.array(self.data_y)

            if X.ndim == 1:
                X = X.reshape(-1, 1)

            # Check for sufficient data for split more carefully
            min_test_samples = (
                1  # We need at least 1 sample in the test set for evaluation
            )
            min_train_samples = 1  # And at least 1 in the train set

            # Ensure there's enough data for a split that results in non-empty train/test sets
            # A common rule is test_size should not lead to test set < 1, and train set should also not be < 1
            if len(X) * self.test_set_size < 1 or len(X) * (1 - self.test_set_size) < 1:
                logger.warning(
                    f"Not enough data for a meaningful train-test split (Total: {len(X)}). Training on all data. Metrics will be on training data."
                )
                X_train, X_test, y_train, y_test = X, X, y, y
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=self.test_set_size, random_state=42
                )

            logger.debug(
                f"Train set size: {len(X_train)}, Test set size: {len(X_test)}"
            )  # Add this

            if (
                len(X_train) < min_train_samples
            ):  # Should be caught by above, but as a safeguard
                logger.warning(
                    "Training set is effectively empty ({len(X_train)} samples). Cannot train model."
                )
                self.is_trained = False
                return False

            self.model.fit(X_train, y_train)
            self.is_trained = True  # Set only after successful fit.
            self.training_samples_count = len(X_train)  # Correctly set here

            # --- Evaluate on test set (if X_test is not same as X_train due to split) ---
            if len(X_test) > 0:
                y_pred_test = self.model.predict(X_test)
                self.mse = mean_squared_error(y_test, y_pred_test)
                self.r2 = r2_score(y_test, y_pred_test)
                eval_set_type = "Test" if X_test is not X_train else "Train (no split)"
                logger.info(
                    f"Slippage model trained with {len(X_train)} samples. {eval_set_type} MSE: {self.mse:.10e}, {eval_set_type} R2: {self.r2:.4f}"
                )
            else:
                # Should not happen if checks above are correct
                logger.warning("Test set was empty, cannot evaluate metrics.")
                self.mse = None
                self.r2 = None

            return True  # Successfully trained
        except Exception as e:
            logger.error(
                f"Error training slippage regression model: {e}", exc_info=True
            )
            self.is_trained = False
            self.mse = None  # Reset metrics on error
            self.r2 = None
            return False  # Training failed

    def predict(self, features: List[float]) -> Optional[float]:
        if not self.is_trained:
            logger.debug("Slippage model not trained yet. Cannot predict.")
            return None
        if len(features) != self.features_dim:
            logger.warning(
                f"Predict: Incorrect feature dimension. Expected {self.features_dim}, got {len(features)}"
            )
            return None

        try:
            prediction = self.model.predict(np.array(features).reshape(1, -1))
            return prediction[0]  # model.predict returns an array
        except Exception as e:
            logger.error(f"Error predicting slippage: {e}", exc_info=True)
            return None

    # --- Getter methods for metrics ---
    def get_metrics(self) -> Dict[str, Optional[float]]:
        return {
            "mse": self.mse,
            "r2": self.r2,
            "training_samples": (
                float(self.training_samples_count) if self.is_trained else 0.0
            ),
        }


if __name__ == "__main__":
    # Mock OrderBookManager for testing
    class MockOrderBookManager:
        def __init__(self, asks, bids):
            self.asks = sorted(
                [(float(p), float(q)) for p, q in asks], key=lambda x: x[0]
            )
            self.bids = sorted(
                [(float(p), float(q)) for p, q in bids],
                key=lambda x: x[0],
                reverse=True,
            )
            self.timestamp = "test_time"
            self.symbol = "TEST/USD"
            self.exchange = "TEST_EX"

        def get_best_ask(self):
            return self.asks[0] if self.asks else None

        def get_best_bid(self):
            return self.bids[0] if self.bids else None

        def get_spread(self):
            ba = self.get_best_ask()
            bb = self.get_best_bid()
            return ba[0] - bb[0] if ba and bb else None

    logging.basicConfig(level=logging.DEBUG)
    print("--- Testing Fee Calculation ---")
    print(
        f"Fee for 100 USD, Regular User LV1: {calculate_expected_fees(100, 'Regular User LV1')} USD"
    )

    book1 = MockOrderBookManager(asks=[(101, 10), (102, 5)], bids=[(100, 10)])
    slp1, avg_p1, ast1, usd1 = calculate_slippage_walk_book(101, book1)
    print(
        f"Test Slippage 1: Slippage={slp1:.4f}%, AvgPrice={avg_p1:.2f}, Asset={ast1:.2f}, SpentUSD={usd1:.2f}"
    )
    impact1 = calculate_market_impact_cost(
        order_quantity_usd=10000, asset_volatility=0.02, asset_symbol="BTC-USDT-SWAP"
    )
    print(f"Test Impact 1 (10k USD order, 2% vol): Impact Cost = {impact1:.4f} USD")
    print("\n--- Testing Slippage Regression Model ---")
    reg_model = SlippageRegressionModel(min_samples_to_train=2)
    reg_model.add_data_point(features=[1000, 1.0, 100000], target_slippage_pct=0.01)
    reg_model.add_data_point(features=[2000, 1.2, 80000], target_slippage_pct=0.03)
    reg_model.add_data_point(
        features=[500, 0.8, 120000], target_slippage_pct=0.005
    )  # More data for split
    reg_model.add_data_point(features=[2500, 1.5, 70000], target_slippage_pct=0.04)
    reg_model.add_data_point(features=[1200, 0.9, 110000], target_slippage_pct=0.015)
    print(f"Training possible: {reg_model.train()}")
    print(f"Model trained: {reg_model.is_trained}")
    if reg_model.is_trained:
        prediction1 = reg_model.predict(features=[1500, 1.1, 90000])
        print(f"Prediction for [1500, 1.1, 90000]: {prediction1}")
        metrics = reg_model.get_metrics()
        print(
            f"Model Metrics: MSE={metrics.get('mse', 'N/A')}, R2={metrics.get('r2', 'N/A')}, Samples={metrics.get('training_samples', 'N/A')}"
        )
