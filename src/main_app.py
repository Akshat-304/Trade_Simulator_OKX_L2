# src/main_app.py
"""
Main application module for the GoQuant Trade Simulator.
Handles the Tkinter UI, orchestrates WebSocket communication,
and triggers financial calculations.
"""

import tkinter as tk
from tkinter import ttk
import threading
import asyncio
import logging
import time
import csv  # NEW import for CSV logging

# --- (Imports from our src modules, including SlippageRegressionModel) ---
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.order_book_manager import OrderBookManager
from src.websocket_handler import connect_and_listen
from src.financial_calculations import (
    calculate_expected_fees,
    calculate_slippage_walk_book,
    calculate_market_impact_cost,
    SlippageRegressionModel,
)

# --- (Logging setup) ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s:%(threadName)s] - %(message)s",
)
logger = logging.getLogger(__name__)

# --- CSV File for logging regression data ---
REGRESSION_DATA_LOG_FILE = "slippage_regression_log.csv"
# Write header if file doesn't exist or is empty
if (
    not os.path.exists(REGRESSION_DATA_LOG_FILE)
    or os.path.getsize(REGRESSION_DATA_LOG_FILE) == 0
):
    with open(REGRESSION_DATA_LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timestamp_data_collected",
                "probe_order_size_usd",
                "market_spread_bps",
                "market_depth_best_ask_usd",
                "true_slippage_pct_walk_the_book",
                "is_model_trained_at_prediction",
                "user_order_size_usd",
                "predicted_slippage_pct_regression",
            ]
        )

# --- CSV File for logging model performance over training ---
MODEL_PERFORMANCE_LOG_FILE = "model_performance_log.csv"
if (
    not os.path.exists(MODEL_PERFORMANCE_LOG_FILE)
    or os.path.getsize(MODEL_PERFORMANCE_LOG_FILE) == 0
):
    with open(MODEL_PERFORMANCE_LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["training_timestamp", "num_training_samples", "test_mse", "test_r2_score"]
        )


class TradingSimulatorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("GoQuant Trade Simulator")
        self.geometry("850x780")
        # Increased height for new latency vars

        # --- (Core components: OrderBookManager, WebSocket thread management) ---

        self.order_book = OrderBookManager()
        self.websocket_thread = None
        self.loop = None
        self.is_connected_with_symbol = False

        # --- Slippage Regression Model ---
        self.slippage_reg_model = SlippageRegressionModel(
            min_samples_to_train=1000, test_set_size=0.2
        )  # Train with more samples (500)
        self.ticks_since_last_train = 0
        self.train_interval_ticks = 200  # Retrain every 200 data updates (generating 200 * num_probes data points)
        self.probe_order_sizes_usd = [
            1000,
            5000,
            10000,
            50000,
            100000,
            500000,
            1e6,
        ]  # USD sizes for probing

        # --- (Intermediate calculation result storage) ---
        self.avg_execution_price = None
        self.actual_asset_traded = None
        self.actual_usd_spent_slippage = None  # USD spent during slippage walk
        self.slippage_percentage_val = None  # Store numeric slippage percentage
        self.fee_cost_usd_val = None  # Store numeric fee cost
        self.market_impact_usd_val = None  # Store numeric market impact cost

        # --- (Tkinter StringVars for UI inputs) ---

        self.exchange_var = tk.StringVar(value="OKX")
        self.spot_asset_var = tk.StringVar(value="BTC-USDT-SWAP")
        self.order_type_var = tk.StringVar(value="Market")
        self.quantity_usd_var = tk.StringVar(value="100")
        self.volatility_var = tk.StringVar(value="0.02")
        self.fee_tier_var = tk.StringVar()

        # --- (Tkinter StringVars for UI outputs) ---

        self.slippage_var = tk.StringVar(value="N/A")  # This will be updated
        self.fees_var = tk.StringVar(value="N/A")
        self.market_impact_var = tk.StringVar(value="N/A")
        self.net_cost_var = tk.StringVar(value="N/A")
        self.maker_taker_proportion_var = tk.StringVar(value="N/A")
        self.calc_latency_var = tk.StringVar(value="N/A")
        self.ws_processing_latency_var = tk.StringVar(value="N/A")  # NEW for L1
        self.ui_update_latency_var = tk.StringVar(
            value="N/A"
        )  # UI StringVar set latency
        self.e2e_latency_var = tk.StringVar(value="N/A")  # End-to-End Latency
        self.timestamp_var = tk.StringVar(value="N/A")
        self.current_best_bid_var = tk.StringVar(value="N/A")
        self.current_best_ask_var = tk.StringVar(value="N/A")
        self.current_spread_var = tk.StringVar(value="N/A")
        # --- NEW StringVars for regression metrics ---
        self.reg_mse_var = tk.StringVar(value="N/A")
        self.reg_r2_var = tk.StringVar(value="N/A")
        self.reg_samples_var = tk.StringVar(value="N/A")

        # --- (UI setup, WebSocket start, Close protocol) ---

        self._setup_ui()
        self._start_websocket_connection()
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _setup_ui(self):
        # ... (UI setup code as before, including traces) ...
        # Configure main window grid
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1, uniform="panel_group")
        self.grid_columnconfigure(1, weight=2, uniform="panel_group")

        # --- Left Panel (Inputs) ---
        self.input_panel = ttk.LabelFrame(self, text="Input Parameters", padding="10")
        self.input_panel.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.input_panel.grid_columnconfigure(0, weight=0)
        self.input_panel.grid_columnconfigure(1, weight=1)

        row_num_input = 0
        ttk.Label(self.input_panel, text="Exchange:").grid(
            row=row_num_input, column=0, sticky="w", pady=3
        )
        ttk.Label(self.input_panel, textvariable=self.exchange_var).grid(
            row=row_num_input, column=1, sticky="ew", pady=3
        )
        row_num_input += 1

        ttk.Label(self.input_panel, text="Spot Asset:").grid(
            row=row_num_input, column=0, sticky="w", pady=3
        )
        ttk.Label(self.input_panel, textvariable=self.spot_asset_var).grid(
            row=row_num_input, column=1, sticky="ew", pady=3
        )
        row_num_input += 1

        ttk.Label(self.input_panel, text="Order Type:").grid(
            row=row_num_input, column=0, sticky="w", pady=3
        )
        ttk.Label(self.input_panel, textvariable=self.order_type_var).grid(
            row=row_num_input, column=1, sticky="ew", pady=3
        )
        row_num_input += 1

        ttk.Label(self.input_panel, text="Quantity (USD):").grid(
            row=row_num_input, column=0, sticky="w", pady=3
        )
        qty_entry = ttk.Entry(self.input_panel, textvariable=self.quantity_usd_var)
        qty_entry.grid(row=row_num_input, column=1, sticky="ew", pady=3)
        self.quantity_usd_var.trace_add("write", self._trigger_recalculation)
        row_num_input += 1

        ttk.Label(self.input_panel, text="Volatility (e.g., 0.02):").grid(
            row=row_num_input, column=0, sticky="w", pady=3
        )
        vol_entry = ttk.Entry(self.input_panel, textvariable=self.volatility_var)
        vol_entry.grid(row=row_num_input, column=1, sticky="ew", pady=3)
        self.volatility_var.trace_add("write", self._trigger_recalculation)
        row_num_input += 1

        ttk.Label(self.input_panel, text="Fee Tier:").grid(
            row=row_num_input, column=0, sticky="w", pady=3
        )
        fee_tier_options = [
            "Regular User LV1",
            "Regular User LV2",
            "Regular User LV3",
            "VIP 1",
            "VIP 2",
            "VIP 3",
            "VIP 4",
            "VIP 5",
            "VIP 6",
            "VIP 7",
            "VIP 8",
            "Custom",
        ]
        fee_tier_combobox = ttk.Combobox(
            self.input_panel,
            textvariable=self.fee_tier_var,
            values=fee_tier_options,
            state="readonly",
        )
        fee_tier_combobox.grid(row=row_num_input, column=1, sticky="ew", pady=3)
        default_fee_tier = "Regular User LV1"
        fee_tier_combobox.set(default_fee_tier)
        self.fee_tier_var.set(default_fee_tier)
        self.fee_tier_var.trace_add("write", self._trigger_recalculation)
        row_num_input += 1

        self.input_panel.grid_rowconfigure(row_num_input, weight=1)

        # --- Right Panel (Outputs) ---
        self.output_panel = ttk.LabelFrame(
            self, text="Processed Outputs & Market Data", padding="10"
        )
        self.output_panel.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        # ... (rest of output panel setup as before) ...
        self.output_panel.grid_columnconfigure(0, weight=0)
        self.output_panel.grid_columnconfigure(1, weight=1)

        row_num_output = 0

        ttk.Label(
            self.output_panel, text="Market Data:", font=("Arial", 12, "bold")
        ).grid(row=row_num_output, column=0, columnspan=2, sticky="w", pady=(0, 5))
        row_num_output += 1

        ttk.Label(self.output_panel, text="Timestamp:").grid(
            row=row_num_output, column=0, sticky="w", pady=2
        )
        ttk.Label(self.output_panel, textvariable=self.timestamp_var).grid(
            row=row_num_output, column=1, sticky="ew", pady=2
        )
        row_num_output += 1

        ttk.Label(self.output_panel, text="Best Bid:").grid(
            row=row_num_output, column=0, sticky="w", pady=2
        )
        ttk.Label(self.output_panel, textvariable=self.current_best_bid_var).grid(
            row=row_num_output, column=1, sticky="ew", pady=2
        )
        row_num_output += 1

        ttk.Label(self.output_panel, text="Best Ask:").grid(
            row=row_num_output, column=0, sticky="w", pady=2
        )
        ttk.Label(self.output_panel, textvariable=self.current_best_ask_var).grid(
            row=row_num_output, column=1, sticky="ew", pady=2
        )
        row_num_output += 1

        ttk.Label(self.output_panel, text="Spread:").grid(
            row=row_num_output, column=0, sticky="w", pady=2
        )
        ttk.Label(self.output_panel, textvariable=self.current_spread_var).grid(
            row=row_num_output, column=1, sticky="ew", pady=2
        )
        row_num_output += 1

        ttk.Separator(self.output_panel, orient="horizontal").grid(
            row=row_num_output, column=0, columnspan=2, sticky="ew", pady=5
        )  # Reduced pady
        row_num_output += 1

        ttk.Label(
            self.output_panel,
            text="Transaction Cost Estimates:",
            font=("Arial", 12, "bold"),
        ).grid(row=row_num_output, column=0, columnspan=2, sticky="w", pady=(5, 5))
        row_num_output += 1

        ttk.Label(self.output_panel, text="Expected Slippage (%):").grid(
            row=row_num_output, column=0, sticky="w", pady=2
        )  # Added (%)
        ttk.Label(self.output_panel, textvariable=self.slippage_var).grid(
            row=row_num_output, column=1, sticky="ew", pady=2
        )  # This will now show regression model prediction
        row_num_output += 1

        ttk.Label(self.output_panel, text="Expected Fees (USD):").grid(
            row=row_num_output, column=0, sticky="w", pady=2
        )
        ttk.Label(self.output_panel, textvariable=self.fees_var).grid(
            row=row_num_output, column=1, sticky="ew", pady=2
        )
        row_num_output += 1

        ttk.Label(self.output_panel, text="Expected Market Impact:").grid(
            row=row_num_output, column=0, sticky="w", pady=2
        )
        ttk.Label(self.output_panel, textvariable=self.market_impact_var).grid(
            row=row_num_output, column=1, sticky="ew", pady=2
        )
        row_num_output += 1

        ttk.Label(self.output_panel, text="Net Cost (USD):").grid(
            row=row_num_output, column=0, sticky="w", pady=2
        )
        ttk.Label(self.output_panel, textvariable=self.net_cost_var).grid(
            row=row_num_output, column=1, sticky="ew", pady=2
        )
        row_num_output += 1

        ttk.Separator(self.output_panel, orient="horizontal").grid(
            row=row_num_output, column=0, columnspan=2, sticky="ew", pady=5
        )
        row_num_output += 1

        ttk.Label(
            self.output_panel, text="Performance Metrics:", font=("Arial", 12, "bold")
        ).grid(
            row=row_num_output, column=0, columnspan=2, sticky="w", pady=(5, 5)
        )  # Renamed from Other Metrics
        row_num_output += 1

        ttk.Label(self.output_panel, text="Maker/Taker Proportion:").grid(
            row=row_num_output, column=0, sticky="w", pady=2
        )
        ttk.Label(self.output_panel, textvariable=self.maker_taker_proportion_var).grid(
            row=row_num_output, column=1, sticky="ew", pady=2
        )
        row_num_output += 1

        # --- Latency Labels ---
        ttk.Label(self.output_panel, text="WS Proc. Latency (ms):").grid(
            row=row_num_output, column=0, sticky="w", pady=2
        )
        ttk.Label(self.output_panel, textvariable=self.ws_processing_latency_var).grid(
            row=row_num_output, column=1, sticky="ew", pady=2
        )
        row_num_output += 1
        ttk.Label(self.output_panel, text="Calc. Latency (ms):").grid(
            row=row_num_output, column=0, sticky="w", pady=2
        )
        ttk.Label(self.output_panel, textvariable=self.calc_latency_var).grid(
            row=row_num_output, column=1, sticky="ew", pady=2
        )
        row_num_output += 1
        ttk.Label(self.output_panel, text="UI Update Latency (ms):").grid(
            row=row_num_output, column=0, sticky="w", pady=2
        )
        ttk.Label(self.output_panel, textvariable=self.ui_update_latency_var).grid(
            row=row_num_output, column=1, sticky="ew", pady=2
        )
        row_num_output += 1  # NEW UI latency label
        ttk.Label(self.output_panel, text="End-to-End Latency (ms):").grid(
            row=row_num_output, column=0, sticky="w", pady=2
        )
        ttk.Label(self.output_panel, textvariable=self.e2e_latency_var).grid(
            row=row_num_output, column=1, sticky="ew", pady=2
        )
        row_num_output += 1  # NEW E2E latency label

        # --- Regression Model Metrics UI ---
        ttk.Separator(self.output_panel, orient="horizontal").grid(
            row=row_num_output, column=0, columnspan=2, sticky="ew", pady=5
        )
        row_num_output += 1  # Reduced pady
        ttk.Label(
            self.output_panel, text="Slippage Model Perf.:", font=("Arial", 12, "bold")
        ).grid(row=row_num_output, column=0, columnspan=2, sticky="w", pady=(5, 5))
        row_num_output += 1
        ttk.Label(self.output_panel, text="Train Samples:").grid(
            row=row_num_output, column=0, sticky="w", pady=2
        )
        ttk.Label(self.output_panel, textvariable=self.reg_samples_var).grid(
            row=row_num_output, column=1, sticky="ew", pady=2
        )
        row_num_output += 1
        ttk.Label(self.output_panel, text="Test MSE:").grid(
            row=row_num_output, column=0, sticky="w", pady=2
        )
        ttk.Label(self.output_panel, textvariable=self.reg_mse_var).grid(
            row=row_num_output, column=1, sticky="ew", pady=2
        )
        row_num_output += 1
        ttk.Label(self.output_panel, text="Test R2 Score:").grid(
            row=row_num_output, column=0, sticky="w", pady=2
        )
        ttk.Label(self.output_panel, textvariable=self.reg_r2_var).grid(
            row=row_num_output, column=1, sticky="ew", pady=2
        )
        row_num_output += 1

        self.output_panel.grid_rowconfigure(row_num_output, weight=1)

        # --- Status Bar ---
        self.status_bar_text = tk.StringVar(value="Status: Initializing...")
        self.status_bar = ttk.Label(
            self, textvariable=self.status_bar_text, relief=tk.SUNKEN, anchor=tk.W
        )
        self.status_bar.grid(row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=5)

    def _trigger_recalculation(self, *args):
        self.after(50, self._recalculate_all_outputs)

    def _recalculate_all_outputs(self):
        # --- Start Latency Measurement ---
        calc_start_time = time.perf_counter()  # Start of L2 latency measurement

        # Reset numeric values at the start of each calculation attempt
        self.slippage_percentage_val = None
        self.fee_cost_usd_val = None
        self.market_impact_usd_val = None
        self.avg_execution_price = None
        self.actual_asset_traded = None
        self.actual_usd_spent_slippage = None  # Important to reset this

        current_calc_latency = "N/A"  # Default latency display
        predicted_slippage_pct_for_log = None  # For CSV logging

        try:
            # 1. Read Input: Quantity USD
            try:
                quantity_usd_val = float(self.quantity_usd_var.get())
                if quantity_usd_val < 0:  # Allow 0 for no trade scenario
                    raise ValueError
            except ValueError:
                for var in [
                    self.fees_var,
                    self.slippage_var,
                    self.market_impact_var,
                    self.net_cost_var,
                ]:
                    var.set("Invalid Qty")
                self.avg_execution_price = None  #
                self.actual_asset_traded = None  #
                self.actual_usd_spent_slippage = None  #
                return

            # 2. Read Input: Fee Tier
            fee_tier_val = self.fee_tier_var.get()

            # 3. Read Input: Volatility
            try:
                volatility_val = float(self.volatility_var.get())
                if volatility_val < 0:
                    raise ValueError
            except ValueError:
                self.market_impact_var.set("Invalid Vol")
                self.net_cost_var.set("Invalid Vol")
                return

            # 4. Read Input: Asset Symbol (from fixed var for now)
            asset_symbol_val = self.spot_asset_var.get()

            #     # --- Calculate Slippage (Walk the Book) ---
            #     # Requires live order book data, so self.order_book must be up-to-date
            #     slippage_cost_usd = 0.0 # Default to 0 if not calculable
            #     if not self.order_book.asks or not self.order_book.bids: # Check if book has data
            #         self.slippage_var.set("No book data")
            #         self.avg_execution_price = None #
            #         self.actual_asset_traded = None #
            #         self.actual_usd_spent_slippage = None #
            #     else:
            #         slp_pct, avg_exec_p, asset_acq, usd_spent = calculate_slippage_walk_book(
            #             quantity_usd_val, self.order_book
            #         )
            #         # Store these values for other calculations (e.g. market impact)
            #         self.avg_execution_price = avg_exec_p
            #         self.actual_asset_traded = asset_acq
            #         self.actual_usd_spent_slippage = usd_spent

            #         if slp_pct is not None:
            #             self.slippage_percentage_val = slp_pct # Store numeric value
            #             self.slippage_var.set(f"{slp_pct:.4f}%")
            #             # Calculate slippage cost in USD.
            #             # Slippage cost is the difference between what you actually paid (usd_spent)
            #             # and what you would have paid at the mid-price for the asset_acquired.
            #             # mid_price_snapshot used inside calculate_slippage_walk_book for asset_acquired:
            #             if self.order_book.get_best_ask() and self.order_book.get_best_bid():
            #                 mid_price = (self.order_book.get_best_ask()[0] + self.order_book.get_best_bid()[0]) / 2
            #                 if asset_acq > 0 : # if any asset was acquired
            #                     slippage_cost_usd = usd_spent - asset_acq * mid_price
            #                 # If slp_pct is positive (paid more), slippage_cost_usd will be positive.
            #             # Alternative simpler slippage cost based on target USD, but less accurate if fill is partial:
            #             # slippage_cost_usd = (slp_pct / 100.0) * quantity_usd_val

            #         else:
            #             if quantity_usd_val > 0 and asset_acq == 0 : # Tried to buy but got nothing
            #                  self.slippage_var.set("Depth Exceeded?")
            #             elif quantity_usd_val == 0:
            #                  self.slippage_var.set("0.0000%") # No slippage for no trade
            #             else: # Other error cases from slippage function
            #                  self.slippage_var.set("Error/No Trade")
            #         logger.debug(f"Slippage: {slp_pct}%, AvgPrice: {avg_exec_p}, Asset: {asset_acq}, Spent: {usd_spent}")

            #     # --- Calculate Expected Fees ---
            #     # Fees should ideally be based on the actual USD spent if slippage is significant
            #     # or if the order couldn't be fully filled for target_usd_val.
            #     # For now, let's use target quantity_usd_val for simplicity as per problem statement.
            #     # Or, use self.actual_usd_spent_slippage if available.
            #     # Let's use quantity_usd_val as it's the "target".
            #     calculated_fees = calculate_expected_fees(quantity_usd_val, fee_tier_val)
            #     self.fees_var.set(f"{calculated_fees:.4f}")
            #     self.fee_cost_usd_val = calculated_fees

            #     # --- Calculate Market Impact Cost ---
            #     # Use actual USD spent from slippage calculation if available and valid, else target quantity
            #     # For simplicity, assignment implies using the input "Quantity (~100 USD equivalent)"
            #     # Let's use quantity_usd_val (target order size) for market impact calculation as well.
            #     market_impact_usd = calculate_market_impact_cost(quantity_usd_val, volatility_val, asset_symbol_val)
            #     if market_impact_usd is not None:
            #         self.market_impact_var.set(f"{market_impact_usd:.4f}")
            #     else:
            #         self.market_impact_var.set("Error")
            #     self.market_impact_usd_val = market_impact_usd

            #     # --- Calculate Net Cost ---
            #     if self.fee_cost_usd_val is not None and \
            #        self.market_impact_usd_val is not None and \
            #        self.slippage_percentage_val is not None: # Check if all components are valid

            #         # If quantity_usd_val is 0, all costs should be 0
            #         if quantity_usd_val == 0:
            #             net_total_cost_usd = 0.0
            #             slippage_cost_usd = 0.0 # Ensure this is zero for zero quantity
            #         else:
            #             net_total_cost_usd = slippage_cost_usd + self.fee_cost_usd_val + self.market_impact_usd_val

            #         self.net_cost_var.set(f"{net_total_cost_usd:.4f}")
            #     else:
            #         self.net_cost_var.set("Error")

            #     # --- Maker/Taker Proportion ---
            #     if quantity_usd_val == 0:
            #          self.maker_taker_proportion_var.set("N/A (No Trade)")
            #     else:
            #          self.maker_taker_proportion_var.set("100% Taker")

            #     # --- End Latency Measurement & Update UI ---
            #     calc_end_time = time.perf_counter()
            #     processing_time_ms = (calc_end_time - calc_start_time) * 1000
            #     current_calc_latency = f"{processing_time_ms:.3f}" # Store as string for UI
            #     logger.debug(f"Internal processing latency: {processing_time_ms:.3f} ms")

            # except Exception as e:
            #     logger.error(f"Error during recalculation: {e}", exc_info=True)
            #     self.fees_var.set("Error")
            #     self.slippage_var.set("Error")
            #     self.market_impact_var.set("Error")
            #     self.net_cost_var.set("Error")
            #     self.maker_taker_proportion_var.set("Error")
            #     self.slippage_var.set("Error")
            # finally:
            #     # This ensures latency is updated even if an error occurred mid-calculation,
            #     # showing the time taken up to the error point or full calculation.
            #     self.calc_latency_var.set(current_calc_latency)

            # --- Calculate Slippage (INTEGRATING REGRESSION) ---
            slippage_cost_usd = 0.0

            # A. Using Regression Model (Primary for UI display)
            if (
                self.slippage_reg_model.is_trained
                and self.order_book.get_best_ask()
                and self.order_book.get_best_bid()
            ):
                best_ask_price, best_ask_qty = self.order_book.get_best_ask()
                best_bid_price, _ = self.order_book.get_best_bid()

                spread_val = best_ask_price - best_bid_price
                mid_price_for_bps = (best_ask_price + best_bid_price) / 2
                spread_bps = (
                    (spread_val / mid_price_for_bps) * 10000
                    if mid_price_for_bps > 0
                    else 0
                )

                depth_best_ask_usd = best_ask_price * best_ask_qty

                features_for_prediction = [
                    quantity_usd_val,
                    spread_bps,
                    depth_best_ask_usd,
                ]
                predicted_slippage_pct = self.slippage_reg_model.predict(
                    features_for_prediction
                )
                predicted_slippage_pct_for_log = predicted_slippage_pct  # For CSV

                if predicted_slippage_pct is not None:
                    # --- Cap negative slippage prediction for BUY orders at 0 ---
                    effective_predicted_slippage_pct = max(0.0, predicted_slippage_pct)
                    self.slippage_var.set(
                        f"{effective_predicted_slippage_pct:.4f}% (Reg)"
                    )
                    self.slippage_percentage_val = (
                        effective_predicted_slippage_pct  # Store this for net cost
                    )
                    # For net cost, we need slippage_cost_usd based on this percentage
                    slippage_cost_usd = (
                        effective_predicted_slippage_pct / 100.0
                    ) * quantity_usd_val
                else:
                    self.slippage_var.set("Reg Pred Err")
            elif quantity_usd_val == 0:
                self.slippage_var.set("0.0000%")
                self.slippage_percentage_val = 0.0
            else:  # Fallback if model not trained or book data missing for features
                self.slippage_var.set("N/A (Model Pending)")
                # Could fall back to walk-the-book for self.slippage_percentage_val if needed for net cost
                # For now, net cost will show error if regression isn't ready.

            # B. Walk-the-book (Internal reference, or fallback if strict)
            # We still need its outputs (actual_usd_spent, asset_acquired) for accurate fee/impact on executed value
            if self.order_book.asks and self.order_book.bids:
                (
                    _,
                    self.avg_execution_price,
                    self.actual_asset_traded,
                    self.actual_usd_spent_slippage,
                ) = calculate_slippage_walk_book(quantity_usd_val, self.order_book)

            # --- Calculate Expected Fees ---
            # Use actual_usd_spent_slippage if available and valid for more accuracy, else target quantity_usd_val
            fee_calc_base_usd = (
                self.actual_usd_spent_slippage
                if self.actual_usd_spent_slippage is not None
                and self.actual_usd_spent_slippage > 0
                else quantity_usd_val
            )
            self.fee_cost_usd_val = calculate_expected_fees(
                fee_calc_base_usd, fee_tier_val
            )
            self.fees_var.set(f"{self.fee_cost_usd_val:.4f}")

            # --- Calculate Market Impact Cost ---
            # Use actual_usd_spent_slippage if available and valid, else target quantity_usd_val
            impact_calc_base_usd = (
                self.actual_usd_spent_slippage
                if self.actual_usd_spent_slippage is not None
                and self.actual_usd_spent_slippage > 0
                else quantity_usd_val
            )
            self.market_impact_usd_val = calculate_market_impact_cost(
                impact_calc_base_usd, volatility_val, asset_symbol_val
            )
            if self.market_impact_usd_val is not None:
                self.market_impact_var.set(f"{self.market_impact_usd_val:.4f}")
            else:
                self.market_impact_var.set("Error")

            # --- Calculate Net Cost ---
            # Uses slippage_cost_usd derived from the regression model's percentage
            if (
                self.fee_cost_usd_val is not None
                and self.market_impact_usd_val is not None
                and self.slippage_percentage_val is not None
            ):
                net_total_cost_usd = (
                    (
                        slippage_cost_usd
                        + self.fee_cost_usd_val
                        + self.market_impact_usd_val
                    )
                    if quantity_usd_val > 0
                    else 0.0
                )
                self.net_cost_var.set(f"{net_total_cost_usd:.4f}")
            else:
                self.net_cost_var.set(
                    "Waiting..."
                )  # More informative than "Error" if components are pending

            # --- Maker/Taker Proportion ---
            self.maker_taker_proportion_var.set(
                "N/A (No Trade)" if quantity_usd_val == 0 else "100% Taker"
            )

            # --- Update Regression Metrics UI ---
            metrics = self.slippage_reg_model.get_metrics()
            self.reg_samples_var.set(f"{metrics.get('training_samples', 0.0):.0f}")
            # --- MSE Display format ---
            self.reg_mse_var.set(
                f"{metrics.get('mse', float('nan')):.3e}"
                if metrics.get("mse") is not None
                else "N/A"
            )  # e.g., 1.234e-08
            self.reg_r2_var.set(
                f"{metrics.get('r2', float('nan')):.4f}"
                if metrics.get("r2") is not None
                else "N/A"
            )

            calc_end_time = time.perf_counter()
            processing_time_ms = (calc_end_time - calc_start_time) * 1000
            current_calc_latency = f"{processing_time_ms:.3f}"  # This is L2.

            # --- Log data for user's current prediction to CSV ---
            # This logs the features for the user's actual order and the model's prediction for it.
            # It does NOT log the probe data here, that's implicit in the model's training data.
            if (
                self.slippage_reg_model.is_trained and self.order_book.get_best_ask()
            ):  # Check if features are available
                # Re-extract features for user's current order size for logging consistency
                best_ask_price_log, best_ask_qty_log = self.order_book.get_best_ask()
                best_bid_price_log, _ = (
                    self.order_book.get_best_bid()
                    if self.order_book.get_best_bid()
                    else (0, 0)
                )
                spread_val_log = best_ask_price_log - best_bid_price_log
                mid_price_for_bps_log = (best_ask_price_log + best_bid_price_log) / 2
                spread_bps_log = (
                    (spread_val_log / mid_price_for_bps_log) * 10000
                    if mid_price_for_bps_log > 0
                    else 0
                )
                depth_best_ask_usd_log = best_ask_price_log * best_ask_qty_log

                with open(REGRESSION_DATA_LOG_FILE, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            time.strftime(
                                "%Y-%m-%dT%H:%M:%S"
                            ),  # timestamp_data_collected (approximate)
                            None,  # probe_order_size_usd (N/A for user prediction row)
                            None,  # market_spread_bps (N/A for user prediction row - features for probes are not re-logged here)
                            None,  # market_depth_best_ask_usd (N/A for user prediction row)
                            None,  # true_slippage_pct_walk_the_book (N/A for user prediction row)
                            self.slippage_reg_model.is_trained,
                            quantity_usd_val,  # user_order_size_usd
                            predicted_slippage_pct_for_log,  # predicted_slippage_pct_regression for user's order
                        ]
                    )

        except Exception as e:
            logger.error(
                f"Error during recalculation: {e}", exc_info=True
            )  # error handling
            for var in [
                self.fees_var,
                self.slippage_var,
                self.market_impact_var,
                self.net_cost_var,
                self.maker_taker_proportion_var,
                self.reg_mse_var,
                self.reg_r2_var,
                self.reg_samples_var,
            ]:
                var.set("Error")
        finally:
            self.calc_latency_var.set(
                current_calc_latency
            )  # Update calculation latency UI

    # --- _update_ui_from_websocket method ---
    def _update_ui_from_websocket(self, book_manager, status_and_timestamps):
        # Unpack status and timestamps (ws_msg_arrival_time is from websockets_handler)
        if isinstance(status_and_timestamps, tuple):
            status, ws_msg_arrival_time = status_and_timestamps
            # Set WS processing latency (L1) as soon as received
            if ws_msg_arrival_time is not None:
                ws_proc_latency_ms = (time.perf_counter() - ws_msg_arrival_time) * 1000
                self.ws_processing_latency_var.set(f"{ws_proc_latency_ms:.3f}")
            else:
                self.ws_processing_latency_var.set("N/A")
        else:  # Fallback for older calls or if latency not passed
            status = status_and_timestamps
            self.ws_processing_latency_var.set("N/A")

        if status == "connected":
            self.status_bar_text.set(
                f"Status: Connected to WebSocket. Waiting for data..."
            )
            logger.info("UI updated: Connected")
            self.after(100, self._trigger_recalculation)
        elif status == "data_update":
            # --- START: UI Update Latency (L3) Measurement ---
            ui_update_start_time = time.perf_counter()

            if not self.is_connected_with_symbol and book_manager.symbol:
                self.status_bar_text.set(
                    f"Status: Connected to WebSocket ({book_manager.symbol})"
                )
                self.is_connected_with_symbol = True
            self.timestamp_var.set(book_manager.timestamp)
            best_bid = book_manager.get_best_bid()
            self.current_best_bid_var.set(
                f"{best_bid[0]:.2f} ({best_bid[1]:.2f})" if best_bid else "N/A"
            )
            best_ask = book_manager.get_best_ask()
            self.current_best_ask_var.set(
                f"{best_ask[0]:.2f} ({best_ask[1]:.2f})" if best_ask else "N/A"
            )
            spread_val = book_manager.get_spread()
            self.current_spread_var.set(
                f"{spread_val:.2f}" if spread_val is not None else "N/A"
            )

            # --- Data Collection for Regression & Periodic Training ---
            if best_ask and best_bid:  # Ensure we have basic book data

                # --- RIGOROUS CHECK FOR CROSSED BOOK ---
                # Recalculate best_ask_price and best_bid_price directly from book_manager for this check
                # to avoid using potentially stale `best_ask` or `best_bid` from above if updates are rapid.
                current_ba_tuple = book_manager.get_best_ask()
                current_bb_tuple = book_manager.get_best_bid()

                if (
                    current_ba_tuple
                    and current_bb_tuple
                    and current_ba_tuple[0] > current_bb_tuple[0]
                ):
                    # Book is NOT crossed, proceed with probe data generation
                    current_best_ask_price, current_best_ask_qty = current_ba_tuple
                    current_best_bid_price, _ = current_bb_tuple

                    current_spread_value = (
                        current_best_ask_price - current_best_bid_price
                    )
                    mid_price_for_bps_calc = (
                        current_best_ask_price + current_best_bid_price
                    ) / 2
                    current_spread_bps = (
                        (current_spread_value / mid_price_for_bps_calc) * 10000
                        if mid_price_for_bps_calc > 0
                        else 0
                    )
                    current_depth_best_ask_usd = (
                        current_best_ask_price * current_best_ask_qty
                    )

                    # Additional check: Ensure spread_bps is not negative due to float issues if very close
                    if current_spread_bps < 0:
                        logger.warning(
                            f"Calculated negative spread_bps ({current_spread_bps:.4f}) even after checking ask > bid. Ask: {current_best_ask_price}, Bid: {current_best_bid_price}. Skipping probes."
                        )
                    else:
                        for probe_size_usd in self.probe_order_sizes_usd:
                            probe_slippage_pct, _, _, _ = calculate_slippage_walk_book(
                                probe_size_usd, book_manager
                            )  # book_manager is up-to-date
                            if probe_slippage_pct is not None:
                                features = [
                                    float(probe_size_usd),
                                    float(current_spread_bps),
                                    float(current_depth_best_ask_usd),
                                ]
                                self.slippage_reg_model.add_data_point(
                                    features, probe_slippage_pct
                                )

                                # Log probe data to CSV
                                with open(
                                    REGRESSION_DATA_LOG_FILE, "a", newline=""
                                ) as f:
                                    writer = csv.writer(f)
                                    writer.writerow(
                                        [
                                            time.strftime("%Y-%m-%dT%H:%M:%S"),
                                            probe_size_usd,
                                            current_spread_bps,
                                            current_depth_best_ask_usd,
                                            probe_slippage_pct,
                                            None,
                                            None,
                                            None,
                                        ]
                                    )

                        self.ticks_since_last_train += 1
                        total_data_points = len(
                            self.slippage_reg_model.data_X
                        )  # Get current total data points

                        # Check conditions for training
                        # Condition 1: Model is not yet trained AND we have enough samples for the first train
                        ready_for_initial_train = (
                            not self.slippage_reg_model.is_trained
                        ) and (
                            total_data_points
                            >= self.slippage_reg_model.min_samples_to_train
                        )

                        # Condition 2: Model is already trained AND enough new ticks have passed for a retrain
                        # Retrain interval should be based on number of data points generated, not just ticks if num_probes varies
                        # For simplicity, let's stick to ticks_since_last_train for now.
                        # The number of actual data points added since last train is ticks_since_last_train * len(self.probe_order_sizes_usd)
                        samples_since_last_train = self.ticks_since_last_train * (
                            len(self.probe_order_sizes_usd)
                            if len(self.probe_order_sizes_usd) > 0
                            else 1
                        )

                        # Let's use self.train_interval_ticks directly as the number of *WebSocket updates* between retrains
                        time_for_retrain = self.slippage_reg_model.is_trained and (
                            self.ticks_since_last_train >= self.train_interval_ticks
                        )

                        if ready_for_initial_train or time_for_retrain:
                            logger.info(
                                f"Attempting to train model. Initial: {ready_for_initial_train}, Retrain: {time_for_retrain}, Total data: {total_data_points}"
                            )
                            if self.slippage_reg_model.train():
                                logger.info(
                                    f"Model (re)trained successfully with {self.slippage_reg_model.training_samples_count} samples. Updating metrics."
                                )
                                # --- Log model performance to separate CSV ---
                                metrics = self.slippage_reg_model.get_metrics()
                                with open(
                                    MODEL_PERFORMANCE_LOG_FILE, "a", newline=""
                                ) as f:
                                    writer = csv.writer(f)
                                    writer.writerow(
                                        [
                                            time.strftime("%Y-%m-%dT%H:%M:%S"),
                                            metrics.get("training_samples", 0),
                                            metrics.get("mse", float("nan")),
                                            metrics.get("r2", float("nan")),
                                        ]
                                    )
                            else:
                                logger.warning(
                                    f"Model training attempt failed or not enough data for split. Total data points: {total_data_points}"
                                )
                            self.ticks_since_last_train = (
                                0  # Reset counter after attempting to train
                            )
                else:
                    crossed_ask = current_ba_tuple[0] if current_ba_tuple else "N/A"
                    crossed_bid = current_bb_tuple[0] if current_bb_tuple else "N/A"
                    logger.warning(
                        f"Book crossed or incomplete: Best Ask {crossed_ask} / Best Bid {crossed_bid}. Skipping probe data generation for this tick."
                    )
            # This call will update self.calc_latency_var (L2)
            self._recalculate_all_outputs()  # This will use the latest model state

            # --- END: UI Update Latency (L3) Measurement ---
            ui_update_end_time = time.perf_counter()
            ui_update_latency_ms = (ui_update_end_time - ui_update_start_time) * 1000
            self.ui_update_latency_var.set(f"{ui_update_latency_ms:.3f}")

            # --- Calculate End-to-End Latency (L4) ---
            if ws_msg_arrival_time is not None:
                e2e_latency_ms = (ui_update_end_time - ws_msg_arrival_time) * 1000
                self.e2e_latency_var.set(f"{e2e_latency_ms:.3f}")
            else:
                self.e2e_latency_var.set("N/A")

        elif status == "disconnected_error":
            self.status_bar_text.set("Status: WebSocket Disconnected (Error).")
            self.is_connected_with_symbol = False
            for var in [
                self.timestamp_var,
                self.current_best_bid_var,
                self.current_best_ask_var,
                self.current_spread_var,
                self.fees_var,
                self.slippage_var,
                self.market_impact_var,
                self.net_cost_var,
                self.maker_taker_proportion_var,
                self.calc_latency_var,
                self.ws_processing_latency_var,
                self.ui_update_latency_var,
                self.e2e_latency_var,
                self.reg_mse_var,
                self.reg_r2_var,
                self.reg_samples_var,
            ]:
                var.set("N/A")
            logger.warning("UI updated: Disconnected (Error)")
        elif status == "disconnected_clean":
            self.status_bar_text.set("Status: WebSocket Disconnected.")
            self.is_connected_with_symbol = False
            for var in [
                self.timestamp_var,
                self.current_best_bid_var,
                self.current_best_ask_var,
                self.current_spread_var,
                self.fees_var,
                self.slippage_var,
                self.market_impact_var,
                self.net_cost_var,
                self.maker_taker_proportion_var,
                self.calc_latency_var,
                self.ws_processing_latency_var,
                self.ui_update_latency_var,
                self.e2e_latency_var,
                self.reg_mse_var,
                self.reg_r2_var,
                self.reg_samples_var,
            ]:
                var.set("N/A")
            logger.info("UI updated: Disconnected (Cleanly)")

    # --- WebSocket and Shutdown methods remain the same ---
    def _start_websocket_connection(self):
        # ...
        self.status_bar_text.set("Status: Connecting to WebSocket...")
        self.is_connected_with_symbol = False
        self.loop = asyncio.new_event_loop()

        self.websocket_thread = threading.Thread(
            target=self._run_websocket_loop, args=(self.loop,), daemon=True
        )
        self.websocket_thread.start()
        logger.info("WebSocket thread started.")

    def _run_websocket_loop(self, loop):
        # ...
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(
                connect_and_listen(self.order_book, self.schedule_ui_update)
            )
        except Exception as e:
            logger.error(f"Critical exception in WebSocket run_until_complete: {e}")
        finally:
            if loop.is_running():
                loop.call_soon_threadsafe(loop.stop)
            logger.info("Asyncio event loop tasks finished in WebSocket thread.")

    # --- schedule_ui_update  to pass L1 latency) ---
    def schedule_ui_update(
        self, book_manager, status, ws_processing_latency_ms=None
    ):  # Add new arg
        # Use self.after to ensure UI updates happen in the main Tkinter thread.
        # Pass status and latency as a tuple.
        self.after(
            0,
            self._update_ui_from_websocket,
            book_manager,
            (status, ws_processing_latency_ms),
        )

    def _on_closing(self):
        # ...
        logger.info("Close button clicked. Initiating shutdown sequence...")
        if self.loop and self.loop.is_running():
            logger.info("Attempting to cancel all tasks in asyncio event loop...")
            # --- NEW LINES TO CANCEL TASKS ---
            for task in asyncio.all_tasks(self.loop):
                if task is not asyncio.current_task(
                    self.loop
                ):  # Don't cancel self if called from within loop
                    task.cancel()
            # --- END OF NEW LINES ---
            logger.info("Attempting to stop asyncio event loop...")
            self.loop.call_soon_threadsafe(self.loop.stop)
        else:
            logger.info(
                "Asyncio event loop was not running or not initialized at close."
            )

        if self.websocket_thread and self.websocket_thread.is_alive():
            logger.info("Waiting for WebSocket thread to join...")
            self.websocket_thread.join(timeout=5.0)
            if self.websocket_thread.is_alive():
                logger.warning("WebSocket thread did not join in time. Forcing exit.")
            else:
                logger.info("WebSocket thread joined successfully.")
        else:
            logger.info("WebSocket thread was not alive or not initialized at close.")

        logger.info("Destroying Tkinter window.")
        self.destroy()

    def run(self):
        # ...
        self.status_bar_text.set("Status: UI Ready. Initializing WebSocket...")
        self.after(100, self._trigger_recalculation)
        self.mainloop()


if __name__ == "__main__":
    app = TradingSimulatorApp()
    app.run()
