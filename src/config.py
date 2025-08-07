# src/config.py

# OKX Spot Trading Fee Rates (Taker fees, as market orders are Takers)
# These are examples. Refer to the latest OKX documentation for accuracy.
# Format: "Fee Tier Name": taker_fee_rate (as a decimal, e.g., 0.001 for 0.1%)
OKX_FEE_RATES = {
    # Regular Users (based on KSM holdings, simplified here)
    "Regular User LV1": {"taker": 0.0010},  # 0.10%
    "Regular User LV2": {"taker": 0.0009},  # 0.09% (example)
    "Regular User LV3": {"taker": 0.0008},  # 0.08% (example)
    # VIP Tiers (based on 30-day trading volume and asset balance)
    "VIP 1": {"taker": 0.0008},  # 0.08%
    "VIP 2": {"taker": 0.0007},  # 0.07%
    "VIP 3": {"taker": 0.0006},  # 0.06%
    "VIP 4": {"taker": 0.0005},  # 0.05%
    "VIP 5": {"taker": 0.0004},  # 0.04%
    "VIP 6": {"taker": 0.0003},  # 0.03%
    "VIP 7": {"taker": 0.0002},  # 0.02%
    "VIP 8": {"taker": 0.0001},  # 0.01%
    # A default/custom if tier not found, or if user selects "Custom"
    "Custom": {"taker": 0.0010},  # Default to 0.10% if custom or not found
}

# Default fee if a tier is not found in the map
DEFAULT_TAKER_FEE_RATE = 0.0010  # 0.10%

# ...
# --- Market Impact Model Parameters (Simplified Almgren-Chriss like) ---
# This is a placeholder. Real daily volume should be fetched or estimated.
# For BTC-USDT, daily volume is typically in billions. Let's use 5 Billion USD as an example.
ASSUMED_DAILY_VOLUME_USD = {"BTC-USDT-SWAP": 5_000_000_000.0}  # 5 Billion USD

# This is a coefficient that needs to be calibrated. It's a simplification.
# It determines how strongly order size relative to daily volume impacts the price.
# For example, if impact_coeff=0.1, volatility=0.02 (2%), X_usd/DailyVol=0.001 (0.1% of daily vol),
# then price impact fraction = 0.1 * 0.02 * 0.001 = 0.000002 or 0.0002%
# And market impact cost = 0.000002 * X_usd
# This might be too small. Let's try a larger coefficient for demonstration.
# A common form for temporary impact is eta * (rate_of_trading).
# A common form for permanent impact is gamma * (total_quantity_traded).
# Let's use a simplified form: ImpactCost = C * volatility * (OrderSizeUSD / DailyVolumeUSD) * OrderSizeUSD
# C needs to be chosen. If C=1, and we trade 1% of daily volume, with 2% vol,
# ImpactCost = 1 * 0.02 * 0.01 * OrderSizeUSD = 0.0002 * OrderSizeUSD (0.02% of order size)
MARKET_IMPACT_COEFFICIENT = 0.5  # Tunable parameter, dimensionless
