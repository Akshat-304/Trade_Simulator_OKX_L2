# analyze_slippage_data.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np  # For log scale and handling potential inf/-inf
import os

# --- Configuration ---
DATA_LOG_FILE = "slippage_regression_log.csv"
PERFORMANCE_LOG_FILE = "model_performance_log.csv"
PLOT_OUTPUT_DIR = "output_plots"

# Ensure plot output directory exists
if not os.path.exists(PLOT_OUTPUT_DIR):
    os.makedirs(PLOT_OUTPUT_DIR)


def plot_model_performance_evolution(df_perf):
    if df_perf.empty or "num_training_samples" not in df_perf.columns:
        print(
            "Performance log is empty or missing required columns for evolution plot."
        )
        return

    df_perf = df_perf.sort_values(by="num_training_samples").dropna(
        subset=["test_mse", "test_r2_score"]
    )
    if df_perf.empty:
        print("No valid performance data to plot after sorting/dropping NaNs.")
        return

    fig, ax1 = plt.subplots(figsize=(14, 7))  # Slightly wider

    color = "tab:red"
    ax1.set_xlabel("Number of Training Samples (Log Scale)")
    ax1.set_ylabel("Test MSE (Log Scale)", color=color)
    # Filter out non-positive MSE values for log scale if any, though MSE should be >= 0
    valid_mse = df_perf[df_perf["test_mse"] > 0]
    if not valid_mse.empty:
        ax1.plot(
            valid_mse["num_training_samples"],
            valid_mse["test_mse"],
            color=color,
            marker=".",
            linestyle="-",
            linewidth=1,
            markersize=4,
        )
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_xscale(
        "log"
    )  # Log scale for training samples if it spans orders of magnitude
    ax1.set_yscale("log")
    ax1.grid(True, which="both", ls="-", alpha=0.5)  # Grid for log scale

    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("Test R2 Score", color=color)
    ax2.plot(
        df_perf["num_training_samples"],
        df_perf["test_r2_score"],
        color=color,
        marker="x",
        linestyle="--",
        linewidth=1,
        markersize=4,
    )
    ax2.tick_params(axis="y", labelcolor=color)
    # R2 can be negative, so a fixed sensible range is good.
    min_r2 = (
        min(-0.2, df_perf["test_r2_score"].min() - 0.05) if not df_perf.empty else -0.2
    )
    max_r2 = (
        max(1.0, df_perf["test_r2_score"].max() + 0.05) if not df_perf.empty else 1.0
    )
    ax2.set_ylim([min_r2, max_r2])
    ax2.axhline(0, color="gray", linestyle=":", linewidth=0.8)  # Line at R2=0

    fig.tight_layout()
    plt.title("Slippage Model Performance Evolution (Test Set)")
    plt.savefig(os.path.join(PLOT_OUTPUT_DIR, "model_performance_evolution.png"))
    plt.close(fig)
    print(f"Saved model_performance_evolution.png to {PLOT_OUTPUT_DIR}")


def plot_feature_vs_slippage(df_probes):
    if df_probes.empty:
        print("Probe data is empty, cannot plot feature relationships.")
        return

    features_to_plot = [
        "probe_order_size_usd",
        "market_spread_bps",
        "market_depth_best_ask_usd",
    ]

    # Filter out extreme true_slippage_pct for better visualization if they exist
    # For example, if some probes exhausted the book leading to huge slippage values
    # This depends on what reasonable slippage looks like. Let's cap at 1% for visualization.
    df_probes_filtered = df_probes[
        df_probes["true_slippage_pct_walk_the_book"].abs() < 1.0
    ].copy()
    if len(df_probes_filtered) < 0.1 * len(
        df_probes
    ):  # If filtering removes too much, don't filter
        df_probes_filtered = df_probes.copy()
        print(
            "Warning: Filtering for true_slippage_pct < 1% removed most data. Plotting all data."
        )

    print(
        f"Plotting feature vs slippage using {len(df_probes_filtered)} (potentially filtered) probe data points."
    )

    for feature in features_to_plot:
        if feature not in df_probes_filtered.columns:
            print(f"Feature {feature} not found in probe data.")
            continue

        plt.figure(figsize=(12, 7))  # Slightly larger
        # Use a sample if data is too large to avoid overplotting and save time
        sample_df = (
            df_probes_filtered.sample(
                n=min(10000, len(df_probes_filtered)), random_state=1
            )
            if len(df_probes_filtered) > 10000
            else df_probes_filtered
        )

        # Consider log scale for features if they span many orders of magnitude
        use_log_x = False
        if feature == "probe_order_size_usd" or feature == "market_depth_best_ask_usd":
            if (
                sample_df[feature].max() / (sample_df[feature].min() + 1e-9) > 100
            ):  # Heuristic for log scale
                use_log_x = True

        sns.scatterplot(
            data=sample_df,
            x=feature,
            y="true_slippage_pct_walk_the_book",
            alpha=0.3,
            s=15,
            edgecolor=None,
        )

        if use_log_x:
            plt.xscale("log")
            # Filter out non-positive values for log scale if any (though order size/depth should be positive)
            # sample_df = sample_df[sample_df[feature] > 0]
            # sns.scatterplot(data=sample_df, x=feature, y='true_slippage_pct_walk_the_book', alpha=0.3, s=15, edgecolor=None)

        plt.title(f"{feature} vs. True Slippage % (Probes)")
        plt.xlabel(f"{feature} {'(Log Scale)' if use_log_x else ''}")
        plt.ylabel("True Slippage % (Walk-the-Book)")

        # Zoom y-axis if slippage is mostly small
        median_slip = sample_df["true_slippage_pct_walk_the_book"].median()
        std_slip = sample_df["true_slippage_pct_walk_the_book"].std()
        if std_slip > 0:  # Avoid issues if all slippage is identical
            lower_bound = median_slip - 3 * std_slip
            upper_bound = (
                median_slip + 5 * std_slip
            )  # Allow more space for positive slippage
            # Ensure bounds are reasonable, e.g., not excessively negative for slippage %
            lower_bound = max(lower_bound, -0.01)  # Cap minimum y at -0.01% for example
            upper_bound = min(
                upper_bound, 0.1
            )  # Cap maximum y at 0.1% if data is very concentrated
            # Or use a percentile approach:
            # upper_bound = sample_df['true_slippage_pct_walk_the_book'].quantile(0.99)
            if upper_bound > lower_bound + 1e-5:  # Only set ylim if range is sensible
                plt.ylim([lower_bound, upper_bound])

        plt.grid(True, which="both", ls="--", alpha=0.7)
        plt.savefig(os.path.join(PLOT_OUTPUT_DIR, f"{feature}_vs_true_slippage.png"))
        plt.close()
        print(f"Saved {feature}_vs_true_slippage.png to {PLOT_OUTPUT_DIR}")


def plot_slippage_distribution(df_probes):
    if df_probes.empty or "true_slippage_pct_walk_the_book" not in df_probes.columns:
        print(
            "Probe data is empty or missing slippage column, cannot plot distribution."
        )
        return

    plt.figure(figsize=(12, 7))

    # Filter for visualization: remove extreme outliers if they skew the plot too much
    # For example, consider only slippage between -1% and 1% for the main distribution plot
    slippage_data_for_hist = df_probes["true_slippage_pct_walk_the_book"]
    q_low = slippage_data_for_hist.quantile(0.001)  # Exclude extreme 0.1% tails
    q_hi = slippage_data_for_hist.quantile(0.999)
    # Only apply filter if it results in a tighter, meaningful range
    if q_hi > q_low and (q_hi - q_low) < slippage_data_for_hist.std() * 10:  # Heuristic
        slippage_data_for_hist = slippage_data_for_hist[
            (slippage_data_for_hist >= q_low) & (slippage_data_for_hist <= q_hi)
        ]

    if slippage_data_for_hist.empty:
        print("No slippage data left after filtering for histogram.")
        return

    sns.histplot(
        slippage_data_for_hist, kde=True, bins=100, stat="density"
    )  # Use density for KDE overlay
    plt.title("Distribution of True Slippage % (Probes - Filtered for Visualization)")
    plt.xlabel("True Slippage % (Walk-the-Book)")
    plt.ylabel("Density")
    plt.grid(True, which="both", ls="--", alpha=0.7)

    # Add text for mean/median if useful
    mean_slip = slippage_data_for_hist.mean()
    median_slip = slippage_data_for_hist.median()
    plt.axvline(
        mean_slip,
        color="r",
        linestyle="--",
        linewidth=0.8,
        label=f"Mean: {mean_slip:.4f}%",
    )
    plt.axvline(
        median_slip,
        color="g",
        linestyle=":",
        linewidth=0.8,
        label=f"Median: {median_slip:.4f}%",
    )
    plt.legend()

    plt.savefig(os.path.join(PLOT_OUTPUT_DIR, "true_slippage_distribution.png"))
    plt.close()
    print(f"Saved true_slippage_distribution.png to {PLOT_OUTPUT_DIR}")


def plot_predicted_vs_user_order_size(df_user_pred):  # Renamed for clarity
    if df_user_pred.empty:
        print("User prediction data is empty.")
        return

    if (
        "user_order_size_usd" in df_user_pred.columns
        and "predicted_slippage_pct_regression" in df_user_pred.columns
    ):
        plt.figure(figsize=(12, 7))
        # Sample if data is very large
        sample_df = (
            df_user_pred.sample(n=min(10000, len(df_user_pred)), random_state=1)
            if len(df_user_pred) > 10000
            else df_user_pred
        )

        # Filter out extreme predictions if any, for better visualization
        # sample_df = sample_df[sample_df['predicted_slippage_pct_regression'].abs() < 1.0] # Example: cap at 1%

        sns.scatterplot(
            data=sample_df,
            x="user_order_size_usd",
            y="predicted_slippage_pct_regression",
            alpha=0.3,
            s=15,
            edgecolor=None,
        )

        # Check if x-axis (user_order_size_usd) needs log scale
        if (
            sample_df["user_order_size_usd"].max()
            / (sample_df["user_order_size_usd"].min() + 1e-9)
            > 100
        ):
            plt.xscale("log")
            plt.xlabel("User Order Size (USD) (Log Scale)")
        else:
            plt.xlabel("User Order Size (USD)")

        plt.title("Predicted Slippage % (Regression) vs. User Order Size")
        plt.ylabel("Predicted Slippage % (Regression)")
        plt.grid(True, which="both", ls="--", alpha=0.7)
        plt.savefig(
            os.path.join(PLOT_OUTPUT_DIR, "predicted_slippage_vs_user_order_size.png")
        )
        plt.close()
        print(f"Saved predicted_slippage_vs_user_order_size.png to {PLOT_OUTPUT_DIR}")
    else:
        print(
            "User prediction data missing required columns for plotting (user_order_size_usd, predicted_slippage_pct_regression)."
        )


def main():
    print(f"Analyzing data from {DATA_LOG_FILE} and {PERFORMANCE_LOG_FILE}")

    df_all = None
    try:
        df_all = pd.read_csv(DATA_LOG_FILE)
    except FileNotFoundError:
        print(f"Error: {DATA_LOG_FILE} not found.")
        # Exit if main data file not found
        return
    except Exception as e:
        print(f"Error loading {DATA_LOG_FILE}: {e}")
        return

    # --- Process Probe Data ---
    df_probes = df_all[df_all["probe_order_size_usd"].notna()].copy()
    if not df_probes.empty:
        for col in [
            "probe_order_size_usd",
            "market_spread_bps",
            "market_depth_best_ask_usd",
            "true_slippage_pct_walk_the_book",
        ]:
            df_probes[col] = pd.to_numeric(df_probes[col], errors="coerce")
        df_probes.dropna(
            subset=[
                "probe_order_size_usd",
                "market_spread_bps",
                "market_depth_best_ask_usd",
                "true_slippage_pct_walk_the_book",
            ],
            inplace=True,
        )

        # Filter out rows where market_spread_bps is negative (crossed book artifacts)
        # This should already be handled by the data generation filter, but as a safeguard for analysis:
        initial_probe_count = len(df_probes)
        df_probes = df_probes[df_probes["market_spread_bps"] >= 0].copy()
        if len(df_probes) < initial_probe_count:
            print(
                f"Filtered out {initial_probe_count - len(df_probes)} probe data points with negative spread_bps."
            )

        print(f"Loaded {len(df_probes)} valid probe data points for analysis.")
    else:
        print("No probe data found in log file.")

    # --- Process User Prediction Data ---
    df_user_pred = df_all[df_all["user_order_size_usd"].notna()].copy()
    if not df_user_pred.empty:
        for col in ["user_order_size_usd", "predicted_slippage_pct_regression"]:
            df_user_pred[col] = pd.to_numeric(df_user_pred[col], errors="coerce")
        df_user_pred.dropna(
            subset=["user_order_size_usd", "predicted_slippage_pct_regression"],
            inplace=True,
        )
        print(f"Loaded {len(df_user_pred)} valid user prediction data points.")
    else:
        print("No user prediction data found in log file.")

    # --- Process Model Performance Data ---
    df_perf = pd.DataFrame()  # Initialize as empty
    try:
        df_perf = pd.read_csv(PERFORMANCE_LOG_FILE)
        if not df_perf.empty:
            for col in ["num_training_samples", "test_mse", "test_r2_score"]:
                df_perf[col] = pd.to_numeric(df_perf[col], errors="coerce")
            # df_perf.dropna(subset=['num_training_samples', 'test_mse', 'test_r2_score'], inplace=True) # Keep NaNs for now, plot func handles
            print(f"Loaded {len(df_perf)} model performance records.")
        else:
            print(f"{PERFORMANCE_LOG_FILE} is empty.")
    except FileNotFoundError:
        print(
            f"Warning: {PERFORMANCE_LOG_FILE} not found. Skipping performance evolution plot."
        )
    except Exception as e:
        print(f"Error loading or processing {PERFORMANCE_LOG_FILE}: {e}")

    # --- Generate Plots ---
    if not df_perf.empty:
        plot_model_performance_evolution(df_perf)
    if not df_probes.empty:
        plot_feature_vs_slippage(df_probes)
        plot_slippage_distribution(df_probes)
    if not df_user_pred.empty:
        plot_predicted_vs_user_order_size(df_user_pred)

    print(
        f"\nAnalysis complete. Plots saved to '{PLOT_OUTPUT_DIR}' directory if data was available."
    )


if __name__ == "__main__":
    main()
