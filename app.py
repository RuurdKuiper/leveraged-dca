import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# Set up page config
st.set_page_config(page_title="Leveraged Hybrid Backtester", layout="wide")
st.title("📈 Hybrid Rolling Investment Backtester")
st.markdown(
    """
    Simulate a rolling historical backtest combining a **Lump Sum** investment with a **Monthly DCA**.

    This tool is specifically designed to help answer the question: **are leveraged ETFs a good long-term investment?**
    It lets you compare two index families across all historical rolling windows — including the dot-com bust (2000–2002)
    and the 2008 financial crisis.

    - **S&P 500 family:** SPY (1×) · SSO (2×) · UPRO (3×)
    - **Nasdaq-100 family:** QQQ (1×) · QLD (2×) · TQQQ (3×)

    > ⚠️ **Data note:** SSO, UPRO, QLD, and TQQQ all launched after 1999. Price history before each ETF's IPO is
    > **simulated** using leveraged daily index returns minus a synthetic tracking-drag (SSO 3.5 %/yr, UPRO 8.0 %/yr,
    > QLD 4.0 %/yr, TQQQ 10.0 %/yr) to give a realistic picture through past crashes.
    """
)

# --- SIDEBAR FOR INTERACTIVE INPUTS ---
st.sidebar.header("Strategy Settings")

start_year = st.sidebar.slider(
    "Data Start Year",
    min_value=1990,
    max_value=2015,
    value=1990,
    step=1,
    help="Earliest year of historical data to include. Sliding forward shrinks the window and removes older crash data.",
)

st.sidebar.markdown("---")

family = st.sidebar.radio(
    "Index Family",
    options=["SPY Family (S&P 500)", "QQQ Family (Nasdaq-100)"],
    index=0,
    horizontal=True,
)
family_tickers = ["SPY", "SSO", "UPRO"] if family.startswith("SPY") else ["QQQ", "QLD", "TQQQ"]

st.sidebar.markdown("---")

initial_lump_sum = st.sidebar.number_input(
    "Initial Lump Sum ($)", min_value=0, value=0, step=5000
)
lump_sum_ticker = st.sidebar.selectbox(
    "Lump Sum Asset Target", options=family_tickers, index=0
)

st.sidebar.markdown("---")

monthly_investment = st.sidebar.number_input(
    "Monthly DCA Amount ($)", min_value=0, value=1000, step=100
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Simulated data:** SSO, UPRO, QLD, and TQQQ price history before their respective IPOs "
    "is back-filled synthetically using leveraged index returns minus a daily tracking-drag "
    "(SSO 3.5 %/yr · UPRO 8.0 %/yr · QLD 4.0 %/yr · TQQQ 6.0 %/yr)."
)

st.sidebar.markdown("---")
use_log_scale = st.sidebar.toggle("Logarithmic Y-axis", value=False)


# --- DATA LOAD & SIMULATION (CACHED) ---
@st.cache_data
def get_and_simulate_data(start_date: str):
    tickers = ["SPY", "QQQ", "SSO", "UPRO", "QLD", "TQQQ"]
    raw_data = yf.download(tickers, start=start_date)

    daily_prices = pd.DataFrame()
    for t in tickers:
        col = "Adj Close" if ("Adj Close", t) in raw_data.columns else "Close"
        daily_prices[t] = raw_data[col][t]

    daily_prices = daily_prices.dropna(subset=["SPY", "QQQ"])

    daily_prices["SPY_ret"] = daily_prices["SPY"].pct_change()
    daily_prices["QQQ_ret"] = daily_prices["QQQ"].pct_change()

    # Synthetic backfill generator
    def generate_synthetic(df, target_ticker, base_return_col, leverage, fee_decay):
        first_real_date = df[target_ticker].first_valid_index()
        simulated_series = df[target_ticker].copy()
        missing_dates = df.loc[:first_real_date].index[::-1][1:]

        current_price = df.loc[first_real_date, target_ticker]
        for date in missing_dates:
            day_forward_return = df.loc[date, base_return_col]
            lev_return = (leverage * day_forward_return) - fee_decay
            current_price = current_price / (1 + lev_return)
            simulated_series.loc[date] = current_price
        return simulated_series

    sso_annual_drag  = 0.035  # 2× S&P 500
    upro_annual_drag = 0.080  # 3× S&P 500
    qld_annual_drag  = 0.040  # 2× Nasdaq-100
    tqqq_annual_drag = 0.10  # 3× Nasdaq-100

    daily_prices["SSO_sim"]  = generate_synthetic(
        daily_prices, "SSO",  "SPY_ret", 2, sso_annual_drag  / 252
    )
    daily_prices["UPRO_sim"] = generate_synthetic(
        daily_prices, "UPRO", "SPY_ret", 3, upro_annual_drag / 252
    )
    daily_prices["QLD_sim"]  = generate_synthetic(
        daily_prices, "QLD",  "QQQ_ret", 2, qld_annual_drag  / 252
    )
    daily_prices["TQQQ_sim"] = generate_synthetic(
        daily_prices, "TQQQ", "QQQ_ret", 3, tqqq_annual_drag / 252
    )

    monthly = {
        "SPY":  daily_prices["SPY"].resample("ME").last(),
        "QQQ":  daily_prices["QQQ"].resample("ME").last(),
        "SSO":  daily_prices["SSO_sim"].resample("ME").last(),
        "UPRO": daily_prices["UPRO_sim"].resample("ME").last(),
        "QLD":  daily_prices["QLD_sim"].resample("ME").last(),
        "TQQQ": daily_prices["TQQQ_sim"].resample("ME").last(),
    }
    return monthly


# Load the data
monthly_data = get_and_simulate_data(f"{start_year}-01-01")


# --- BACKTEST CORE LOGIC ---
def hybrid_rolling(ticker_series, ls_series, months):
    dates = []
    final_values = []

    combined = pd.concat([ticker_series, ls_series], axis=1).dropna()
    combined.columns = ["ticker_price", "ls_price"]

    for start in range(len(combined) - months + 1):
        window = combined.iloc[start : start + months]

        # Dynamic Lump Sum Target
        ls_start_price = window["ls_price"].iloc[0]
        ls_final_price = window["ls_price"].iloc[-1]
        lump_sum_shares = initial_lump_sum / ls_start_price
        lump_sum_final_value = lump_sum_shares * ls_final_price

        # DCA Target Ticker
        ticker_prices = window["ticker_price"].values
        dca_shares = 0
        for price in ticker_prices:
            dca_shares += monthly_investment / price
        dca_final_value = dca_shares * ticker_prices[-1]

        total_final_value = lump_sum_final_value + dca_final_value

        dates.append(window.index[0])
        final_values.append(total_final_value)

    return pd.DataFrame({"date": dates, "final_value": final_values})


horizons = {"1Y": 12, "3Y": 36, "5Y": 60, "10Y": 120}
plot_tickers = family_tickers
all_results = []

# Process historical paths dynamically based on choices
for t in plot_tickers:
    for label, m in horizons.items():
        df = hybrid_rolling(monthly_data[t], monthly_data[lump_sum_ticker], m)
        df["ticker"] = t
        df["horizon"] = label
        all_results.append(df)

results_df = pd.concat(all_results)
summary = (
    results_df.groupby(["horizon", "ticker"])["final_value"]
    .mean()
    .reset_index()
)

# --- GRID SUBPLOT GRAPHICS ---
colors = {
    "SPY": "tab:blue",  "SSO": "tab:cyan",   "UPRO": "tab:green",
    "QQQ": "tab:purple", "QLD": "tab:orange", "TQQQ": "tab:red",
}

# Establish a 2x2 grid framework
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
axes_flat = axes.flatten()

for idx, (label, m) in enumerate(horizons.items()):
    ax = axes_flat[idx]

    subset = results_df[results_df["horizon"] == label]
    mean_subset = summary[summary["horizon"] == label]
    total_principal = initial_lump_sum + (monthly_investment * m)

    for t in plot_tickers:
        s = subset[subset["ticker"] == t]
        ax.scatter(s["date"], s["final_value"], label=t, color=colors[t], alpha=0.4, s=15)

        mean_val = mean_subset[mean_subset["ticker"] == t]["final_value"].values[
            0
        ]
        ax.axhline(
            mean_val,
            linestyle="--",
            linewidth=1.5,
            color=colors[t],
            label="_nolegend_",
        )

    ax.axhline(
        total_principal,
        color="black",
        linestyle="-",
        linewidth=1.5,
        label="_nolegend_",
    )

    # Styling and Log Scaling per Subplot
    ax.set_title(f"{label} Investment Horizon", fontsize=12, fontweight="bold")
    ax.set_yscale("log" if use_log_scale else "linear")
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.tick_params(axis="x", rotation=30)

    # Legend only on the first subplot
    if idx == 0:
        ax.legend(loc="upper left", fontsize=10)

fig.suptitle(
    f"Portfolio Ending Value Scenarios\n(Lump Sum in {lump_sum_ticker} / Monthly DCA in Selected Asset)",
    fontsize=16,
    fontweight="bold",
    y=0.98,
)
plt.tight_layout()

# Display tabular data block above the charts
st.markdown("### Average Terminal Values Across All Historical Windows")
summary_pivot = summary.pivot(
    index="horizon", columns="ticker", values="final_value"
)
# Reorder index to match timeline flow
summary_pivot = summary_pivot.reindex(["1Y", "3Y", "5Y", "10Y"])
# Prepend total principal column
principals = {label: initial_lump_sum + (monthly_investment * m) for label, m in horizons.items()}
summary_pivot.insert(0, "Principal", pd.Series(principals))
summary_pivot = summary_pivot[["Principal"] + plot_tickers]
st.dataframe(summary_pivot.style.format("${:,.2f}"))

# Render cleanly into Streamlit
st.pyplot(fig)