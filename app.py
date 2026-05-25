import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# Set up page config
st.set_page_config(page_title="Leveraged Hybrid Backtester", layout="wide")
st.title("📈 Hybrid Rolling Investment Backtester")
st.markdown(
    """
    Simulate a rolling historical backtest combining a **Lump Sum** investment with a **Monthly DCA** or **Value Averaging**.

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
    "Monthly Contribution ($)",
    min_value=0,
    value=1000,
    step=100,
    help="For DCA: fixed amount invested each month. For Value Averaging: the target monthly portfolio increment.",
)

st.sidebar.markdown("---")

st.sidebar.subheader("Strategies")
show_dca = st.sidebar.checkbox("Dollar Cost Averaging (DCA)", value=True)
show_va = st.sidebar.checkbox("Value Averaging (VA)", value=False)
if show_va:
    allow_va_selling = st.sidebar.checkbox(
        "Allow selling in VA",
        value=True,
        help=(
            "When the portfolio overshoots the VA target, sell shares to return to target. "
            "If disabled, invest $0 that month instead of selling."
        ),
    )
else:
    allow_va_selling = True

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
    tqqq_annual_drag = 0.10   # 3× Nasdaq-100

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
    """DCA: invest a fixed monthly_investment every month."""
    dates, final_values, totals_invested = [], [], []
    total_invested = initial_lump_sum + monthly_investment * months

    combined = pd.concat([ticker_series, ls_series], axis=1).dropna()
    combined.columns = ["ticker_price", "ls_price"]

    for start in range(len(combined) - months + 1):
        window = combined.iloc[start : start + months]

        ls_start_price = window["ls_price"].iloc[0]
        ls_final_price = window["ls_price"].iloc[-1]
        lump_sum_shares = initial_lump_sum / ls_start_price if ls_start_price > 0 else 0.0
        lump_sum_final_value = lump_sum_shares * ls_final_price

        ticker_prices = window["ticker_price"].values
        dca_shares = 0.0
        for price in ticker_prices:
            dca_shares += monthly_investment / price
        dca_final_value = dca_shares * ticker_prices[-1]

        dates.append(window.index[0])
        final_values.append(lump_sum_final_value + dca_final_value)
        totals_invested.append(total_invested)

    return pd.DataFrame({"date": dates, "final_value": final_values, "total_invested": totals_invested})


def hybrid_rolling_va(ticker_series, ls_series, months, allow_selling):
    """Value Averaging: each month invest whatever is needed to hit monthly_investment * (month+1)."""
    dates, final_values, totals_invested = [], [], []

    combined = pd.concat([ticker_series, ls_series], axis=1).dropna()
    combined.columns = ["ticker_price", "ls_price"]

    for start in range(len(combined) - months + 1):
        window = combined.iloc[start : start + months]

        ls_start_price = window["ls_price"].iloc[0]
        ls_final_price = window["ls_price"].iloc[-1]
        lump_sum_shares = initial_lump_sum / ls_start_price if ls_start_price > 0 else 0.0
        lump_sum_final_value = lump_sum_shares * ls_final_price

        ticker_prices = window["ticker_price"].values
        va_shares = 0.0
        total_va_invested = 0.0

        for month_idx, price in enumerate(ticker_prices):
            target_value = monthly_investment * (month_idx + 1)
            current_value = va_shares * price
            required = target_value - current_value

            if not allow_selling and required < 0:
                required = 0.0
            elif required < 0:
                # Cap selling at what we actually own (can't go short)
                required = max(required, -va_shares * price)

            va_shares = max(va_shares + required / price, 0.0)
            total_va_invested += required

        va_final_value = va_shares * ticker_prices[-1]
        dates.append(window.index[0])
        final_values.append(lump_sum_final_value + va_final_value)
        totals_invested.append(initial_lump_sum + total_va_invested)

    return pd.DataFrame({"date": dates, "final_value": final_values, "total_invested": totals_invested})


# --- RUN BACKTESTS ---
horizons = {"1Y": 12, "3Y": 36, "5Y": 60, "10Y": 120}
plot_tickers = family_tickers
all_results = []

for t in plot_tickers:
    for label, m in horizons.items():
        if show_dca:
            df = hybrid_rolling(monthly_data[t], monthly_data[lump_sum_ticker], m)
            df["ticker"] = t
            df["horizon"] = label
            df["strategy"] = "DCA"
            all_results.append(df)
        if show_va:
            df = hybrid_rolling_va(monthly_data[t], monthly_data[lump_sum_ticker], m, allow_va_selling)
            df["ticker"] = t
            df["horizon"] = label
            df["strategy"] = "VA"
            all_results.append(df)

if not all_results:
    st.warning("Select at least one strategy in the sidebar to see results.")
    st.stop()

results_df = pd.concat(all_results, ignore_index=True)

# --- PLOTLY CHART ---
COLORS = {
    "SPY":  "#1f77b4",
    "SSO":  "#2ca02c",
    "UPRO": "#d62728",
    "QQQ":  "#1f77b4",
    "QLD":  "#2ca02c",
    "TQQQ": "#d62728",
}
STRATEGY_MARKERS = {"DCA": "circle", "VA": "x"}


def build_chart(df, horizon_label, tickers, log_scale, pct=False):
    fig = go.Figure()
    subset = df[df["horizon"] == horizon_label]
    strategies = [s for s in ["DCA", "VA"] if s in df["strategy"].unique()]

    for strategy in strategies:
        for t in tickers:
            s = subset[(subset["ticker"] == t) & (subset["strategy"] == strategy)]
            if s.empty:
                continue
            net_gain = s["final_value"] - s["total_invested"]
            if pct:
                y_vals = net_gain / s["total_invested"] * 100
                hover_y = "Return: <b>%{y:.1f}%</b>"
            else:
                y_vals = net_gain
                hover_y = "Net gain: <b>$%{y:,.0f}</b>"
            mean_val = y_vals.mean()

            fig.add_trace(go.Scatter(
                x=s["date"],
                y=y_vals,
                mode="markers",
                name=f"{t} ({strategy})",
                marker=dict(
                    color=COLORS[t],
                    symbol=STRATEGY_MARKERS[strategy],
                    size=5,
                    opacity=0.35,
                ),
                customdata=s[["total_invested", "final_value"]].values,
                hovertemplate=(
                    "<b>%{x|%b %Y}</b><br>"
                    f"Ticker: {t}  \u00b7  Strategy: {strategy}<br>"
                    + hover_y + "<br>"
                    "Final value: $%{customdata[1]:,.0f}<br>"
                    "Total invested: $%{customdata[0]:,.0f}"
                    "<extra></extra>"
                ),
            ))

            # Dashed mean line (not in legend)
            x_range = [s["date"].min(), s["date"].max()]
            fig.add_trace(go.Scatter(
                x=x_range,
                y=[mean_val, mean_val],
                mode="lines",
                line=dict(color=COLORS[t], dash="dash", width=1.5),
                showlegend=False,
                hoverinfo="skip",
            ))

    # Break-even reference line
    fig.add_hline(
        y=0,
        line_color="black",
        line_width=1.5,
        annotation_text="Break-even",
        annotation_position="bottom right",
    )

    if pct:
        y_title = "Return on Invested (%)"
        y_fmt = ".1f"
        y_suffix = "%"
        chart_subtitle = "Return on Invested Capital"
    else:
        y_title = "Net Gain ($)"
        y_fmt = "$,.0f"
        y_suffix = ""
        chart_subtitle = "Net Gain (Final Value \u2212 Total Invested)"

    fig.update_layout(
        title=dict(
            text=(
                f"<b>{horizon_label} Rolling Horizon</b> \u2014 {chart_subtitle}"
                f"<br><sup>Lump sum in {lump_sum_ticker} \u00b7 periodic contribution in selected asset"
                f" \u00b7 circle\u202f=\u202fDCA, \u00d7\u202f=\u202fVA</sup>"
            ),
            font=dict(size=15),
        ),
        xaxis_title="Window Start Date",
        yaxis_title=y_title,
        yaxis_type="log" if (log_scale and not pct) else "linear",
        yaxis_tickformat=y_fmt,
        yaxis_ticksuffix=y_suffix,
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="left", x=0),
        height=540,
        hovermode="closest",
        margin=dict(t=110),
    )
    return fig


# --- HORIZON SELECTOR + CHART ---
st.markdown("### Portfolio Ending Value by Rolling Window")
selected_horizon = st.selectbox(
    "Rolling return period",
    options=list(horizons.keys()),
    index=0,
)

fig = build_chart(results_df, selected_horizon, plot_tickers, use_log_scale)
st.plotly_chart(fig, use_container_width=True)

fig_pct = build_chart(results_df, selected_horizon, plot_tickers, use_log_scale, pct=True)
st.plotly_chart(fig_pct, use_container_width=True)


# --- SUMMARY TABLES ---
def render_summary_table(df, strategy_label, tickers):
    s = df[df["strategy"] == strategy_label]

    avg_final = s.groupby(["horizon", "ticker"])["final_value"].mean().reset_index()
    pivot = avg_final.pivot(index="horizon", columns="ticker", values="final_value")
    pivot = pivot.reindex(["1Y", "3Y", "5Y", "10Y"])[tickers]
    pivot.columns.name = None

    if strategy_label == "DCA":
        st.markdown("#### DCA \u2014 Average Terminal Values Across All Historical Windows")
        principals = {label: initial_lump_sum + monthly_investment * m for label, m in horizons.items()}
        pivot.insert(0, "Principal Invested", pd.Series(principals))
        st.dataframe(pivot.style.format("${:,.2f}"))

    else:
        st.markdown("#### Value Averaging (VA) \u2014 Average Terminal Values Across All Historical Windows")
        st.dataframe(pivot.style.format("${:,.2f}"))

        # Per (horizon, ticker): mean / min / max of total_invested across windows
        invested_stats = (
            s.groupby(["horizon", "ticker"])["total_invested"]
            .agg(["mean", "min", "max"])
            .reset_index()
        )

        combined_rows = {}
        for t in tickers:
            t_stats = (
                invested_stats[invested_stats["ticker"] == t]
                .set_index("horizon")
                .reindex(["1Y", "3Y", "5Y", "10Y"])
            )
            combined_rows[t] = t_stats.apply(
                lambda r: f"${r['mean']:,.0f}  (${r['min']:,.0f} \u2013 ${r['max']:,.0f})", axis=1
            )

        combined_df = pd.DataFrame(combined_rows)
        combined_df.columns.name = None
        st.markdown("**VA \u2014 Avg Total Invested per Window (min \u2013 max range)**")
        st.dataframe(combined_df)


for strat in ["DCA", "VA"]:
    if strat in results_df["strategy"].unique():
        render_summary_table(results_df, strat, plot_tickers)
