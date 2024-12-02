import streamlit as st
import yfinance as yf
from scipy.stats import norm  # Ensure norm is imported
import math
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Black-Scholes Delta Calculation
def calculate_delta(S, K, r, T, sigma, option_type="call"):
    if T <= 0:
        return 0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    if option_type.lower() == "call":
        return norm.cdf(d1)
    elif option_type.lower() == "put":
        return norm.cdf(d1) - 1

# Streamlit UI
st.set_page_config(page_title="Delta Hedging Exposure Visualization Tool", layout="wide")
st.title("📈 Delta Hedging Exposure Visualization Tool")

# Sidebar for Input Parameters
st.sidebar.header("🛠️ Input Parameters")

# User Inputs with Descriptions
ticker = st.sidebar.text_input(
    "Stock Ticker",
    value="AAPL",
    help="Enter the ticker symbol of the stock you want to analyze (e.g., AAPL for Apple Inc.)."
)

sigma = st.sidebar.slider(
    "📊 Implied Volatility (σ)",
    0.1,
    1.0,
    0.2,
    0.01,
    help=(
        "Implied Volatility represents the market's forecast of a likely movement in the stock's price. "
        "Higher volatility indicates a higher risk and potentially higher option premiums."
    )
)

r = st.sidebar.slider(
    "🏦 Risk-Free Rate (r)",
    0.0,
    0.1,
    0.01,
    0.001,
    help=(
        "The Risk-Free Rate is the theoretical rate of return of an investment with zero risk, "
        "typically based on government bonds. It is used in option pricing models to discount future payoffs."
    )
)

expiry_index = st.sidebar.slider(
    "📅 Select Expiry Index",
    0,
    4,
    0,
    help=(
        "Select the index corresponding to the option's expiry date from the available list. "
        "Different expiry dates can significantly impact option pricing and delta exposure."
    )
)

range_percent = st.sidebar.slider(
    "📏 Strike Range (% around current price)",
    0,
    50,
    15,
    1,
    help=(
        "Define the percentage range around the current stock price to include strike prices for analysis. "
        "For example, a range of 15% will include strikes from -15% to +15% around the current price."
    )
)

# Fetch Data
stock = yf.Ticker(ticker)
try:
    S = stock.history(period="1d")["Close"].iloc[-1]
except IndexError:
    st.error("❌ Failed to fetch stock data. Please check the ticker symbol.")
    st.stop()

expiry_dates = stock.options
if expiry_index >= len(expiry_dates):
    st.error(f"❌ Expiry index out of range. Available indices: 0 to {len(expiry_dates)-1}")
    st.stop()

expiry_date = expiry_dates[expiry_index]
try:
    options_chain = stock.option_chain(expiry_date)
except Exception as e:
    st.error(f"❌ Failed to fetch options data: {e}")
    st.stop()

# Calculate Time to Expiry in Years
T = (pd.to_datetime(expiry_date) - pd.Timestamp.now()).days / 365
T = max(T, 0.0001)  # Prevent division by zero or negative time

# Calculate Delta for Calls and Puts
calls = options_chain.calls.copy()
puts = options_chain.puts.copy()
calls["Delta"] = calls.apply(lambda x: calculate_delta(S, x["strike"], r, T, sigma, "call"), axis=1)
puts["Delta"] = puts.apply(lambda x: calculate_delta(S, x["strike"], r, T, sigma, "put"), axis=1)

# Add Type Column
calls["Type"] = "Call"
puts["Type"] = "Put"
options = pd.concat([calls, puts])

# Filter Strikes Within Specified Range
lower_bound = S * (1 - range_percent / 100)
upper_bound = S * (1 + range_percent / 100)
filtered_options = options[(options['strike'] >= lower_bound) & (options['strike'] <= upper_bound)]

# Verify if 'openInterest' exists
if 'openInterest' not in filtered_options.columns:
    st.error("❌ The fetched options data does not contain 'openInterest'. Please verify the data source.")
    st.stop()

# Ensure 'openInterest' is numeric and handle missing values
filtered_options['openInterest'] = pd.to_numeric(filtered_options['openInterest'], errors='coerce').fillna(0)

# Calculate Delta Exposure Separately for Calls and Puts
# Multiply by 100 to account for contract size
filtered_options['Delta_Exposure'] = filtered_options.apply(
    lambda row: row['Delta'] * row['openInterest'] * 100, axis=1
)

# Separate Delta Exposure for Calls and Puts
delta_exposure_calls = filtered_options[filtered_options['Type'] == 'Call'].groupby('strike')['Delta_Exposure'].sum().reset_index()
delta_exposure_puts = filtered_options[filtered_options['Type'] == 'Put'].groupby('strike')['Delta_Exposure'].sum().reset_index()

# Merge Calls and Puts Delta Exposure
delta_exposure = pd.merge(delta_exposure_calls, delta_exposure_puts, on='strike', how='outer', suffixes=('_Call', '_Put')).fillna(0)

# Display Data
st.subheader(f"📊 Delta Hedging Exposure for {ticker.upper()} (Expiry: {expiry_date})")
st.markdown(f"**Current Stock Price:** ${S:.2f} | **Strike Range:** ±{range_percent}%")
st.dataframe(
    delta_exposure.set_index('strike').rename(
        columns={'Delta_Exposure_Call': 'Call Delta Exposure', 'Delta_Exposure_Put': 'Put Delta Exposure'}
    ).astype(int)
)

# Function to format tick labels
def format_ticks(value, tick_type):
    if tick_type == 'x':
        if abs(value) >= 1_000_000:
            return f"{value / 1_000_000:.1f}M"
        elif abs(value) >= 1_000:
            return f"{value / 1_000:.1f}K"
    return value

# Plot Delta Exposure using Plotly
st.subheader("📈 Delta Exposure by Strike Price")

fig = go.Figure()

# Add Call Delta Exposure Bar
fig.add_trace(
    go.Bar(
        x=delta_exposure['Delta_Exposure_Call'],
        y=delta_exposure['strike'],
        orientation='h',
        marker=dict(
            color='rgba(54, 162, 235, 0.7)',  # Soft Blue
            line=dict(color='rgba(54, 162, 235, 1.0)', width=1)
        ),
        hovertemplate='Strike: %{y}<br>Call Delta Exposure: %{x:,}<extra></extra>',
        name='Call Delta Exposure'
    )
)

# Add Put Delta Exposure Bar
fig.add_trace(
    go.Bar(
        x=delta_exposure['Delta_Exposure_Put'],
        y=delta_exposure['strike'],
        orientation='h',
        marker=dict(
            color='rgba(255, 159, 64, 0.7)',  # Soft Orange
            line=dict(color='rgba(255, 159, 64, 1.0)', width=1)
        ),
        hovertemplate='Strike: %{y}<br>Put Delta Exposure: %{x:,}<extra></extra>',
        name='Put Delta Exposure'
    )
)

# Determine the maximum absolute value for symmetric x-axis
max_delta = max(
    delta_exposure['Delta_Exposure_Call'].abs().max(),
    delta_exposure['Delta_Exposure_Put'].abs().max()
)
# Add a buffer (e.g., 10%) to the max_delta for better visualization
buffer = max_delta * 0.1
x_limit = max_delta + buffer

# Update the layout to have symmetric x-axis
fig.update_layout(
    xaxis=dict(
        title="Delta Exposure",
        range=[-x_limit, x_limit],
        tickformat=".2s",
        ticksuffix=" ",
        tickvals=np.linspace(-x_limit, x_limit, num=5),
        ticktext=[format_ticks(val, 'x') for val in np.linspace(-x_limit, x_limit, num=5)]
    ),
    yaxis_title="Strike Price",
    yaxis=dict(autorange='reversed'),  # To have lower strikes at the bottom
    template='plotly_white',
    hovermode='closest',
    height=700,
    margin=dict(l=100, r=50, t=100, b=50),
    legend=dict(x=0.8, y=1.1, orientation='h'),
    title=f"Delta Exposure by Strike Price for {ticker.upper()} ({expiry_date})",
    barmode='relative',  # Use 'group' for side-by-side bars or 'relative' for stacked bars
)

# Add a vertical dashed line at x=0 for reference
fig.add_shape(
    type='line',
    x0=0,
    y0=lower_bound,
    x1=0,
    y1=upper_bound,
    line=dict(color='black', width=1, dash='dash')
)

# Add a horizontal dotted line at the current stock price
fig.add_shape(
    type='line',
    x0=-x_limit,
    y0=S,
    x1=x_limit,
    y1=S,
    line=dict(color='green', width=2, dash='dot'),
)

# Add annotation for the current stock price line
fig.add_annotation(
    x=x_limit * 0.95,  # Position near the end of the x-axis
    y=S,
    xref="x",
    yref="y",
    text=f"Current Price: ${S:.2f}",
    showarrow=False,
    yshift=10,
    font=dict(color="green", size=12)
)

# Customize x-axis tick labels for better readability
fig.update_xaxes(
    tickmode='linear',
    tick0=-x_limit,
    dtick=x_limit / 4,
    tickformat=".2s"
)

# Display the plot
st.plotly_chart(fig, use_container_width=True)

# Additional Insights (Optional)
st.markdown("---")
st.subheader("🔍 Additional Insights")

# Net Delta Exposure for Calls and Puts
net_delta_calls = delta_exposure['Delta_Exposure_Call'].sum()
net_delta_puts = delta_exposure['Delta_Exposure_Put'].sum()
st.metric(
    "📌 Net Call Delta Exposure",
    f"{net_delta_calls:,}",
    delta="🔺" if net_delta_calls >= 0 else "🔻",
    delta_color="normal" if net_delta_calls >=0 else "inverse"
)
st.metric(
    "📌 Net Put Delta Exposure",
    f"{net_delta_puts:,}",
    delta="🔺" if net_delta_puts >= 0 else "🔻",
    delta_color="normal" if net_delta_puts >=0 else "inverse"
)

# Overall Net Delta Exposure
overall_net_delta = net_delta_calls + net_delta_puts
st.metric(
    "📌 Overall Net Delta Exposure",
    f"{overall_net_delta:,}",
    delta="🔺" if overall_net_delta >=0 else "🔻",
    delta_color="normal" if overall_net_delta >=0 else "inverse"
)

# Footer
st.markdown("---")
st.markdown(
    "💡 **Note:** Delta hedging exposure represents the sensitivity of option positions to changes in the underlying asset's price. "
    "Positive delta indicates a net long position, while negative delta indicates a net short position."
)
