import streamlit as st
import yfinance as yf
from scipy.stats import norm  # Ensure norm is imported
import math
import pandas as pd
import numpy as np
import plotly.graph_objects as go

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

# Sidebar for Input Parameters
ticker = st.sidebar.text_input("Stock Ticker", value="AAPL")
sigma = st.sidebar.slider("Implied Volatility (σ)", 0.1, 1.0, 0.2, 0.01)
r = st.sidebar.slider("Risk-Free Rate (r)", 0.0, 0.1, 0.01, 0.001)
expiry_index = st.sidebar.slider("Select Expiry Index", 0, 4, 0)
range_percent = st.sidebar.slider("Strike Range (% around current price)", 0, 50, 15, 1)

# Fetch Data
stock = yf.Ticker(ticker)
try:
    S = stock.history(period="1d")["Close"].iloc[-1]
except IndexError:
    st.error("Failed to fetch stock data. Please check the ticker symbol.")
    st.stop()

expiry_dates = stock.options
if expiry_index >= len(expiry_dates):
    st.error(f"Expiry index out of range. Available indices: 0 to {len(expiry_dates)-1}")
    st.stop()

expiry_date = expiry_dates[expiry_index]
try:
    options_chain = stock.option_chain(expiry_date)
except Exception as e:
    st.error(f"Failed to fetch options data: {e}")
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

# Ensure 'openInterest' is numeric and handle missing values
filtered_options['openInterest'] = pd.to_numeric(filtered_options['openInterest'], errors='coerce').fillna(0)

# Calculate Delta Exposure Separately for Calls and Puts
filtered_options['Delta_Exposure'] = filtered_options.apply(
    lambda row: row['Delta'] * row['openInterest'] * 100, axis=1
)

# Separate Delta Exposure for Calls and Puts
delta_exposure_calls = filtered_options[filtered_options['Type'] == 'Call'].groupby('strike')['Delta_Exposure'].sum().reset_index()
delta_exposure_puts = filtered_options[filtered_options['Type'] == 'Put'].groupby('strike')['Delta_Exposure'].sum().reset_index()

# Merge Calls and Puts Delta Exposure
delta_exposure = pd.merge(delta_exposure_calls, delta_exposure_puts, on='strike', how='outer', suffixes=('_Call', '_Put')).fillna(0)

# Plot Delta Exposure using Plotly
fig = go.Figure()

# Add Call Delta Exposure Bar
fig.add_trace(
    go.Bar(
        x=-delta_exposure['Delta_Exposure_Call'],  # Invert the x-axis for calls
        y=delta_exposure['strike'],
        orientation='h',
        marker=dict(color='rgba(54, 162, 235, 0.7)', line=dict(color='rgba(54, 162, 235, 1.0)', width=1)),
        hovertemplate='Strike: %{y}<br>Call Delta Exposure: %{x:,}<extra></extra>',
        name='Call Delta Exposure'
    )
)

# Add Put Delta Exposure Bar
fig.add_trace(
    go.Bar(
        x=-delta_exposure['Delta_Exposure_Put'],  # Invert the x-axis for puts
        y=delta_exposure['strike'],
        orientation='h',
        marker=dict(color='rgba(255, 159, 64, 0.7)', line=dict(color='rgba(255, 159, 64, 1.0)', width=1)),
        hovertemplate='Strike: %{y}<br>Put Delta Exposure: %{x:,}<extra></extra>',
        name='Put Delta Exposure'
    )
)

# Determine the maximum absolute value for symmetric x-axis
max_delta = max(
    delta_exposure['Delta_Exposure_Call'].abs().max(),
    delta_exposure['Delta_Exposure_Put'].abs().max()
)
buffer = max_delta * 0.1
x_limit = max_delta + buffer

# Update the layout to have symmetric x-axis
fig.update_layout(
    xaxis=dict(
        title="Delta Exposure",
        range=[x_limit, -x_limit],  # Invert the x-axis
        tickvals=[3_000_000, 0, -3_000_000],  # Custom tick labels
        ticktext=["3M", "0", "-3M"]
    ),
    yaxis_title="Strike Price",
    yaxis=dict(autorange='reversed'),
    template='plotly_white',
    height=700,
    margin=dict(l=100, r=50, t=100, b=50),
    legend=dict(x=0.8, y=1.1, orientation='h'),
    title=f"Delta Exposure by Strike Price for {ticker.upper()} ({expiry_date})",
    barmode='relative',
)

# Add vertical and horizontal reference lines
fig.add_shape(
    type='line',
    x0=0,
    y0=lower_bound,
    x1=0,
    y1=upper_bound,
    line=dict(color='black', width=1, dash='dash')
)
fig.add_shape(
    type='line',
    x0=x_limit,
    y0=S,
    x1=-x_limit,
    y1=S,
    line=dict(color='green', width=2, dash='dot')
)
fig.add_annotation(
    x=x_limit * 0.95,
    y=S,
    text=f"Current Price: ${S:.2f}",
    showarrow=False,
    font=dict(color="green", size=12)
)

# Display the plot
st.plotly_chart(fig, use_container_width=True)
