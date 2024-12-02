import streamlit as st
import yfinance as yf
from scipy.stats import norm
import math
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Function to abbreviate numbers with K, M, B
def abbreviate_number(num):
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.1f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.1f}K"
    elif num >= 0:
        return f"{num:.0f}"
    else:
        return f"-{abbreviate_number(abs(num))}"

# Black-Scholes Delta Calculation
def calculate_delta(S, K, r, T, sigma, option_type="call"):
    if T <= 0:
        return 0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    if option_type.lower() == "call":
        return norm.cdf(d1)
    elif option_type.lower() == "put":
        return norm.cdf(d1) - 1

# Streamlit UI Configuration
st.set_page_config(page_title="Delta Hedging Exposure Visualization Tool", layout="wide")
st.title("📈 Delta Hedging Exposure Visualization Tool")

# Sidebar for Input Parameters
with st.sidebar:
    st.header("🛠️ Input Parameters")
    
    with st.expander("📉 Stock Information"):
        ticker = st.text_input(
            "Stock Ticker",
            value="AAPL",
            help="Enter the ticker symbol of the stock you want to analyze (e.g., AAPL for Apple Inc.)."
        )
    
    with st.expander("📊 Volatility & Rates"):
        sigma = st.slider(
            "Implied Volatility (σ)",
            0.1,
            1.0,
            0.2,
            0.01,
            help=(
                "Implied Volatility represents the market's forecast of a likely movement in the stock's price. "
                "Higher volatility indicates a higher risk and potentially higher option premiums."
            )
        )
        
        r = st.slider(
            "Risk-Free Rate (r)",
            0.0,
            0.1,
            0.01,
            0.001,
            help=(
                "The Risk-Free Rate is the theoretical rate of return of an investment with zero risk, "
                "typically based on government bonds. It is used in option pricing models to discount future payoffs."
            )
        )
    
    with st.expander("📅 Expiry & Strikes"):
        expiry_index = st.slider(
            "Select Expiry Index",
            0,
            4,
            0,
            help=(
                "Select the index corresponding to the option's expiry date from the available list. "
                "Different expiry dates can significantly impact option pricing and delta exposure."
            )
        )
        
        range_percent = st.slider(
            "Strike Range (% around current price)",
            0,
            50,
            15,
            1,
            help=(
                "Define the percentage range around the current stock price to include strike prices for analysis. "
                "For example, a range of 15% will include strikes from -15% to +15% around the current price."
            )
        )
    
    st.markdown("---")
    st.markdown(
        "💡 **Note:** Adjust the parameters in the sidebar to update the visualization and insights dynamically."
    )

# Fetch Data
stock = yf.Ticker(ticker)
try:
    S = stock.history(period="1d")["Close"].iloc[-1]
except IndexError:
    st.error("❌ Failed to fetch stock data. Please check the ticker symbol.")
    st.stop()

expiry_dates = stock.options
if len(expiry_dates) == 0:
    st.error("❌ No options data available for this ticker.")
    st.stop()

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

# Tabs for Organized Display (Visualization first)
tab1, tab2, tab3 = st.tabs(["📈 Visualization", "📊 Data Overview", "🔍 Analytics"])

with tab1:
    st.subheader("📈 Delta Exposure by Strike Price")
    
    # Function to format tick labels
    def format_ticks(value, tick_type):
        if tick_type == 'x':
            return abbreviate_number(value)
        return value
    
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
            tickmode='linear',
            tick0=-x_limit,
            dtick=x_limit / 4,
            tickformat=".2s",
            tickvals=np.linspace(-x_limit, x_limit, num=5),
            ticktext=[format_ticks(val, 'x') for val in np.linspace(-x_limit, x_limit, num=5)]
        ),
        yaxis_title="Strike Price",
        yaxis=dict(autorange='reversed'),  # To have lower strikes at the bottom
        template='plotly_white',
        hovermode='closest',
        height=700,
        margin=dict(l=120, r=50, t=100, b=50),
        legend=dict(x=0.8, y=1.15, orientation='h', bgcolor='rgba(0,0,0,0)'),
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
    
    # Update hover templates to use abbreviated numbers
    fig.update_traces(
        hovertemplate=[
            f"Strike: {strike}<br>Call Delta Exposure: {abbreviate_number(call):s}" 
            for strike, call in zip(delta_exposure['strike'], delta_exposure['Delta_Exposure_Call'])
        ] + [
            f"Strike: {strike}<br>Put Delta Exposure: {abbreviate_number(put):s}" 
            for strike, put in zip(delta_exposure['strike'], delta_exposure['Delta_Exposure_Put'])
        ]
    )
    
    # Display the plot
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader(f"📊 Delta Hedging Exposure for {ticker.upper()} (Expiry: {expiry_date})")
    st.markdown(f"**Current Stock Price:** ${S:.2f} | **Strike Range:** ±{range_percent}%")
    
    # Styled DataFrame with abbreviated numbers
    styled_df = delta_exposure.set_index('strike').rename(
        columns={
            'Delta_Exposure_Call': 'Call Delta Exposure', 
            'Delta_Exposure_Put': 'Put Delta Exposure'
        }
    ).astype(int).copy()
    
    # Apply abbreviation
    styled_df['Call Delta Exposure'] = styled_df['Call Delta Exposure'].apply(abbreviate_number)
    styled_df['Put Delta Exposure'] = styled_df['Put Delta Exposure'].apply(abbreviate_number)
    
    st.dataframe(styled_df.style.format("{:}"), height=400)

with tab3:
    st.subheader("🔍 Analytics")

    # Ensure that required data is available
    if filtered_options.empty:
        st.warning("No options data available within the specified strike range.")
        st.stop()

    # Classify options as ITM or OTM
    def get_moneyness(row):
        if row['Type'] == 'Call':
            return 'ITM' if row['strike'] < S else 'OTM'
        else:
            return 'ITM' if row['strike'] > S else 'OTM'

    filtered_options['Moneyness'] = filtered_options.apply(get_moneyness, axis=1)

    # Group by Type and Moneyness
    grouped = filtered_options.groupby(['Type', 'Moneyness'])

    # Sum delta exposures
    delta_exposure_summary = grouped['Delta_Exposure'].sum().reset_index()

    # Calculate total call delta exposure and total put delta exposure
    total_call_delta = delta_exposure_summary[delta_exposure_summary['Type'] == 'Call']['Delta_Exposure'].sum()
    total_put_delta = delta_exposure_summary[delta_exposure_summary['Type'] == 'Put']['Delta_Exposure'].sum()

    # Display overall positioning
    if total_call_delta > abs(total_put_delta) * 1.1:
        overall_positioning = 'Bullish'
    elif abs(total_put_delta) > total_call_delta * 1.1:
        overall_positioning = 'Bearish'
    else:
        overall_positioning = 'Neutral'

    st.write(f"**Overall Positioning:** {overall_positioning}")

    # Identify and display only one scenario based on priority
    scenario_text = ""

    # Priority 1: Bullish Positioning
    if overall_positioning == 'Bullish':
        scenario_text = """
        **Scenario:** Call delta nodes dominate, indicating bullish positioning.
        The delta exposure chart shows that call delta nodes are larger and more frequent than put delta nodes.
        This suggests that traders are predominantly buying call options, expecting the stock price to rise.
        Market makers, in response, may hedge by buying the underlying stock, adding upward pressure to the price.
        """

    # Priority 2: Bearish Positioning
    elif overall_positioning == 'Bearish':
        scenario_text = """
        **Scenario:** Put delta nodes dominate, indicating bearish positioning.
        The delta exposure chart reveals that put delta nodes are larger and more frequent than call delta nodes.
        This implies that traders are heavily buying put options, anticipating a decline in the stock price.
        Market makers may hedge by selling the underlying stock, contributing to downward pressure on the price.
        """

    # Priority 3: Neutral Positioning
    elif overall_positioning == 'Neutral':
        scenario_text = """
        **Scenario:** Call and put delta nodes are balanced, indicating neutral positioning.
        The delta exposure chart indicates a balance between call and put delta nodes.
        This balance suggests that traders are uncertain about the stock's direction, possibly leading to range-bound or sideways price action.
        Market makers' hedging activities may have a neutral effect on the stock's price.
        """

    # Additional Priority 4: Significant ITM Call Delta (Support Levels)
    # This scenario is considered only if no other scenario has been set
    if not scenario_text:
        itm_calls = filtered_options[(filtered_options['Type'] == 'Call') & (filtered_options['Moneyness'] == 'ITM')]
        if not itm_calls.empty and itm_calls['Delta_Exposure'].sum() > 0:
            scenario_text = """
            **Scenario:** Significant ITM call delta nodes acting as support levels.
            There are substantial in-the-money call delta exposures below the current price, indicating strong support levels.
            These levels may act as a floor, as market makers hedge by buying the stock when prices approach these strikes.
            """

    # Priority 5: Significant OTM Call Delta (Bullish Targets)
    if not scenario_text:
        otm_calls = filtered_options[(filtered_options['Type'] == 'Call') & (filtered_options['Moneyness'] == 'OTM')]
        if not otm_calls.empty and otm_calls['Delta_Exposure'].sum() > 0:
            scenario_text = """
            **Scenario:** Significant OTM call delta nodes acting as bullish targets.
            There is notable out-of-the-money call delta exposure above the current price, suggesting traders are targeting higher price levels.
            These strikes may act as magnets, drawing the price upward as market makers hedge their positions.
            """

    # Priority 6: Significant ITM Put Delta (Resistance Levels)
    if not scenario_text:
        itm_puts = filtered_options[(filtered_options['Type'] == 'Put') & (filtered_options['Moneyness'] == 'ITM')]
        if not itm_puts.empty and itm_puts['Delta_Exposure'].sum() < 0:
            scenario_text = """
            **Scenario:** Significant ITM put delta nodes acting as resistance levels.
            There are substantial in-the-money put delta exposures above the current price, indicating strong resistance levels.
            These levels may act as a ceiling, as market makers hedge by selling the stock when prices approach these strikes.
            """

    # Priority 7: Significant OTM Put Delta (Bearish Targets)
    if not scenario_text:
        otm_puts = filtered_options[(filtered_options['Type'] == 'Put') & (filtered_options['Moneyness'] == 'OTM')]
        if not otm_puts.empty and otm_puts['Delta_Exposure'].sum() < 0:
            scenario_text = """
            **Scenario:** Significant OTM put delta nodes acting as bearish targets.
            There is notable out-of-the-money put delta exposure below the current price, suggesting traders are targeting lower price levels.
            These strikes may act as targets for bearish price movement as market makers adjust their hedges.
            """

    # Display the applicable scenario
    st.markdown("---")
    st.subheader("📖 Detailed Analysis")
    if scenario_text:
        st.markdown(scenario_text)
    else:
        st.markdown("No significant scenarios detected based on the current delta exposure data.")

# Additional Insights Section
st.markdown("---")
st.subheader("📌 Net Delta Exposures")

# Calculate Net Delta Exposures
net_delta_calls = delta_exposure['Delta_Exposure_Call'].sum()
net_delta_puts = delta_exposure['Delta_Exposure_Put'].sum()
overall_net_delta = net_delta_calls + net_delta_puts

# Display Metrics in Columns with Abbreviated Numbers
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "📌 Net Call Delta Exposure",
        f"{abbreviate_number(net_delta_calls)}",
        delta="🔺" if net_delta_calls >= 0 else "🔻",
        delta_color="normal" if net_delta_calls >=0 else "inverse"
    )

with col2:
    st.metric(
        "📌 Net Put Delta Exposure",
        f"{abbreviate_number(net_delta_puts)}",
        delta="🔺" if net_delta_puts >= 0 else "🔻",
        delta_color="normal" if net_delta_puts >=0 else "inverse"
    )

with col3:
    st.metric(
        "📌 Overall Net Delta Exposure",
        f"{abbreviate_number(overall_net_delta)}",
        delta="🔺" if overall_net_delta >=0 else "🔻",
        delta_color="normal" if overall_net_delta >=0 else "inverse"
    )

# Footer with Note
st.markdown("---")
st.markdown(
    "💡 **Note:** Delta hedging exposure represents the sensitivity of option positions to changes in the underlying asset's price. "
    "Positive delta indicates a net long position, while negative delta indicates a net short position."
)
