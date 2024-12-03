import streamlit as st
import yfinance as yf
from scipy.stats import norm
import math
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots  # Added import for make_subplots

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
    st.markdown("### Stock Information")
    
    # Directly placed the ticker input without expander
    ticker = st.text_input(
        "Stock Ticker",
        value="NVDA",
        help="Enter the ticker symbol of the stock you want to analyze (e.g., NVDA for NVIDIA Corporation)."
    )
    
    st.markdown("---")
    
    st.subheader("📊 Volatility & Rates")
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
    
    st.markdown("---")
    
    st.subheader("📅 Expiry & Strikes")
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

    dte_range = st.slider(
        "Days to Expiry (DTE)",
        0,
        365,
        (0, 50),
        1,
        help=(
            "Select the range of days to expiry for the options contracts to include in the analysis. "
            "Default is 0 to 50 days."
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

# Calculate Days to Expiry (DTE)
expiry_dates_dte = [
    (expiry_date, (pd.to_datetime(expiry_date) - pd.Timestamp.now()).days)
    for expiry_date in expiry_dates
]

# Filter expiry dates based on DTE range
filtered_expiry_dates = [
    date for date, dte in expiry_dates_dte if dte_range[0] <= dte <= dte_range[1]
]

if not filtered_expiry_dates:
    st.error("❌ No options contracts available within the specified DTE range.")
    st.stop()

# Use the first expiry date from the filtered list by default
expiry_date = filtered_expiry_dates[0]

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
filtered_options = options[(options['strike'] >= lower_bound) & (options['strike'] <= upper_bound)].copy()

# Verify if 'openInterest' exists
if 'openInterest' not in filtered_options.columns:
    st.error("❌ The fetched options data does not contain 'openInterest'. Please verify the data source.")
    st.stop()

# Ensure 'openInterest' is numeric and handle missing values
filtered_options.loc[:, 'openInterest'] = pd.to_numeric(filtered_options['openInterest'], errors='coerce').fillna(0)

# Calculate Delta Exposure Separately for Calls and Puts
# Multiply by 100 to account for contract size
filtered_options.loc[:, 'Delta_Exposure'] = filtered_options.apply(
    lambda row: row['Delta'] * row['openInterest'] * 100, axis=1
)

# Separate Delta Exposure for Calls and Puts
delta_exposure_calls = filtered_options[filtered_options['Type'] == 'Call'].groupby('strike')['Delta_Exposure'].sum().reset_index()
delta_exposure_puts = filtered_options[filtered_options['Type'] == 'Put'].groupby('strike')['Delta_Exposure'].sum().reset_index()

# Merge Calls and Puts Delta Exposure
delta_exposure = pd.merge(delta_exposure_calls, delta_exposure_puts, on='strike', how='outer', suffixes=('_Call', '_Put')).fillna(0)

# Tabs for Organized Display
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
    
    # Update the layout to have reversed x-axis (higher to lower) and normal y-axis (lower to higher)
    fig.update_layout(
        xaxis=dict(
            title="Delta Exposure",
            range=[x_limit, -x_limit],  # Reversed from higher to lower
            tickmode='linear',
            tick0=-x_limit,
            dtick=x_limit / 4,
            tickformat=".2s",
            tickvals=np.linspace(x_limit, -x_limit, num=5),
            ticktext=[format_ticks(val, 'x') for val in np.linspace(x_limit, -x_limit, num=5)]
        ),
        yaxis_title="Strike Price",
        yaxis=dict(
            tickmode='linear',
            tick0=math.floor(lower_bound / 5) * 5,  # Adjust tick0 to nearest multiple of 5
            dtick=math.ceil((upper_bound - lower_bound) / 10 / 5) * 5,  # Adjust dtick to nearest multiple of 5
            # Removed 'autorange' to let Plotly handle it automatically
        ),
        template='plotly_white',
        hovermode='closest',
        height=700 if st.session_state.get("is_desktop", True) else 500,  # Adjust height based on device
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
        x0=x_limit,
        y0=S,
        x1=-x_limit,
        y1=S,
        line=dict(color='green', width=2, dash='dot'),
    )
    
    # Add annotation for the current stock price line
    fig.add_annotation(
        x=-x_limit * 0.95,  # Adjusted position due to reversed x-axis
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
    ).copy()
    
    # Convert Delta Exposures to integers for abbreviation
    styled_df['Call Delta Exposure'] = styled_df['Call Delta Exposure'].astype(int).apply(abbreviate_number)
    styled_df['Put Delta Exposure'] = styled_df['Put Delta Exposure'].astype(int).apply(abbreviate_number)
    
    # Make the DataFrame horizontally scrollable for mobile
    st.markdown(
        """
        <style>
        .dataframe {
            width: 100%;
            overflow-x: auto;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
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

    filtered_options.loc[:, 'Moneyness'] = filtered_options.apply(get_moneyness, axis=1)

    # Group by Type and Moneyness
    grouped = filtered_options.groupby(['Type', 'Moneyness'])

    # Sum delta exposures
    delta_exposure_summary = grouped['Delta_Exposure'].sum().reset_index()

    # Calculate total delta exposures
    total_call_itm_delta = delta_exposure_summary[
        (delta_exposure_summary['Type'] == 'Call') & (delta_exposure_summary['Moneyness'] == 'ITM')
    ]['Delta_Exposure'].sum()
    
    total_put_itm_delta = delta_exposure_summary[
        (delta_exposure_summary['Type'] == 'Put') & (delta_exposure_summary['Moneyness'] == 'ITM')
    ]['Delta_Exposure'].sum()
    
    total_otm_call_delta = delta_exposure_summary[
        (delta_exposure_summary['Type'] == 'Call') & (delta_exposure_summary['Moneyness'] == 'OTM')
    ]['Delta_Exposure'].sum()
    
    total_otm_put_delta = delta_exposure_summary[
        (delta_exposure_summary['Type'] == 'Put') & (delta_exposure_summary['Moneyness'] == 'OTM')
    ]['Delta_Exposure'].sum()

    # Initialize lists for scenarios
    support_levels = []
    resistance_levels = []
    bullish_targets = []
    bearish_targets = []

    # 1. Support Levels (Significant ITM Calls)
    itm_calls = filtered_options[
        (filtered_options['Type'] == 'Call') & 
        (filtered_options['Moneyness'] == 'ITM') & 
        (filtered_options['Delta_Exposure'] > 0)
    ]
    if not itm_calls.empty:
        threshold_call = itm_calls['Delta_Exposure'].quantile(0.75)
        significant_itm_calls = itm_calls[itm_calls['Delta_Exposure'] >= threshold_call]
        if not significant_itm_calls.empty:
            support_levels = sorted(significant_itm_calls['strike'].tolist())

    # 2. Resistance Levels (Significant ITM Puts)
    itm_puts = filtered_options[
        (filtered_options['Type'] == 'Put') & 
        (filtered_options['Moneyness'] == 'ITM') & 
        (filtered_options['Delta_Exposure'] < 0)
    ]
    if not itm_puts.empty:
        threshold_put = itm_puts['Delta_Exposure'].abs().quantile(0.75)
        significant_itm_puts = itm_puts[itm_puts['Delta_Exposure'].abs() >= threshold_put]
        if not significant_itm_puts.empty:
            resistance_levels = sorted(significant_itm_puts['strike'].tolist(), reverse=True)

    # 3. Bullish Targets (Significant OTM Calls)
    otm_calls = filtered_options[
        (filtered_options['Type'] == 'Call') & 
        (filtered_options['Moneyness'] == 'OTM') & 
        (filtered_options['Delta_Exposure'] > 0)
    ]
    if not otm_calls.empty:
        threshold_bullish = otm_calls['Delta_Exposure'].quantile(0.75)
        significant_otm_calls = otm_calls[otm_calls['Delta_Exposure'] >= threshold_bullish]
        if not significant_otm_calls.empty:
            bullish_targets = sorted(significant_otm_calls['strike'].tolist())

    # 4. Bearish Targets (Significant OTM Puts)
    otm_puts = filtered_options[
        (filtered_options['Type'] == 'Put') & 
        (filtered_options['Moneyness'] == 'OTM') & 
        (filtered_options['Delta_Exposure'] < 0)
    ]
    if not otm_puts.empty:
        threshold_bearish = otm_puts['Delta_Exposure'].abs().quantile(0.75)
        significant_otm_puts = otm_puts[otm_puts['Delta_Exposure'].abs() >= threshold_bearish]
        if not significant_otm_puts.empty:
            bearish_targets = sorted(significant_otm_puts['strike'].tolist(), reverse=True)

    # Identify Primary and Secondary Resistances
    primary_resistance = resistance_levels[0] if len(resistance_levels) > 0 else None
    secondary_resistance = resistance_levels[1] if len(resistance_levels) > 1 else None

    # Identify Primary and Secondary Supports
    primary_support = support_levels[0] if len(support_levels) > 0 else None
    secondary_support = support_levels[1] if len(support_levels) > 1 else None

    # Calculate Relative Strengths for Resistances
    resistance_strengths = []
    if primary_resistance:
        strength = filtered_options[
            (filtered_options['strike'] == primary_resistance) & 
            (filtered_options['Type'] == 'Put')
        ]['Delta_Exposure'].abs().sum()
        resistance_strengths.append({'Level': primary_resistance, 'Strength': strength})
    if secondary_resistance:
        strength = filtered_options[
            (filtered_options['strike'] == secondary_resistance) & 
            (filtered_options['Type'] == 'Put')
        ]['Delta_Exposure'].abs().sum()
        resistance_strengths.append({'Level': secondary_resistance, 'Strength': strength})

    # Calculate Relative Strengths for Supports
    support_strengths = []
    if primary_support:
        strength = filtered_options[
            (filtered_options['strike'] == primary_support) & 
            (filtered_options['Type'] == 'Call')
        ]['Delta_Exposure'].sum()
        support_strengths.append({'Level': primary_support, 'Strength': strength})
    if secondary_support:
        strength = filtered_options[
            (filtered_options['strike'] == secondary_support) & 
            (filtered_options['Type'] == 'Call')
        ]['Delta_Exposure'].sum()
        support_strengths.append({'Level': secondary_support, 'Strength': strength})

    # Create DataFrames for Resistances and Supports
    resistances_df = pd.DataFrame(resistance_strengths)
    supports_df = pd.DataFrame(support_strengths)

    # Add Priority Labels
    if not resistances_df.empty:
        resistances_df = resistances_df.reset_index(drop=True)
        priorities_res = ['Primary', 'Secondary'][:len(resistances_df)]
        resistances_df.insert(0, 'Priority', priorities_res)

    if not supports_df.empty:
        supports_df = supports_df.reset_index(drop=True)
        priorities_sup = ['Primary', 'Secondary'][:len(supports_df)]
        supports_df.insert(0, 'Priority', priorities_sup)

    # Display Current Price with Enhanced Visibility
    st.markdown(f"### **Current Stock Price:** ${S:.2f}", unsafe_allow_html=False)

    # Layout for Support and Resistance Tables Side by Side on Desktop and Stacked on Mobile
    st.markdown("### 📌 Support and Resistance Levels")
    res_col, sup_col = st.columns(2) if st.session_state.get("is_desktop", True) else st.columns(1)

    with res_col:
        st.markdown("#### Resistances")
        if not resistances_df.empty:
            st.table(resistances_df.style.format({"Level": "${:,.2f}", "Strength": "{:,.0f}"}))
        else:
            st.write("No significant resistance levels identified.")

    with sup_col:
        st.markdown("#### Supports")
        if not supports_df.empty:
            st.table(supports_df.style.format({"Level": "${:,.2f}", "Strength": "{:,.0f}"}))
        else:
            st.write("No significant support levels identified.")

    # Layout for Visualizations
    col1, col2 = st.columns(2) if st.session_state.get("is_desktop", True) else st.columns(1)

    with col1:
        # Bar Chart for Resistances and Supports
        fig_strengths = go.Figure()

        if not resistances_df.empty:
            fig_strengths.add_trace(
                go.Bar(
                    x=resistances_df['Level'],
                    y=resistances_df['Strength'],
                    name='Resistance',
                    marker_color='rgba(255, 99, 132, 0.7)',  # Soft Red
                    text=[abbreviate_number(val) for val in resistances_df['Strength']],
                    textposition='auto'
                )
            )
        if not supports_df.empty:
            fig_strengths.add_trace(
                go.Bar(
                    x=supports_df['Level'],
                    y=supports_df['Strength'],
                    name='Support',
                    marker_color='rgba(75, 192, 192, 0.7)',  # Soft Teal
                    text=[abbreviate_number(val) for val in supports_df['Strength']],
                    textposition='auto'
                )
            )

        fig_strengths.update_layout(
            title="📊 Relative Strength of Supports and Resistances",
            xaxis_title="Price Level",
            yaxis_title="Relative Strength (Delta Exposure)",
            barmode='group',
            template='plotly_white',
            height=400 if st.session_state.get("is_desktop", True) else 300,  # Adjust height based on device
            margin=dict(l=50, r=50, t=50, b=50)
        )

        st.plotly_chart(fig_strengths, use_container_width=True)

    with col2:
        # Gauge Charts for Current Price Proximity to Primary Resistance and Support
        gauges = []
        if primary_resistance:
            proximity_res = (S / primary_resistance) * 100
            proximity_res = min(proximity_res, 100)  # Cap at 100%
            gauges.append(go.Indicator(
                mode="gauge+number",
                value=proximity_res,
                title={'text': f"🔍 Proximity to Primary Resistance (${primary_resistance:.2f})"},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkred"},
                    'steps': [
                        {'range': [0, 50], 'color': "#FFDDDD"},
                        {'range': [50, 75], 'color': "#FFAAAA"},
                        {'range': [75, 100], 'color': "#FF0000"},
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': proximity_res
                    }
                }
            ))
        if primary_support:
            proximity_sup = (primary_support / S) * 100
            proximity_sup = min(proximity_sup, 100)  # Cap at 100%
            gauges.append(go.Indicator(
                mode="gauge+number",
                value=proximity_sup,
                title={'text': f"🔍 Proximity to Primary Support (${primary_support:.2f})"},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkgreen"},
                    'steps': [
                        {'range': [0, 50], 'color': "#DDFFDD"},
                        {'range': [50, 75], 'color': "#AAFFAA"},
                        {'range': [75, 100], 'color': "#00FF00"},
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': proximity_sup
                    }
                }
            ))

        if gauges:
            cols = len(gauges)
            if st.session_state.get("is_desktop", True):
                fig_gauges = make_subplots(rows=1, cols=cols, specs=[[{'type': 'indicator'}]*cols])
                for i, gauge in enumerate(gauges):
                    fig_gauges.add_trace(gauge, row=1, col=i+1)

                fig_gauges.update_layout(
                    title="📊 Proximity to Key Levels",
                    template='plotly_white',
                    height=400 if st.session_state.get("is_desktop", True) else 300,
                    margin=dict(l=50, r=50, t=50, b=50)
                )
            else:
                # For mobile, stack the gauges vertically
                fig_gauges = make_subplots(rows=cols, cols=1, specs=[[{'type': 'indicator'}]]*cols)
                for i, gauge in enumerate(gauges):
                    fig_gauges.add_trace(gauge, row=i+1, col=1)

                fig_gauges.update_layout(
                    title="📊 Proximity to Key Levels",
                    template='plotly_white',
                    height=300 * cols if cols > 1 else 300,
                    margin=dict(l=50, r=50, t=50, b=50)
                )

            st.plotly_chart(fig_gauges, use_container_width=True)
        else:
            st.write("No primary resistance or support levels to display proximity.")

    # Dynamic Description Based on Scenarios
    st.markdown("### 📝 Market Analysis Summary")
    analysis = ""

    # Calculate total support and resistance strengths
    total_resistance_strength = resistances_df['Strength'].sum() if not resistances_df.empty else 0
    total_support_strength = supports_df['Strength'].sum() if not supports_df.empty else 0

    # Ratios
    if total_resistance_strength > 0:
        support_resistance_ratio = total_support_strength / total_resistance_strength
    else:
        support_resistance_ratio = float('inf') if total_support_strength > 0 else 0

    # Dynamic Insights
    if support_resistance_ratio > 1.5:
        analysis += "- **Bullish Sentiment:** Strong support levels indicate significant buying interest, suggesting potential upward movement.\n"
    elif support_resistance_ratio < 0.75:
        analysis += "- **Bearish Sentiment:** Strong resistance levels indicate significant selling pressure, suggesting potential downward movement.\n"
    else:
        analysis += "- **Neutral Sentiment:** Balanced support and resistance levels indicate a possible consolidation phase.\n"

    # Proximity Analysis
    if primary_resistance and (S >= primary_resistance * 0.95):
        analysis += "- **Approaching Resistance:** The current price is nearing the primary resistance level, which might act as a ceiling.\n"
    if primary_support and (S <= primary_support * 1.05):
        analysis += "- **Approaching Support:** The current price is nearing the primary support level, which might act as a floor.\n"

    # Actionable Recommendations
    if support_resistance_ratio > 1.5:
        analysis += "- **Recommendation:** Consider bullish strategies such as buying calls or entering long positions near support levels.\n"
    elif support_resistance_ratio < 0.75:
        analysis += "- **Recommendation:** Consider bearish strategies such as buying puts or entering short positions near resistance levels.\n"
    else:
        analysis += "- **Recommendation:** Monitor the stock for breakouts or breakdowns beyond support and resistance levels.\n"

    # Color Coding Based on Sentiment
    if support_resistance_ratio > 1.5:
        # Bullish Sentiment - Green
        st.markdown("### 🟢 **Bullish Sentiment Detected**")
    elif support_resistance_ratio < 0.75:
        # Bearish Sentiment - Red
        st.markdown("### 🔴 **Bearish Sentiment Detected**")
    else:
        # Neutral Sentiment - Yellow
        st.markdown("### 🟡 **Neutral Sentiment Detected**")

    st.markdown(analysis)

    # Actionable Summary
    st.markdown("### 📌 Actionable Summary")
    summary = ""

    if support_resistance_ratio > 1.5:
        summary += "The market exhibits **strong bullish sentiment** with support levels providing a solid foundation for potential price increases. "
        if primary_resistance and (S < primary_resistance):
            summary += "Monitoring the approach towards the primary resistance can provide opportunities to enter bullish positions before potential breakouts.\n"
    elif support_resistance_ratio < 0.75:
        summary += "The market exhibits **strong bearish sentiment** with resistance levels posing significant barriers to price increases. "
        if primary_support and (S > primary_support):
            summary += "Monitoring the approach towards the primary support can provide opportunities to enter bearish positions before potential breakdowns.\n"
    else:
        summary += "The market is in a **neutral phase** with balanced support and resistance levels. "
        summary += "It's advisable to watch for breakouts or breakdowns beyond these levels to determine the next trend direction.\n"

    # Suggest Buy and Sell Ranges
    buy_range = ""
    sell_range = ""

    if not supports_df.empty:
        primary_sup = primary_support
        secondary_sup = secondary_support if secondary_support else primary_sup
        buy_low = primary_sup * 0.98  # 2% below primary support
        buy_high = primary_sup * 1.02  # 2% above primary support
        buy_range = f"${buy_low:.2f} - ${buy_high:.2f}"
        summary += f"- **Suggested Buy Range:** Consider buying between **${buy_low:.2f}** and **${buy_high:.2f}** near the primary support level.\n"
    
    if not resistances_df.empty:
        primary_res = primary_resistance
        secondary_res = secondary_resistance if len(resistance_levels) >1 else primary_resistance
        sell_low = primary_res * 0.98  # 2% below primary resistance
        sell_high = primary_res * 1.02  # 2% above primary resistance
        sell_range = f"${sell_low:.2f} - ${sell_high:.2f}"
        summary += f"- **Suggested Sell Range:** Consider selling between **${sell_low:.2f}** and **${sell_high:.2f}** near the primary resistance level.\n"

    st.markdown(summary)

    # Additional Insights
    st.markdown("---")
    st.markdown("### 📈 Additional Insights")
    st.markdown(
        "- **Primary Resistance:** The most significant price level where selling pressure is expected to be strong enough to prevent the price from increasing further."
    )
    st.markdown(
        "- **Secondary Resistance:** The second most significant resistance level, indicating another potential price ceiling."
    )
    st.markdown(
        "- **Primary Support:** The most significant price level where buying interest is expected to be strong enough to prevent the price from decreasing further."
    )
    st.markdown(
        "- **Secondary Support:** The second most significant support level, indicating another potential price floor."
    )
    st.markdown(
        "- **Relative Strength:** Indicates the magnitude of delta exposure at each support and resistance level, helping to assess the strength of each level."
    )
    st.markdown(
        "- **Buy Range:** Suggested price range near support levels where buying interest is strong.\n"
        "- **Sell Range:** Suggested price range near resistance levels where selling pressure is strong."
    )

# Additional Insights Section
st.markdown("---")
st.subheader("📌 Net Delta Exposures")

# Calculate Net Delta Exposures
net_delta_calls = delta_exposure['Delta_Exposure_Call'].sum()
net_delta_puts = delta_exposure['Delta_Exposure_Put'].sum()
overall_net_delta = net_delta_calls + net_delta_puts

# Display Metrics in Columns with Abbreviated Numbers
col1, col2, col3 = st.columns(3) if st.session_state.get("is_desktop", True) else st.columns(1)

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
