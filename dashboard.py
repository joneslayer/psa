import streamlit as st
import pandas as pd
import json
import os
import plotly.graph_objects as go
import yfinance as yf
import numpy as np
import datetime

# --- Dashboard Configuration ---
ANALYSIS_FILE = 'latest_analysis.json' # This is where your analyzer.py script saves its results
DASHBOARD_REFRESH_INTERVAL_SECONDS = 10 # How often the dashboard tries to refresh its data

# --- Streamlit Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="My Personal Stock Analysis Dashboard",
    initial_sidebar_state="expanded"
)

st.title("📈 My Personal Stock Analysis Dashboard")

# Placeholder for real-time updates and metrics (top section)
analysis_info_placeholder = st.empty()


# --- Utility Functions for Data Loading and API Calls ---
@st.cache_data(ttl=DASHBOARD_REFRESH_INTERVAL_SECONDS) # Cache data for efficiency and auto-refresh
def load_analysis_results():
    """Loads the latest analysis results from the JSON file."""
    if not os.path.exists(ANALYSIS_FILE):
        # Return a default empty structure if file doesn't exist yet
        return {"timestamp": "N/A", "analysis_date": "N/A", "top_stocks": [], "top_stock_prices": {}}
    
    try:
        with open(ANALYSIS_FILE, 'r') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError:
        st.error("Error reading analysis file. It might be corrupted or empty.")
        return {"timestamp": "N/A", "analysis_date": "N/A", "top_stocks": [], "top_stock_prices": {}}

@st.cache_data(ttl=60*60*4) # Cache yfinance data for 4 hours to avoid hitting API too often
def get_historical_data_for_chart(symbol, period="1y"): # Default to 1 year for better charts
    """Downloads historical stock data using yfinance."""
    try:
        # Fetching daily data ('1d' interval)
        data = yf.download(symbol, period=period, interval="1d", progress=False)
        if data.empty:
            st.warning(f"No historical data found for {symbol} with period {period}.")
            return pd.DataFrame()
        return data
    except Exception as e:
        st.error(f"Could not download historical data for {symbol}: {e}")
        return pd.DataFrame()


# --- Technical Indicator Calculation Functions ---
def compute_rsi(close, period=14):
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan) # Handle division by zero
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(close, fast_period=12, slow_period=26, signal_period=9):
    exp1 = close.ewm(span=fast_period, adjust=False).mean()
    exp2 = close.ewm(span=slow_period, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    histogram = macd - signal
    return macd, signal, histogram

def compute_atr(high, low, close, period=14):
    # Ensure inputs are explicitly 1-dimensional pandas Series, making copies
    high = high.squeeze().copy()
    low = low.squeeze().copy()
    close = close.squeeze().copy()

    # NEW: Check if inputs have any valid data points BEFORE proceeding
    if high.dropna().empty or low.dropna().empty or close.dropna().empty:
        return pd.Series(dtype=float) # Return empty Series if no valid data

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1) # .max(axis=1) on a DataFrame produces a Series

    # Ensure there are non-NaN values after TR calculation
    if tr.dropna().empty:
        return pd.Series(dtype=float)

    atr = tr.rolling(window=period).mean()
    return atr

def compute_adx(high, low, close, period=14):
    # Ensure inputs are explicitly 1-dimensional pandas Series, making copies
    high = high.squeeze().copy()
    low = low.squeeze().copy()
    close = close.squeeze().copy()

    # NEW: Check if inputs have any valid data points BEFORE proceeding
    if high.dropna().empty or low.dropna().empty or close.dropna().empty:
        return pd.Series(dtype=float) # Return empty Series if no valid data

    # --- Robustness check for insufficient data ---
    min_required_len = period * 2 + 5 # For safety with shift and rolling

    if len(high) < min_required_len or len(low) < min_required_len or len(close) < min_required_len:
        return pd.Series(dtype=float) # Not enough data for meaningful ADX calculation

    # Calculate Directional Movement (DM)
    plus_dm = high.diff()
    minus_dm = low.diff().abs()

    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    # Calculate True Range (TR)
    tr_components = pd.DataFrame({
        'h_l': high - low,
        'h_pc': (high - close.shift()).abs(),
        'l_pc': (low - close.shift()).abs()
    })
    
    # Drop rows that are entirely NaN in tr_components, which can happen at the beginning of the series
    tr_components = tr_components.dropna(how='all')

    # If, after dropping NaNs, the dataframe is empty, then we can't compute TR
    if tr_components.empty:
        return pd.Series(dtype=float)

    tr = tr_components.max(axis=1) # Get the max of each row, resulting in a Series

    # Smooth DM and TR using Exponential Moving Average (EWM)
    plus_dm_smooth = plus_dm.ewm(span=period, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(span=period, adjust=False).mean()
    tr_smooth = tr.ewm(span=period, adjust=False).mean()

    # Calculate Directional Indicators (DI)
    plus_di = 100 * (plus_dm_smooth / tr_smooth).replace([np.inf, -np.inf], np.nan)
    minus_di = 100 * (minus_dm_smooth / tr_smooth).replace([np.inf, -np.inf], np.nan)

    # Calculate Directional Movement Index (DX)
    sum_di = plus_di + minus_di
    dx = (abs(plus_di - minus_di) / sum_di).replace([np.inf, -np.inf], np.nan) * 100
    
    # If dx becomes all NaN (e.g., from earlier divisions), return empty Series
    if dx.dropna().empty:
        return pd.Series(dtype=float)

    # Calculate Average Directional Index (ADX) - smoothed DX
    adx = dx.ewm(span=period, adjust=False).mean()
    
    return adx

def compute_vwap(close, volume, period=20):
    # Ensure inputs are explicitly 1-dimensional pandas Series, making copies
    close = close.squeeze().copy()
    volume = volume.squeeze().copy()
    
    # NEW: Critical check - if inputs are all NaN or empty after squeezing, return early
    if close.dropna().empty or volume.dropna().empty:
        return pd.Series(dtype=float) # Return empty Series if no valid close or volume data

    # Create DataFrame to align Close and Volume, then drop any rows with NaNs
    aligned_data = pd.DataFrame({'Close': close, 'Volume': volume}).dropna()

    # If, after dropping NaNs, the aligned data is empty or volume is all zero, return empty Series
    if aligned_data.empty or 'Volume' not in aligned_data or aligned_data['Volume'].sum() == 0:
        return pd.Series(dtype=float)

    # Calculate (Price * Volume)
    pv = aligned_data['Close'] * aligned_data['Volume']
    
    # Calculate cumulative sums for PV and Volume over the rolling window
    cum_pv = pv.rolling(window=period).sum()
    cum_vol = aligned_data['Volume'].rolling(window=period).sum()

    # Calculate VWAP, handling potential division by zero (results in NaN)
    vwap = (cum_vol / cum_vol).replace([np.inf, -np.inf], np.nan) # Corrected to cum_pv / cum_vol
    return vwap


# --- Composite Stock Quality Score Function ---
def calculate_stock_quality_score(latest_close, latest_rsi, latest_macd_hist, latest_vwap, latest_adx):
    """
    Calculates a heuristic 'goodness' score for a stock based on indicator values.
    Returns a score from 0 to 100.
    """
    # Initialize component scores to neutral (0.5 means 50%)
    rsi_score = 0.5
    macd_hist_score = 0.5
    vwap_score = 0.5
    adx_score = 0.5

    # Handle NaN values by treating them as neutral for the score,
    # or by letting the default 0.5 stand if they are NaN.
    if pd.isna(latest_rsi): pass
    elif latest_rsi <= 30: rsi_score = 1.0 # Very oversold: potentially good
    elif latest_rsi >= 70: rsi_score = 0.0 # Very overbought: potentially bad
    elif latest_rsi < 50: rsi_score = 0.5 + ((50 - latest_rsi) / 40 * 0.5) # Leaning good (lower RSI towards 30)
    else: rsi_score = 0.5 - ((latest_rsi - 50) / 40 * 0.5) # Leaning bad (higher RSI towards 70)

    # Note: latest_close == 0 check is safe now as latest_close is guaranteed scalar
    if pd.isna(latest_macd_hist) or pd.isna(latest_close) or latest_close == 0: pass # Handle close == 0 to avoid division by zero
    elif latest_macd_hist > 0:
        macd_hist_score = 0.5 + min(0.5, latest_macd_hist / latest_close * 5)
    else: # latest_macd_hist <= 0
        macd_hist_score = 0.5 - min(0.5, abs(latest_macd_hist) / latest_close * 5)

    if pd.isna(latest_vwap) or pd.isna(latest_close): pass
    elif latest_close > latest_vwap: vwap_score = 1.0 # Price above VWAP: good
    else: vwap_score = 0.0 # Price below VWAP: bad

    if pd.isna(latest_adx): pass
    elif latest_adx >= 25: adx_score = 1.0 # Strong trend: often good for trend-following strategies
    else: adx_score = 0.5 # Weak or no strong trend

    # Simple average of component scores
    overall_score = (rsi_score + macd_hist_score + vwap_score + adx_score) / 4.0

    return overall_score * 100 # Convert to percentage


# --- Main Dashboard Display Function ---
def display_dashboard():
    # Load the latest analysis results from the file
    results = load_analysis_results()

    # --- Header Information ---
    analysis_time_str = datetime.datetime.fromisoformat(results["timestamp"]).strftime('%Y-%m-%d %H:%M:%S') if results["timestamp"] != "N/A" else "N/A"
    analysis_info_placeholder.info(f"**Last Analysis Run:** {analysis_time_str} (Based on data from: {results['analysis_date']})")

    # Use columns for layout of Top Stocks and its chart
    col1, col2 = st.columns([1, 2])

    with col1: # This column will contain Top Stocks and the selector for it
        st.subheader("Top Stock Candidates (from Strategy)")
        if results["top_stocks"]:
            top_stocks_df = pd.DataFrame({
                "Symbol": results["top_stocks"],
                "Current Price": [results["top_stock_prices"].get(s, "N/A") for s in results["top_stocks"]]
            })
            st.dataframe(top_stocks_df, hide_index=True, use_container_width=True)

            selected_chart_symbol_from_top_stocks = st.selectbox(
                "Select a top stock for detailed chart:",
                results["top_stocks"],
                key="top_stock_selector" # Unique key for this selectbox
            )
        else:
            st.info("No top stocks identified yet. Waiting for analysis...")
            selected_chart_symbol_from_top_stocks = None # No symbol to select if list is empty

    with col2: # This column will display the chart based on the top stock selector
        if selected_chart_symbol_from_top_stocks:
            st.subheader(f"Historical Chart for {selected_chart_symbol_from_top_stocks}")
            # Use the existing get_historical_data_for_chart function
            chart_data_top = get_historical_data_for_chart(selected_chart_symbol_from_top_stocks, period="1y")

            # --- Charting logic for Top Stock Candidates ---
            # Add a data point check for plotting indicators on top stocks too
            MIN_DATA_POINTS_FOR_TOP_CHART = 40 # Minimum for all indicators
            if not chart_data_top.empty and len(chart_data_top) >= MIN_DATA_POINTS_FOR_TOP_CHART:
                # Calculate indicators for the selected top stock's chart
                chart_data_top['RSI'] = compute_rsi(chart_data_top['Close'])
                macd, signal, hist = compute_macd(chart_data_top['Close'])
                chart_data_top['MACD'] = macd
                chart_data_top['MACD_Signal'] = signal
                chart_data_top['MACD_Hist'] = hist # Store histogram for consistency if needed later

                # For plotting, ensure indicators are not all NaNs
                show_indicators_on_top_chart = not (chart_data_top['RSI'].dropna().empty or \
                                                    chart_data_top['MACD'].dropna().empty)

                fig_top = go.Figure()
                fig_top.add_trace(go.Candlestick(x=chart_data_top.index,
                                                      open=chart_data_top['Open'],
                                                      high=chart_data_top['High'],
                                                      low=chart_data_top['Low'],
                                                      close=chart_data_top['Close'],
                                                      name='Price'))

                if show_indicators_on_top_chart:
                    # Add RSI subplot
                    fig_top.add_trace(go.Scatter(x=chart_data_top.index, y=chart_data_top['RSI'], mode='lines', name='RSI',
                                             line=dict(color='purple', width=1), yaxis='y2'))

                    # Add MACD subplot
                    fig_top.add_trace(go.Scatter(x=chart_data_top.index, y=chart_data_top['MACD'], mode='lines', name='MACD',
                                             line=dict(color='blue', width=1), yaxis='y3'))
                    fig_top.add_trace(go.Scatter(x=chart_data_top.index, y=chart_data_top['MACD_Signal'], mode='lines', name='MACD Signal',
                                             line=dict(color='orange', width=1), yaxis='y3'))

                    # Update layout for multiple y-axes
                    fig_top.update_layout(
                        xaxis_rangeslider_visible=False,
                        height=600,
                        margin=dict(l=20, r=20, t=30, b=20),
                        yaxis=dict(title='Price', domain=[0.4, 1.0]),
                        yaxis2=dict(title='RSI', domain=[0.2, 0.35], anchor='x', side='right'),
                        yaxis3=dict(title='MACD', domain=[0.0, 0.15], anchor='x', side='right'),
                        hovermode="x unified"
                    )
                else: # Only price chart if indicators can't be plotted
                    st.warning(f"Indicators for {selected_chart_symbol_from_top_stocks} could not be fully calculated for plotting. Displaying price chart only.")
                    fig_top.update_layout(xaxis_rangeslider_visible=True, height=400, margin=dict(l=20, r=20, t=30, b=20), hovermode="x unified")
                
                st.plotly_chart(fig_top, use_container_width=True)

            elif not chart_data_top.empty:
                st.warning(f"Not enough data ({len(chart_data_top)} rows) to plot full indicators for {selected_chart_symbol_from_top_stocks}. Displaying price chart only.")
                fig_top = go.Figure(data=[go.Candlestick(x=chart_data_top.index,
                                                        open=chart_data_top['Open'],
                                                        high=chart_data_top['High'],
                                                        low=chart_data_top['Low'],
                                                        close=chart_data_top['Close'],
                                                        name='Price')])
                fig_top.update_layout(xaxis_rangeslider_visible=True, height=400, margin=dict(l=20, r=20, t=30, b=20), hovermode="x unified")
                st.plotly_chart(fig_top, use_container_width=True)
            else:
                st.warning(f"No chart data available for {selected_chart_symbol_from_top_stocks}.")
        else:
            st.info("Select a stock from the left to view its chart.")

    # --- Individual Stock Analysis Section ---
    st.markdown("---") # A horizontal separator
    st.header("🔍 Analyze Any Stock")

    # Input for a custom stock symbol
    user_custom_symbol = st.text_input("Enter Stock Symbol:", value="AAPL", key="custom_stock_input_main").upper()

    # Button to trigger analysis for the custom symbol
    analyze_button = st.button(f"Analyze {user_custom_symbol}", key="analyze_button")

    if analyze_button and user_custom_symbol: # Only proceed if button is clicked AND symbol is entered
        st.subheader(f"Detailed Analysis for {user_custom_symbol}")

        # Fetch data for the custom symbol (1 year period for better analysis)
        custom_chart_data = get_historical_data_for_chart(user_custom_symbol, period="1y")

        if not custom_chart_data.empty:
            # --- Check for sufficient data points for indicator calculations ---
            MIN_DATA_POINTS = 40 # A robust minimum for all indicators (including ADX's internal needs)
            if len(custom_chart_data) < MIN_DATA_POINTS:
                st.warning(f"Not enough historical data ({len(custom_chart_data)} rows) for **{user_custom_symbol}** to compute all technical indicators. Displaying price chart only. (Need at least {MIN_DATA_POINTS} rows for full analysis)")

                # Display only the basic candlestick chart without indicators
                fig_custom = go.Figure(data=[go.Candlestick(x=custom_chart_data.index,
                                                              open=custom_chart_data['Open'],
                                                              high=custom_chart_data['High'],
                                                              low=custom_chart_data['Low'],
                                                              close=custom_chart_data['Close'],
                                                              name='Price')])
                # Add a rangeslider for basic navigation on shorter charts
                fig_custom.update_layout(xaxis_rangeslider_visible=True, height=400, margin=dict(l=20, r=20, t=30, b=20), hovermode="x unified")
                st.plotly_chart(fig_custom, use_container_width=True)

                st.info("No detailed indicator values or score available due to insufficient data.")

            else: # --- ELSE: Enough data points, proceed with full analysis ---
                # Calculate ALL indicators for the custom stock
                custom_chart_data['RSI'] = compute_rsi(custom_chart_data['Close'])
                macd, signal, hist = compute_macd(custom_chart_data['Close'])
                custom_chart_data['MACD'] = macd
                custom_chart_data['MACD_Signal'] = signal
                custom_chart_data['MACD_Hist'] = hist # Store histogram for easy access

                # Using the revised, more robust indicator functions
                custom_chart_data['ATR'] = compute_atr(custom_chart_data['High'], custom_chart_data['Low'], custom_chart_data['Close'])
                custom_chart_data['ADX'] = compute_adx(custom_chart_data['High'], custom_chart_data['Low'], custom_chart_data['Close'])
                custom_chart_data['VWAP'] = compute_vwap(custom_chart_data['Close'], custom_chart_data['Volume'])

                # Display the chart for the custom stock (with indicators)
                fig_custom = go.Figure(data=[go.Candlestick(x=custom_chart_data.index,
                                                              open=custom_chart_data['Open'],
                                                              high=custom_chart_data['High'],
                                                              low=custom_chart_data['Low'],
                                                              close=custom_chart_data['Close'],
                                                              name='Price')])
                fig_custom.add_trace(go.Scatter(x=custom_chart_data.index, y=custom_chart_data['RSI'], mode='lines', name='RSI', line=dict(color='purple', width=1), yaxis='y2'))
                fig_custom.add_trace(go.Scatter(x=custom_chart_data.index, y=custom_chart_data['MACD'], mode='lines', name='MACD', line=dict(color='blue', width=1), yaxis='y3'))
                fig_custom.add_trace(go.Scatter(x=custom_chart_data.index, y=custom_chart_data['MACD_Signal'], mode='lines', name='MACD Signal', line=dict(color='orange', width=1), yaxis='y3'))

                fig_custom.update_layout(xaxis_rangeslider_visible=False, height=600, margin=dict(l=20, r=20, t=30, b=20),
                                         yaxis=dict(title='Price', domain=[0.4, 1.0]),
                                         yaxis2=dict(title='RSI', domain=[0.2, 0.35], anchor='x', side='right'),
                                         yaxis3=dict(title='MACD', domain=[0.0, 0.15], anchor='x', side='right'),
                                         hovermode="x unified"
                )
                st.plotly_chart(fig_custom, use_container_width=True)

                # Display individual indicator values with NaN handling
                # CRITICAL FIX HERE: Use .item() to ensure scalar values, and check .dropna().empty
                latest_close = custom_chart_data['Close'].iloc[-1].item() if not custom_chart_data['Close'].dropna().empty else np.nan

                last_rsi_val = custom_chart_data['RSI'].iloc[-1].item() if not custom_chart_data['RSI'].dropna().empty else np.nan
                last_macd_hist_val = custom_chart_data['MACD_Hist'].iloc[-1].item() if not custom_chart_data['MACD_Hist'].dropna().empty else np.nan
                last_atr_val = custom_chart_data['ATR'].iloc[-1].item() if not custom_chart_data['ATR'].dropna().empty else np.nan
                latest_vwap = custom_chart_data['VWAP'].iloc[-1].item() if not custom_chart_data['VWAP'].dropna().empty else np.nan
                latest_adx = custom_chart_data['ADX'].iloc[-1].item() if not custom_chart_data['ADX'].dropna().empty else np.nan


                # Display values (now guaranteed to be scalars or np.nan)
                if pd.isna(latest_close): st.write("**Current Close Price:** N/A")
                else: st.write(f"**Current Close Price:** {latest_close:.2f}")

                if pd.isna(last_rsi_val): st.write("**Current RSI:** N/A")
                else: st.write(f"**Current RSI:** {last_rsi_val:.2f}")

                if pd.isna(last_macd_hist_val): st.write("**Current MACD Histogram:** N/A")
                else: st.write(f"**Current MACD Histogram:** {last_macd_hist_val:.2f}")

                if pd.isna(last_atr_val): st.write("**Current ATR:** N/A")
                else: st.write(f"**Current ATR:** {last_atr_val:.2f}")

                if pd.isna(latest_vwap): st.write("**Current VWAP:** N/A")
                else: st.write(f"**Current VWAP:** {latest_vwap:.2f}")

                if pd.isna(latest_adx): st.write("**Current ADX:** N/A")
                else: st.write(f"**Current ADX:** {latest_adx:.2f}")


                # Calculate and display the composite score
                composite_score = calculate_stock_quality_score(
                    latest_close,
                    last_rsi_val,
                    last_macd_hist_val,
                    latest_vwap,
                    latest_adx
                )

                if pd.isna(composite_score):
                    st.markdown("---")
                    st.error("Cannot compute overall score: Not enough data for all indicators, or critical data is missing.")
                else:
                    st.markdown("---")
                    st.subheader("Overall Stock Quality Score")
                    # Dynamic color for the score
                    color = 'green'
                    if composite_score < 40:
                        color = 'red'
                    elif composite_score < 60:
                        color = 'orange'
                    st.markdown(f"**Based on current indicators: ** <span style='font-size: 30px; color: {color}; font-weight: bold;'>{composite_score:.2f}%</span>", unsafe_allow_html=True)
                    st.info("*(This score is a simplified heuristic based on predefined rules for RSI, MACD Histogram, VWAP, and ADX, and is for informational purposes only. It is not financial advice.)*")

        else: # custom_chart_data is empty
            st.warning(f"No historical data found for **{user_custom_symbol}**. Please check the symbol or try a different one.")
    elif analyze_button: # If button is clicked but no symbol entered
        st.warning("Please enter a stock symbol to analyze.")


# --- Main Application Entry Point (Streamlit runs this once) ---
if __name__ == "__main__":
    display_dashboard()