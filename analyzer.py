import pandas as pd
import numpy as np
import yfinance as yf
import time
import datetime
import logging
import json

# --- Configuration (You can change these!) ---
STOCK_SYMBOLS = ['MSFT', 'AAPL', 'GOOG', 'INFY.NS', 'TCS.NS'] # Add your desired symbols
UPDATE_INTERVAL_SECONDS = 300 # Check for new data every 300 seconds (5 minutes)
N_STOCKS_SELECT = 5 # How many top stocks to select

# --- Logging Setup (for better messages) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Technical Indicators ---
def compute_atr(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def compute_adx(high, low, close, period=14):
    plus_dm = high.diff()
    minus_dm = low.diff().abs()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    tr = tr.rolling(window=period).sum()

    plus_di = 100 * (plus_dm.rolling(window=period).sum() / tr)
    minus_di = 100 * (minus_dm.rolling(window=period).sum() / tr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=period).mean()
    return adx

def compute_vwap(close, volume, period=20):
    aligned_data = pd.DataFrame({'Close': close, 'Volume': volume}).dropna()
    if aligned_data.empty: return pd.Series(dtype=float)
    vwap = (aligned_data['Close'] * aligned_data['Volume']).rolling(window=period).sum() / aligned_data['Volume'].rolling(window=period).sum()
    return vwap

def compute_rsi(close, period=14):
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(close, fast_period=12, slow_period=26, signal_period=9):
    exp1 = close.ewm(span=fast_period, adjust=False).mean()
    exp2 = close.ewm(span=slow_period, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    histogram = macd - signal
    return macd, signal, histogram

# --- Trading Strategy ---
class TradingStrategy:
    def __init__(self, n_stocks_select=10):
        self.n_stocks_select = n_stocks_select

    def select_portfolio(self, current_date, tech_scores, fund_scores, qual_scores):
        try:
            tech_mean = tech_scores.loc[current_date]
        except KeyError:
            available_dates = tech_scores.index[tech_scores.index <= current_date]
            if not available_dates.empty:
                tech_mean = tech_scores.loc[available_dates[-1]]
            else:
                tech_mean = pd.Series(0.5, index=tech_scores.columns)

        try:
            qual_mean = qual_scores.loc[current_date]
        except KeyError:
            available_dates = qual_scores.index[qual_scores.index <= current_date]
            if not available_dates.empty:
                qual_mean = qual_scores.loc[available_dates[-1]]
            else:
                qual_mean = pd.Series(0.5, index=qual_scores.columns)

        if fund_scores.empty:
            fund_mean = pd.Series(0.5, index=tech_scores.columns)
        elif isinstance(fund_scores, pd.DataFrame):
            fund_mean = fund_scores.iloc[0].fillna(0)
        else:
            fund_mean = fund_scores.fillna(0)

        tech_mean = tech_mean.fillna(0.5)
        fund_mean = fund_mean.fillna(0.5)
        qual_mean = qual_mean.fillna(0.5)

        def safe_normalize(series):
            min_val = series.min()
            max_val = series.max()
            if max_val > min_val:
                return (series - min_val) / (max_val - min_val)
            else:
                return pd.Series(0.5, index=series.index)

        tech_norm = safe_normalize(tech_mean)
        fund_norm = safe_normalize(fund_mean)
        qual_norm = safe_normalize(qual_mean)

        common_symbols = tech_norm.index.intersection(fund_norm.index).intersection(qual_norm.index)
        if common_symbols.empty:
            logging.warning(f"No common symbols for score alignment on {current_date}.")
            return []

        tech_norm = tech_norm.reindex(common_symbols)
        fund_norm = fund_norm.reindex(common_symbols)
        qual_norm = qual_norm.reindex(common_symbols)

        combined_score = tech_norm + fund_norm + qual_norm
        combined_score = combined_score.fillna(0)

        if combined_score.empty:
            return []

        top_symbols = combined_score.sort_values(ascending=False).head(self.n_stocks_select).index.tolist()
        return top_symbols

# --- Real-time Data Fetching and Analysis Logic ---
def run_realtime_analysis():
    logging.info("--- Starting Real-time Analysis Cycle ---")
    current_time = datetime.datetime.now()
    logging.info(f"Analysis triggered at: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        data = yf.download(STOCK_SYMBOLS, period="60d", interval="1d", group_by='ticker', progress=False)

        if data.empty:
            logging.warning("No data downloaded. Skipping analysis.")
            return

        closes = pd.DataFrame()
        highs = pd.DataFrame()
        lows = pd.DataFrame()
        volumes = pd.DataFrame()

        for symbol in STOCK_SYMBOLS:
            if symbol in data.columns:
                if len(STOCK_SYMBOLS) > 1:
                    symbol_data = data[symbol]
                else:
                    symbol_data = data

                closes[symbol] = symbol_data['Close']
                highs[symbol] = symbol_data['High']
                lows[symbol] = symbol_data['Low']
                volumes[symbol] = symbol_data['Volume']
            else:
                logging.warning(f"No data available for {symbol}. Skipping for this cycle.")

        closes = closes.dropna(axis=1, how='all')
        highs = highs.dropna(axis=1, how='all')
        lows = lows.dropna(axis=1, how='all')
        volumes = volumes.dropna(axis=1, how='all')

        if closes.empty:
            logging.warning("No valid stock data found after filtering. Skipping analysis.")
            return

        tech_scores_current = pd.DataFrame(index=closes.index, columns=closes.columns, dtype=float)

        for symbol in closes.columns:
            stock_closes = closes[symbol]
            stock_highs = highs[symbol]
            stock_lows = lows[symbol]
            stock_volumes = volumes[symbol]

            atr = compute_atr(stock_highs, stock_lows, stock_closes)
            adx = compute_adx(stock_highs, stock_lows, stock_closes)
            vwap = compute_vwap(stock_closes, stock_volumes)
            rsi = compute_rsi(stock_closes)
            macd, signal, hist = compute_macd(stock_closes)

            temp_tech_df = pd.DataFrame({
                'ATR_Inv': atr,
                'ADX': adx,
                'VWAP_Ratio': stock_closes / vwap,
                'RSI': rsi,
                'MACD_Hist': hist
            }).fillna(0)

            normalized_indicators = pd.DataFrame(dtype=float, index=temp_tech_df.index)
            for col in temp_tech_df.columns:
                min_val = temp_tech_df[col].min()
                max_val = temp_tech_df[col].max()
                if max_val > min_val:
                    normalized_indicators[col] = (temp_tech_df[col] - min_val) / (max_val - min_val)
                else:
                    normalized_indicators[col] = 0.5

            tech_scores_current[symbol] = normalized_indicators.mean(axis=1)
            tech_scores_current[symbol] = tech_scores_current[symbol].rolling(window=5).mean()
            tech_scores_current[symbol] = tech_scores_current[symbol].fillna(method='bfill').fillna(0.5)

        fund_scores_mock = pd.Series(np.random.rand(len(closes.columns)), index=closes.columns)
        qual_scores_mock = pd.DataFrame(np.random.rand(len(closes.index), len(closes.columns)),
                                        index=closes.index, columns=closes.columns)

        latest_date_with_data = closes.index[-1]

        my_strategy = TradingStrategy(n_stocks_select=N_STOCKS_SELECT)

        top_stocks = my_strategy.select_portfolio(
            latest_date_with_data,
            tech_scores_current,
            fund_scores_mock,
            qual_scores_mock
        )

        # 5. Display Results / Insights & Save for Dashboard
        logging.info(f"--- Analysis Results for {latest_date_with_data.strftime('%Y-%m-%d')} ---")

        analysis_results = {
            "timestamp": datetime.datetime.now().isoformat(),
            "analysis_date": latest_date_with_data.strftime('%Y-%m-%d'),
            "top_stocks": [],
            "top_stock_prices": {}
        }

        if top_stocks:
            logging.info(f"Top {N_STOCKS_SELECT} stocks identified: {', '.join(top_stocks)}")
            analysis_results["top_stocks"] = top_stocks

            current_prices = closes.loc[latest_date_with_data, top_stocks]
            analysis_results["top_stock_prices"] = current_prices.to_dict()

            logging.info(f"Current prices: {current_prices.to_string()}")
        else:
            logging.info("No top stocks identified based on current data.")
        logging.info("----------------------------------")

        try:
            with open('latest_analysis.json', 'w') as f:
                json.dump(analysis_results, f, indent=4)
            logging.info("Analysis results saved to latest_analysis.json")
        except Exception as e:
            logging.error(f"Failed to save analysis results to file: {e}", exc_info=True)

    except Exception as e:
        logging.error(f"An error occurred during real-time analysis: {e}", exc_info=True)

# --- Main Application Loop ---
if __name__ == "__main__":
    logging.info(f"Starting real-time stock analyzer. Checking every {UPDATE_INTERVAL_SECONDS} seconds...")
    logging.info("Press Ctrl+C to stop the application.")

    while True:
        run_realtime_analysis()
        logging.info(f"Next analysis in {UPDATE_INTERVAL_SECONDS} seconds...")
        time.sleep(UPDATE_INTERVAL_SECONDS)
