"""
Nifty Option Screener v6.0 — 100% SELLER'S PERSPECTIVE + MOMENT DETECTOR + AI ANALYSIS + EXPIRY SPIKE DETECTOR
EVERYTHING interpreted from Option Seller/Market Maker viewpoint
CALL building = BEARISH (sellers selling calls, expecting price to stay below)
PUT building = BULLISH (sellers selling puts, expecting price to stay above)

NEW FEATURES ADDED:
1. Momentum Burst Detection
2. Orderbook Pressure Analysis
3. Gamma Cluster Concentration
4. OI Velocity/Acceleration
5. Telegram Signal Generation
6. AI-Powered Market Analysis (Perplexity)
7. EXPIRY SPIKE DETECTOR (NEW)
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import pytz
from math import log, sqrt
from scipy.stats import norm
import os
import json

# -----------------------
#  IST TIMEZONE SETUP
# -----------------------
IST = pytz.timezone('Asia/Kolkata')

def get_ist_now():
    return datetime.now(IST)

def get_ist_time_str():
    return get_ist_now().strftime("%H:%M:%S")

def get_ist_date_str():
    return get_ist_now().strftime("%Y-%m-%d")

def get_ist_datetime_str():
    return get_ist_now().strftime("%Y-%m-%d %H:%M:%S")

# -----------------------
#  CONFIG
# -----------------------
AUTO_REFRESH_SEC = 60
LOT_SIZE = 50
RISK_FREE_RATE = 0.06
ATM_STRIKE_WINDOW = 8
SCORE_WEIGHTS = {"chg_oi": 2.0, "volume": 0.5, "oi": 0.2, "iv": 0.3}
BREAKOUT_INDEX_WEIGHTS = {"atm_oi_shift": 0.4, "winding_balance": 0.3, "vol_oi_div": 0.2, "gamma_pressure": 0.1}
SAVE_INTERVAL_SEC = 300

# NEW: Moment detector weights
MOMENT_WEIGHTS = {
    "momentum_burst": 0.40,        # Vol × IV × |ΔOI|
    "orderbook_pressure": 0.20,    # buy/sell depth imbalance
    "gamma_cluster": 0.25,         # ATM ±2 gamma concentration
    "oi_acceleration": 0.15        # OI speed-up (break/hold)
}

TIME_WINDOWS = {
    "morning": {"start": (9, 15), "end": (10, 30), "label": "Morning (09:15-10:30 IST)"},
    "mid": {"start": (10, 30), "end": (12, 30), "label": "Mid (10:30-12:30 IST)"},
    "afternoon": {"start": (14, 0), "end": (15, 30), "label": "Afternoon (14:00-15:30 IST)"},
    "evening": {"start": (15, 0), "end": (15, 30), "label": "Evening (15:00-15:30 IST)"}
}

# -----------------------
#  UTILITY FUNCTIONS
# -----------------------
def safe_int(x, default=0):
    try:
        return int(x)
    except:
        return default

def safe_float(x, default=np.nan):
    try:
        return float(x)
    except:
        return default

def strike_gap_from_series(series):
    diffs = series.sort_values().diff().dropna()
    if diffs.empty:
        return 50
    mode = diffs.mode()
    return int(mode.iloc[0]) if not mode.empty else int(diffs.median())

# Black-Scholes Greeks
def bs_d1(S, K, r, sigma, tau):
    if sigma <= 0 or tau <= 0:
        return 0.0
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))

def bs_delta(S, K, r, sigma, tau, option_type="call"):
    if tau <= 0 or sigma <= 0:
        return 1.0 if (option_type=="call" and S>K) else (-1.0 if (option_type=="put" and S<K) else 0.0)
    d1 = bs_d1(S,K,r,sigma,tau)
    if option_type == "call":
        return norm.cdf(d1)
    return -norm.cdf(-d1)

def bs_gamma(S, K, r, sigma, tau):
    if sigma <= 0 or tau <= 0:
        return 0.0
    d1 = bs_d1(S,K,r,sigma,tau)
    return norm.pdf(d1) / (S * sigma * np.sqrt(tau))

def bs_vega(S,K,r,sigma,tau):
    if sigma <= 0 or tau <= 0:
        return 0.0
    d1 = bs_d1(S,K,r,sigma,tau)
    return S * norm.pdf(d1) * np.sqrt(tau)

def bs_theta(S,K,r,sigma,tau,option_type="call"):
    if sigma <=0 or tau<=0:
        return 0.0
    d1 = bs_d1(S,K,r,sigma,tau)
    d2 = d1 - sigma*np.sqrt(tau)
    term1 = - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(tau))
    if option_type=="call":
        term2 = r*K*np.exp(-r*tau)*norm.cdf(d2)
        return term1 - term2
    else:
        term2 = r*K*np.exp(-r*tau)*norm.cdf(-d2)
        return term1 + term2

# -----------------------
#  SELLER'S PERSPECTIVE FUNCTIONS
# -----------------------
def seller_strength_score(row, weights=SCORE_WEIGHTS):
    chg_oi = abs(safe_float(row.get("Chg_OI_CE",0))) + abs(safe_float(row.get("Chg_OI_PE",0)))
    vol = safe_float(row.get("Vol_CE",0)) + safe_float(row.get("Vol_PE",0))
    oi = safe_float(row.get("OI_CE",0)) + safe_float(row.get("OI_PE",0))
    iv_ce = safe_float(row.get("IV_CE", np.nan))
    iv_pe = safe_float(row.get("IV_PE", np.nan))
    iv = np.nanmean([v for v in (iv_ce, iv_pe) if not np.isnan(v)]) if (not np.isnan(iv_ce) or not np.isnan(iv_pe)) else 0

    score = weights["chg_oi"]*chg_oi + weights["volume"]*vol + weights["oi"]*oi + weights["iv"]*iv
    return score

def seller_price_oi_divergence(chg_oi, vol, ltp_change, option_type="CE"):
    vol_up = vol > 0
    oi_up = chg_oi > 0
    price_up = (ltp_change is not None and ltp_change > 0)

    if option_type == "CE":
        if oi_up and vol_up and price_up:
            return "Sellers WRITING calls as price rises (Bearish conviction)"
        if oi_up and vol_up and not price_up:
            return "Sellers WRITING calls on weakness (Strong bearish)"
        if not oi_up and vol_up and price_up:
            return "Sellers BUYING back calls as price rises (Covering bearish)"
        if not oi_up and vol_up and not price_up:
            return "Sellers BUYING back calls on weakness (Reducing bearish exposure)"
    else:
        if oi_up and vol_up and price_up:
            return "Sellers WRITING puts on strength (Bullish conviction)"
        if oi_up and vol_up and not price_up:
            return "Sellers WRITING puts as price falls (Strong bullish)"
        if not oi_up and vol_up and price_up:
            return "Sellers BUYING back puts on strength (Covering bullish)"
        if not oi_up and vol_up and not price_up:
            return "Sellers BUYING back puts as price falls (Reducing bullish exposure)"

    if oi_up and not vol_up:
        return "Sellers quietly WRITING options"
    if (not oi_up) and not vol_up:
        return "Sellers quietly UNWINDING"

    return "Sellers inactive"

def calculate_seller_max_pain(df):
    pain_dict = {}
    for _, row in df.iterrows():
        strike = row["strikePrice"]
        ce_oi = safe_int(row.get("OI_CE", 0))
        pe_oi = safe_int(row.get("OI_PE", 0))
        ce_ltp = safe_float(row.get("LTP_CE", 0))
        pe_ltp = safe_float(row.get("LTP_PE", 0))

        ce_pain = ce_oi * max(0, ce_ltp) if strike < df["strikePrice"].mean() else 0
        pe_pain = pe_oi * max(0, pe_ltp) if strike > df["strikePrice"].mean() else 0

        pain = ce_pain + pe_pain
        pain_dict[strike] = pain

    if pain_dict:
        return min(pain_dict, key=pain_dict.get)
    return None

def calculate_seller_market_bias(merged_df, spot, atm_strike):
    polarity = 0.0

    for _, r in merged_df.iterrows():
        strike = r["strikePrice"]
        chg_ce = safe_int(r.get("Chg_OI_CE", 0))
        chg_pe = safe_int(r.get("Chg_OI_PE", 0))

        if strike < atm_strike:
            if chg_ce > 0:
                polarity -= 2.0
            elif chg_ce < 0:
                polarity += 1.5
        elif strike > atm_strike:
            if chg_ce > 0:
                polarity -= 0.7
            elif chg_ce < 0:
                polarity += 0.5

        if strike > atm_strike:
            if chg_pe > 0:
                polarity += 2.0
            elif chg_pe < 0:
                polarity -= 1.5
        elif strike < atm_strike:
            if chg_pe > 0:
                polarity += 0.7
            elif chg_pe < 0:
                polarity -= 0.5

    total_ce_oi = merged_df["OI_CE"].sum()
    total_pe_oi = merged_df["OI_PE"].sum()
    if total_ce_oi > 0:
        pcr = total_pe_oi / total_ce_oi
        if pcr > 2.0:
            polarity += 1.0
        elif pcr < 0.5:
            polarity -= 1.0

    avg_iv_ce = merged_df["IV_CE"].mean()
    avg_iv_pe = merged_df["IV_PE"].mean()
    if avg_iv_ce > avg_iv_pe + 5:
        polarity -= 0.3
    elif avg_iv_pe > avg_iv_ce + 5:
        polarity += 0.3

    total_gex_ce = merged_df["GEX_CE"].sum()
    total_gex_pe = merged_df["GEX_PE"].sum()
    net_gex = total_gex_ce + total_gex_pe
    if net_gex < -1000000:
        polarity -= 0.4
    elif net_gex > 1000000:
        polarity += 0.4

    max_pain = calculate_seller_max_pain(merged_df)
    if max_pain:
        distance_to_spot = abs(spot - max_pain) / spot * 100
        if distance_to_spot < 1.0:
            polarity += 0.5

    if polarity > 3.0:
        return {
            "bias": "STRONG BULLISH SELLERS 🚀",
            "polarity": polarity,
            "color": "#00ff88",
            "explanation": "Sellers aggressively WRITING PUTS (bullish conviction). Expecting price to STAY ABOVE strikes.",
            "action": "Bullish breakout likely. Sellers confident in upside."
        }
    elif polarity > 1.0:
        return {
            "bias": "BULLISH SELLERS 📈",
            "polarity": polarity,
            "color": "#00cc66",
            "explanation": "Sellers leaning towards PUT writing. Moderate bullish sentiment.",
            "action": "Expect support to hold. Upside bias."
        }
    elif polarity < -3.0:
        return {
            "bias": "STRONG BEARISH SELLERS 🐻",
            "polarity": polarity,
            "color": "#ff4444",
            "explanation": "Sellers aggressively WRITING CALLS (bearish conviction). Expecting price to STAY BELOW strikes.",
            "action": "Bearish breakdown likely. Sellers confident in downside."
        }
    elif polarity < -1.0:
        return {
            "bias": "BEARISH SELLERS 📉",
            "polarity": polarity,
            "color": "#ff6666",
            "explanation": "Sellers leaning towards CALL writing. Moderate bearish sentiment.",
            "action": "Expect resistance to hold. Downside bias."
        }
    else:
        return {
            "bias": "NEUTRAL SELLERS ⚖️",
            "polarity": polarity,
            "color": "#66b3ff",
            "explanation": "Balanced seller activity. No clear directional bias.",
            "action": "Range-bound expected. Wait for clearer signals."
        }

def compute_pcr_df(merged_df):
    df = merged_df.copy()
    df["OI_CE"] = pd.to_numeric(df.get("OI_CE", 0), errors="coerce").fillna(0).astype(int)
    df["OI_PE"] = pd.to_numeric(df.get("OI_PE", 0), errors="coerce").fillna(0).astype(int)

    def pcr_calc(row):
        ce = int(row["OI_CE"]) if row["OI_CE"] is not None else 0
        pe = int(row["OI_PE"]) if row["OI_PE"] is not None else 0
        if ce <= 0:
            if pe > 0:
                return float("inf")
            else:
                return np.nan
        return pe / ce

    df["PCR"] = df.apply(pcr_calc, axis=1)
    return df

# -----------------------
#  DATA PROCESSING FUNCTIONS
# -----------------------
def process_option_chain(records, current_expiry, spot_price):
    """Process raw NSE option chain data into structured dataframe"""
    processed_data = []

    for record in records:
        try:
            strike = record.get('strikePrice', 0)

            # Initialize row data
            row_data = {'strikePrice': strike}

            # Process CE (Call) data
            if 'CE' in record:
                ce = record['CE']
                if ce.get('expiryDate') == current_expiry:
                    row_data.update({
                        'OI_CE': safe_int(ce.get('openInterest', 0)),
                        'Chg_OI_CE': safe_int(ce.get('changeinOpenInterest', 0)),
                        'Vol_CE': safe_int(ce.get('totalTradedVolume', 0)),
                        'IV_CE': safe_float(ce.get('impliedVolatility', 0)),
                        'LTP_CE': safe_float(ce.get('lastPrice', 0)),
                        'Chg_CE': safe_float(ce.get('change', 0)),
                        'Bid_CE': safe_float(ce.get('bidprice', 0)),
                        'Ask_CE': safe_float(ce.get('askPrice', 0)),
                    })

            # Process PE (Put) data
            if 'PE' in record:
                pe = record['PE']
                if pe.get('expiryDate') == current_expiry:
                    row_data.update({
                        'OI_PE': safe_int(pe.get('openInterest', 0)),
                        'Chg_OI_PE': safe_int(pe.get('changeinOpenInterest', 0)),
                        'Vol_PE': safe_int(pe.get('totalTradedVolume', 0)),
                        'IV_PE': safe_float(pe.get('impliedVolatility', 0)),
                        'LTP_PE': safe_float(pe.get('lastPrice', 0)),
                        'Chg_PE': safe_float(pe.get('change', 0)),
                        'Bid_PE': safe_float(pe.get('bidprice', 0)),
                        'Ask_PE': safe_float(pe.get('askPrice', 0)),
                    })

            # Only add if we have at least CE or PE data
            if len(row_data) > 1:
                processed_data.append(row_data)

        except Exception as e:
            continue

    df = pd.DataFrame(processed_data)

    if df.empty:
        return df

    # Fill missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # Calculate days to expiry
    try:
        expiry_date = datetime.strptime(current_expiry, "%d-%b-%Y")
        now = get_ist_now()
        days_to_expiry = (expiry_date - now).days
    except:
        days_to_expiry = 0

    # Calculate Greeks
    tau = max(days_to_expiry / 365.0, 0.001)

    for idx, row in df.iterrows():
        strike = row['strikePrice']

        # CE Greeks
        if row.get('IV_CE', 0) > 0:
            sigma_ce = row['IV_CE'] / 100.0
            df.loc[idx, 'Delta_CE'] = bs_delta(spot_price, strike, RISK_FREE_RATE, sigma_ce, tau, "call")
            df.loc[idx, 'Gamma_CE'] = bs_gamma(spot_price, strike, RISK_FREE_RATE, sigma_ce, tau)
            df.loc[idx, 'Vega_CE'] = bs_vega(spot_price, strike, RISK_FREE_RATE, sigma_ce, tau)
            df.loc[idx, 'Theta_CE'] = bs_theta(spot_price, strike, RISK_FREE_RATE, sigma_ce, tau, "call")

            # GEX (Gamma Exposure) = Gamma * OI * Lot Size * Spot^2 / 100
            gamma_ce = df.loc[idx, 'Gamma_CE']
            oi_ce = row['OI_CE']
            df.loc[idx, 'GEX_CE'] = gamma_ce * oi_ce * LOT_SIZE * (spot_price ** 2) / 100

        # PE Greeks
        if row.get('IV_PE', 0) > 0:
            sigma_pe = row['IV_PE'] / 100.0
            df.loc[idx, 'Delta_PE'] = bs_delta(spot_price, strike, RISK_FREE_RATE, sigma_pe, tau, "put")
            df.loc[idx, 'Gamma_PE'] = bs_gamma(spot_price, strike, RISK_FREE_RATE, sigma_pe, tau)
            df.loc[idx, 'Vega_PE'] = bs_vega(spot_price, strike, RISK_FREE_RATE, sigma_pe, tau)
            df.loc[idx, 'Theta_PE'] = bs_theta(spot_price, strike, RISK_FREE_RATE, sigma_pe, tau, "put")

            gamma_pe = df.loc[idx, 'Gamma_PE']
            oi_pe = row['OI_PE']
            df.loc[idx, 'GEX_PE'] = gamma_pe * oi_pe * LOT_SIZE * (spot_price ** 2) / 100

    # Fill NaN Greeks with 0
    greek_cols = [c for c in df.columns if any(g in c for g in ['Delta', 'Gamma', 'Vega', 'Theta', 'GEX'])]
    df[greek_cols] = df[greek_cols].fillna(0)

    return df.sort_values('strikePrice')

def find_atm_strike(spot_price, df):
    """Find the at-the-money strike"""
    if df.empty:
        return round(spot_price / 50) * 50

    strikes = df['strikePrice'].values
    atm = min(strikes, key=lambda x: abs(x - spot_price))
    return atm

# -----------------------
#  DISPLAY FUNCTIONS
# -----------------------
def display_key_metrics(df, spot_price, atm_strike, current_expiry, expiry_dates):
    """Display key metrics at the top"""

    # Calculate metrics
    total_ce_oi = df['OI_CE'].sum()
    total_pe_oi = df['OI_PE'].sum()
    pcr_oi = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0

    total_ce_vol = df['Vol_CE'].sum()
    total_pe_vol = df['Vol_PE'].sum()
    pcr_vol = total_pe_vol / total_ce_vol if total_ce_vol > 0 else 0

    total_gex_ce = df['GEX_CE'].sum()
    total_gex_pe = df['GEX_PE'].sum()
    net_gex = total_gex_ce + total_gex_pe

    # Calculate seller bias
    bias_result = calculate_seller_market_bias(df, spot_price, atm_strike)

    # Calculate days to expiry
    try:
        expiry_date = datetime.strptime(current_expiry, "%d-%b-%Y")
        now = get_ist_now()
        days_to_expiry = (expiry_date - now).days
    except:
        days_to_expiry = 0

    # Display metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            label="NIFTY Spot",
            value=f"₹{spot_price:,.2f}",
            delta=None
        )

    with col2:
        st.metric(
            label="Seller Bias",
            value=bias_result['bias'],
            delta=f"Score: {bias_result['polarity']:.1f}"
        )
        st.markdown(f"<p style='color:{bias_result['color']}; font-size:0.8rem;'>{bias_result['action']}</p>",
                   unsafe_allow_html=True)

    with col3:
        st.metric(
            label="PCR (OI)",
            value=f"{pcr_oi:.2f}",
            delta="Bullish" if pcr_oi > 1.2 else ("Bearish" if pcr_oi < 0.8 else "Neutral"),
            delta_color="normal" if pcr_oi > 1.2 else ("inverse" if pcr_oi < 0.8 else "off")
        )

    with col4:
        st.metric(
            label="Net GEX",
            value=f"₹{net_gex/10000000:.2f}Cr",
            delta="Positive" if net_gex > 0 else "Negative",
            delta_color="normal" if net_gex > 0 else "inverse"
        )

    with col5:
        expiry_emoji = "🔥" if days_to_expiry <= 5 else "📅"
        st.metric(
            label="Expiry",
            value=f"{days_to_expiry} days {expiry_emoji}",
            delta=current_expiry
        )
        if days_to_expiry <= 5:
            st.markdown("<p style='color:#ff4444; font-size:0.8rem;'>⚠️ EXPIRY SPIKE ZONE!</p>",
                       unsafe_allow_html=True)

def display_option_chain_table(df, atm_strike):
    """Display the full option chain table"""
    st.subheader("📊 Complete Option Chain")

    # Create display dataframe
    display_df = df.copy()

    # Highlight ATM row
    def highlight_atm(row):
        if row['strikePrice'] == atm_strike:
            return ['background-color: #2a2a4a'] * len(row)
        return [''] * len(row)

    # Select columns to display
    display_cols = [
        'strikePrice',
        'OI_CE', 'Chg_OI_CE', 'Vol_CE', 'IV_CE', 'LTP_CE',
        'OI_PE', 'Chg_OI_PE', 'Vol_PE', 'IV_PE', 'LTP_PE'
    ]

    # Filter available columns
    display_cols = [col for col in display_cols if col in display_df.columns]

    styled_df = display_df[display_cols].style.apply(highlight_atm, axis=1)
    st.dataframe(styled_df, use_container_width=True, height=400)

def display_sellers_bias_analysis(df, spot_price, atm_strike):
    """Display seller's perspective bias analysis"""
    st.subheader("🎯 Seller's Perspective Analysis")

    bias_result = calculate_seller_market_bias(df, spot_price, atm_strike)

    # Main bias card
    st.markdown(f"""
    <div style='background-color: {bias_result['color']}22; border-left: 5px solid {bias_result['color']}; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
        <h2 style='color: {bias_result['color']}; margin: 0;'>{bias_result['bias']}</h2>
        <p style='font-size: 1.1rem; margin-top: 10px;'><b>Polarity Score:</b> {bias_result['polarity']:.2f}</p>
        <p style='margin-top: 10px;'><b>Explanation:</b> {bias_result['explanation']}</p>
        <p style='margin-top: 10px; font-weight: 600;'><b>Expected Action:</b> {bias_result['action']}</p>
    </div>
    """, unsafe_allow_html=True)

    # Detailed breakdown
    st.markdown("#### 🔍 Detailed Seller Activity Breakdown")

    # Filter strikes around ATM
    atm_window = df[abs(df['strikePrice'] - atm_strike) <= 300].copy()

    for _, row in atm_window.iterrows():
        strike = row['strikePrice']

        # CE Analysis
        ce_chg_oi = safe_int(row.get('Chg_OI_CE', 0))
        ce_vol = safe_int(row.get('Vol_CE', 0))
        ce_ltp_chg = safe_float(row.get('Chg_CE', 0))
        ce_signal = seller_price_oi_divergence(ce_chg_oi, ce_vol, ce_ltp_chg, "CE")

        # PE Analysis
        pe_chg_oi = safe_int(row.get('Chg_OI_PE', 0))
        pe_vol = safe_int(row.get('Vol_PE', 0))
        pe_ltp_chg = safe_float(row.get('Chg_PE', 0))
        pe_signal = seller_price_oi_divergence(pe_chg_oi, pe_vol, pe_ltp_chg, "PE")

        if "WRITING" in ce_signal or "WRITING" in pe_signal:
            col1, col2 = st.columns(2)

            with col1:
                if "WRITING" in ce_signal:
                    st.markdown(f"""
                    **{strike} CE:** 🔴 {ce_signal}
                    - ΔOI: {ce_chg_oi:,} | Vol: {ce_vol:,}
                    """)

            with col2:
                if "WRITING" in pe_signal:
                    st.markdown(f"""
                    **{strike} PE:** 🟢 {pe_signal}
                    - ΔOI: {pe_chg_oi:,} | Vol: {pe_vol:,}
                    """)

def display_moment_detector(df, atm_strike):
    """Display moment detector analysis"""
    st.subheader("⚡ Moment Detector — High Probability Moves")

    st.info("💡 Moment Detector identifies strikes with unusual activity that may signal imminent moves")

    # Calculate moment scores for each strike
    moment_data = []

    for _, row in df.iterrows():
        strike = row['strikePrice']

        # Calculate for CE
        ce_vol = safe_float(row.get('Vol_CE', 0))
        ce_iv = safe_float(row.get('IV_CE', 0))
        ce_chg_oi = abs(safe_float(row.get('Chg_OI_CE', 0)))

        if ce_vol > 0 and ce_iv > 0:
            ce_momentum_burst = ce_vol * ce_iv * ce_chg_oi / 100000
            ce_gamma = safe_float(row.get('Gamma_CE', 0))

            moment_data.append({
                'Strike': strike,
                'Type': 'CE',
                'Momentum_Burst': ce_momentum_burst,
                'Gamma': ce_gamma,
                'Volume': ce_vol,
                'IV': ce_iv,
                'ΔOI': ce_chg_oi
            })

        # Calculate for PE
        pe_vol = safe_float(row.get('Vol_PE', 0))
        pe_iv = safe_float(row.get('IV_PE', 0))
        pe_chg_oi = abs(safe_float(row.get('Chg_OI_PE', 0)))

        if pe_vol > 0 and pe_iv > 0:
            pe_momentum_burst = pe_vol * pe_iv * pe_chg_oi / 100000
            pe_gamma = safe_float(row.get('Gamma_PE', 0))

            moment_data.append({
                'Strike': strike,
                'Type': 'PE',
                'Momentum_Burst': pe_momentum_burst,
                'Gamma': pe_gamma,
                'Volume': pe_vol,
                'IV': pe_iv,
                'ΔOI': pe_chg_oi
            })

    if moment_data:
        moment_df = pd.DataFrame(moment_data)
        moment_df = moment_df.sort_values('Momentum_Burst', ascending=False).head(10)

        st.markdown("#### 🔥 Top 10 Momentum Bursts")
        st.dataframe(moment_df, use_container_width=True)

        # Show top 3 in detail
        st.markdown("#### ⭐ Top 3 Moment Opportunities")
        for idx, row in moment_df.head(3).iterrows():
            strike = row['Strike']
            opt_type = row['Type']
            momentum = row['Momentum_Burst']

            direction = "📉 Bearish (Sellers writing calls)" if opt_type == "CE" else "📈 Bullish (Sellers writing puts)"

            st.markdown(f"""
            **Strike {strike} {opt_type}** — Momentum Score: {momentum:.2f}
            - Direction: {direction}
            - Volume: {row['Volume']:,.0f} | IV: {row['IV']:.1f}% | ΔOI: {row['ΔOI']:,.0f}
            """)
    else:
        st.warning("No significant momentum detected at this time")

def display_hot_strikes(df):
    """Display hottest strikes by activity"""
    st.subheader("🔥 Hottest Strikes — Maximum Seller Action")

    # Calculate activity score for each strike
    df_copy = df.copy()
    df_copy['Activity_Score'] = (
        abs(df_copy.get('Chg_OI_CE', 0)) + abs(df_copy.get('Chg_OI_PE', 0)) +
        df_copy.get('Vol_CE', 0) * 0.5 + df_copy.get('Vol_PE', 0) * 0.5
    )

    hot_strikes = df_copy.nlargest(10, 'Activity_Score')[
        ['strikePrice', 'OI_CE', 'Chg_OI_CE', 'Vol_CE', 'OI_PE', 'Chg_OI_PE', 'Vol_PE', 'Activity_Score']
    ]

    st.dataframe(hot_strikes, use_container_width=True)

    # Visualize top 5
    st.markdown("#### 📊 Top 5 Strike Visualization")
    top5 = hot_strikes.head(5)

    for _, row in top5.iterrows():
        strike = row['strikePrice']
        ce_activity = abs(row['Chg_OI_CE']) + row['Vol_CE'] * 0.5
        pe_activity = abs(row['Chg_OI_PE']) + row['Vol_PE'] * 0.5

        col1, col2 = st.columns(2)
        with col1:
            st.metric(f"{strike} CE Activity", f"{ce_activity:,.0f}")
        with col2:
            st.metric(f"{strike} PE Activity", f"{pe_activity:,.0f}")

def display_greeks_analysis(df, spot_price):
    """Display Greeks and GEX analysis"""
    st.subheader("📈 Greeks & Gamma Exposure Analysis")

    # Calculate total Greeks
    total_delta_ce = df['Delta_CE'].sum() if 'Delta_CE' in df.columns else 0
    total_delta_pe = df['Delta_PE'].sum() if 'Delta_PE' in df.columns else 0
    total_gamma_ce = df['Gamma_CE'].sum() if 'Gamma_CE' in df.columns else 0
    total_gamma_pe = df['Gamma_PE'].sum() if 'Gamma_PE' in df.columns else 0
    total_gex_ce = df['GEX_CE'].sum() if 'GEX_CE' in df.columns else 0
    total_gex_pe = df['GEX_PE'].sum() if 'GEX_PE' in df.columns else 0

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Delta (CE)", f"{total_delta_ce:.2f}")
        st.metric("Total Delta (PE)", f"{total_delta_pe:.2f}")
        net_delta = total_delta_ce + total_delta_pe
        st.metric("Net Delta", f"{net_delta:.2f}",
                 delta="Bullish" if net_delta > 0 else "Bearish")

    with col2:
        st.metric("Total Gamma (CE)", f"{total_gamma_ce:.4f}")
        st.metric("Total Gamma (PE)", f"{total_gamma_pe:.4f}")
        net_gamma = total_gamma_ce + total_gamma_pe
        st.metric("Net Gamma", f"{net_gamma:.4f}")

    with col3:
        st.metric("Total GEX (CE)", f"₹{total_gex_ce/10000000:.2f}Cr")
        st.metric("Total GEX (PE)", f"₹{total_gex_pe/10000000:.2f}Cr")
        net_gex = total_gex_ce + total_gex_pe
        st.metric("Net GEX", f"₹{net_gex/10000000:.2f}Cr",
                 delta="Positive" if net_gex > 0 else "Negative")

    # GEX Interpretation
    st.markdown("#### 💡 GEX Interpretation")
    if net_gex > 1000000:
        st.success("✅ Positive GEX suggests market makers will resist large moves (range-bound environment)")
    elif net_gex < -1000000:
        st.warning("⚠️ Negative GEX suggests potential for volatile moves (trending environment)")
    else:
        st.info("ℹ️ Neutral GEX suggests balanced market conditions")

def display_max_pain_and_pcr(df, spot_price):
    """Display Max Pain and PCR analysis"""
    st.subheader("🎲 Max Pain & Put-Call Ratio")

    # Calculate Max Pain
    max_pain = calculate_seller_max_pain(df)

    # Calculate PCR
    pcr_df = compute_pcr_df(df)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🎯 Max Pain Analysis")
        if max_pain:
            distance = abs(spot_price - max_pain)
            pct_distance = (distance / spot_price) * 100

            st.metric("Max Pain Strike", f"₹{max_pain:,.0f}")
            st.metric("Current Spot", f"₹{spot_price:,.2f}")
            st.metric("Distance from Max Pain", f"₹{distance:,.2f} ({pct_distance:.2f}%)")

            if pct_distance < 1.0:
                st.success("✅ Spot is very close to Max Pain — sellers in control!")
            elif pct_distance < 2.0:
                st.info("📍 Spot is near Max Pain — expect range-bound action")
            else:
                st.warning("⚠️ Spot is away from Max Pain — potential for a move toward Max Pain")

    with col2:
        st.markdown("#### 📊 Put-Call Ratio Analysis")

        total_ce_oi = df['OI_CE'].sum()
        total_pe_oi = df['OI_PE'].sum()
        pcr_oi = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0

        total_ce_vol = df['Vol_CE'].sum()
        total_pe_vol = df['Vol_PE'].sum()
        pcr_vol = total_pe_vol / total_ce_vol if total_ce_vol > 0 else 0

        st.metric("PCR (Open Interest)", f"{pcr_oi:.2f}")
        st.metric("PCR (Volume)", f"{pcr_vol:.2f}")

        if pcr_oi > 1.2:
            st.success("📈 PCR > 1.2 — Bullish sentiment (more puts = sellers bullish)")
        elif pcr_oi < 0.8:
            st.error("📉 PCR < 0.8 — Bearish sentiment (more calls = sellers bearish)")
        else:
            st.info("⚖️ PCR Neutral — Balanced sentiment")

# Main render function
def render_nifty_option_screener_v6():
    """
    Main function to render the Nifty Option Screener v6.0
    Can be called from app.py tabs
    """
    st.markdown("### 🎯 NIFTY Option Screener v6.0 — SELLER'S PERSPECTIVE + MOMENT DETECTOR")

    current_ist = get_ist_datetime_str()
    st.markdown(f"""
    <div style='text-align: center; margin-bottom: 20px;'>
        <span style='background-color: #1a1f2e; color: #ff66cc; padding: 8px 15px; border-radius: 20px; border: 2px solid #ff66cc; font-weight: 700; font-size: 1.1rem;'>
            🕐 IST: {current_ist}
        </span>
    </div>
    """, unsafe_allow_html=True)

    try:
        # Import option chain analyzer
        from option_chain_analysis import OptionChainAnalyzer

        # Initialize analyzer
        analyzer = OptionChainAnalyzer()

        # Fetch option chain data for NIFTY
        with st.spinner("🔄 Fetching live NIFTY option chain data from NSE..."):
            oc_data = analyzer.fetch_option_chain('NIFTY')

        if not oc_data['success']:
            st.error(f"❌ Failed to fetch option chain data: {oc_data.get('error', 'Unknown error')}")
            st.warning("⚠️ This might be due to market hours, NSE website issues, or network connectivity.")
            st.info("💡 The screener works during market hours (9:15 AM - 3:30 PM IST) and uses live NSE data.")
            return

        # Extract data
        spot_price = oc_data['spot_price']
        records = oc_data['records']
        expiry_dates = oc_data['expiry_dates']
        current_expiry = oc_data['current_expiry']

        # Process option chain data
        chain_df = process_option_chain(records, current_expiry, spot_price)

        if chain_df.empty:
            st.error("❌ No option chain data available")
            return

        # Calculate ATM strike
        atm_strike = find_atm_strike(spot_price, chain_df)

        # Display key metrics
        display_key_metrics(chain_df, spot_price, atm_strike, current_expiry, expiry_dates)

        st.markdown("---")

        # Create tabs for different analyses
        analysis_tabs = st.tabs([
            "📊 Option Chain",
            "🎯 Seller's Bias",
            "⚡ Moment Detector",
            "🔥 Hot Strikes",
            "📈 Greeks Analysis",
            "🎲 Max Pain & PCR"
        ])

        with analysis_tabs[0]:
            display_option_chain_table(chain_df, atm_strike)

        with analysis_tabs[1]:
            display_sellers_bias_analysis(chain_df, spot_price, atm_strike)

        with analysis_tabs[2]:
            display_moment_detector(chain_df, atm_strike)

        with analysis_tabs[3]:
            display_hot_strikes(chain_df)

        with analysis_tabs[4]:
            display_greeks_analysis(chain_df, spot_price)

        with analysis_tabs[5]:
            display_max_pain_and_pcr(chain_df, spot_price)

    except ImportError as e:
        st.error(f"❌ Failed to import required modules: {e}")
        st.info("Please ensure option_chain_analysis.py is in the project directory")
    except Exception as e:
        st.error(f"❌ Error loading Nifty Option Screener: {e}")
        st.exception(e)

if __name__ == "__main__":
    # For standalone testing
    st.set_page_config(page_title="Nifty Option Screener v6.0", layout="wide")
    render_nifty_option_screener_v6()
