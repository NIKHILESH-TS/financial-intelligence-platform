import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

import core_pipelines as cp
import risk_sentiment as rs
import ml_inference as ml

st.set_page_config(
    page_title="AlphaIntel | AI Powered Real-TimeFinancial Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# MODERN DARK GLASS INTERFACE THEME
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@300;400;600;700&display=swap');
        
        .stApp { background-color: #07111F !important; color: #E2E8F0 !important; }
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        
        .terminal-block { 
            background: #0C192E; 
            border: 1px solid #1E293B; 
            padding: 1.25rem; 
            border-radius: 10px; 
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
            margin-bottom: 1.25rem;
        }
        
        .metric-title { font-size: 0.75rem; color: #94A3B8; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; }
        .metric-value { font-family: 'JetBrains Mono', monospace; font-size: 1.65rem; font-weight: 700; color: #F8FAFC; margin-top: 0.25rem; }
        
        .signal-bullish { color: #00C896 !important; font-weight: 700; }
        .signal-bearish { color: #FF6B6B !important; font-weight: 700; }
        .signal-neutral { color: #FBBF24 !important; font-weight: 700; }
        
        .info-tag { font-size: 0.8rem; background: #1E293B; padding: 0.25rem 0.5rem; border-radius: 6px; color: #94A3B8; font-weight: 500; }
        hr { border-color: #1E293B !important; margin: 1rem 0 !important; }
        h1, h2, h3, h4, h5, h6 { color: #F8FAFC !important; font-weight: 600 !important; }
        
        /* Subtle styling override for cleaner sidebar button lists */
        .element-container button { width: 100% !important; text-align: left !important; }
    </style>
""", unsafe_allow_html=True)

# Main Header
st.title("🧠 AlphaIntel™ AI Powered Real-Time Financial Intelligence Platform")
st.caption("Unified Decision-Support Platform • Institutional Core Analytics Framework & Quantitative Inference System")
st.markdown("---")

# -----------------------------------------------------------------------------
# SIDEBAR SEARCH ENGINE & FAST METADATA ROUTING LOOKUPS
# -----------------------------------------------------------------------------
st.sidebar.header("🔎 Asset Search Matrix")

# Initialize persistent session state tracking variables for fast button navigation clicks
if "current_ticker" not in st.session_state:
    st.session_state.current_ticker = "AAPL"

# Core fuzzy matching index table maps natural language queries to yfinance symbols
COMPANY_MAP = {
    "apple": "AAPL", "microsoft": "MSFT", "alphabet": "GOOGL", "google": "GOOGL",
    "amazon": "AMZN", "tesla": "TSLA", "nvidia": "NVDA", "goldman sachs": "GS",
    "goldman": "GS", "deutsche bank": "DB", "deutsche": "DB", "reliance": "RELIANCE.NS",
    "tcs": "TCS.NS", "infosys": "INFY.NS", "bitcoin": "BTC-USD", "ethereum": "ETH-USD"
}

user_search_input = st.sidebar.text_input(
    "Search by Company Name or Ticker",
    value=st.session_state.current_ticker,
    placeholder="e.g., Apple, TSLA, RELIANCE.NS, BTC-USD...",
    help="Enter standard symbols or popular company names directly."
)

# Fast routing parsing block
cleaned_query = user_search_input.lower().strip()
if cleaned_query in COMPANY_MAP:
    final_ticker = COMPANY_MAP[cleaned_query]
else:
    final_ticker = user_search_input.upper().strip()

# POPULAR QUICK-CLICK ASSET SELECTORS
st.sidebar.markdown("### ⭐ Popular Core Assets")

def update_session_ticker(symbol_token: str):
    st.session_state.current_ticker = symbol_token

# Build modular quick action navigation matrices
col_sidebar_a, col_sidebar_b = st.sidebar.columns(2)
with col_sidebar_a:
    if st.button("🍎 Apple (AAPL)", on_click=update_session_ticker, args=("AAPL",)): final_ticker = "AAPL"
    if st.button("🚗 Tesla (TSLA)", on_click=update_session_ticker, args=("TSLA",)): final_ticker = "TSLA"
    if st.button("📈 NVIDIA (NVDA)", on_click=update_session_ticker, args=("NVDA",)): final_ticker = "NVDA"
    if st.button("🛒 Amazon (AMZN)", on_click=update_session_ticker, args=("AMZN",)): final_ticker = "AMZN"
    if st.button("🖥️ Microsoft (MSFT)", on_click=update_session_ticker, args=("MSFT",)): final_ticker = "MSFT"
with col_sidebar_b:
    if st.button("🏦 Goldman (GS)", on_click=update_session_ticker, args=("GS",)): final_ticker = "GS"
    if st.button("🦁 Deutsche (DB)", on_click=update_session_ticker, args=("DB",)): final_ticker = "DB"
    if st.button("🇮🇳 Reliance", on_click=update_session_ticker, args=("RELIANCE.NS",)): final_ticker = "RELIANCE.NS"
    if st.button("🇮🇳 TCS", on_click=update_session_ticker, args=("TCS.NS",)): final_ticker = "TCS.NS"
    if st.button("🪙 Bitcoin", on_click=update_session_ticker, args=("BTC-USD",)): final_ticker = "BTC-USD"

st.sidebar.markdown("---")
st.sidebar.markdown("### Parameters")
lookback_period = st.sidebar.selectbox("Lookback Query Window", options=["1mo", "6mo", "1y", "2y", "5y"], index=2)
lag_selection = st.sidebar.slider("Autoregressive Feature Lags", min_value=3, max_value=10, value=5)

# -----------------------------------------------------------------------------
# CORE ANALYTICAL COMPUTATION HANDOFF ENGINE
# -----------------------------------------------------------------------------
with st.spinner(f"Ingesting market data streams for {final_ticker}..."):
    data_success, historical_df = cp.fetch_historical_data(final_ticker, period=lookback_period)
    news_feed = cp.fetch_ticker_news(final_ticker)

if not data_success or historical_df.empty:
    st.error(f"🚨 Target Parsing Defect: Historical ledger compilation failed for asset string signature: '{final_ticker}'. Please verify ticker alignment syntax rules.")
else:
    risk_metrics = rs.calculate_advanced_risk(historical_df)
    sentiment_payload = rs.analyze_sentiment_pool(news_feed)
    ml_success, prediction_results = ml.generate_price_prediction(historical_df, lag_days=lag_selection)
    
    # Extract structural naming profile records via the core pipeline index
    meta_profile = cp.get_asset_profile(final_ticker)

    # 🚀 PREMIUM VISUAL UPGRADE: DYNAMIC ASSET METADATA BLOCK BANNER
    st.markdown(f"""
        <div class="terminal-block" style="background: linear-gradient(90deg, #0F2042 0%, #0C192E 100%); border-left: 4px solid #3B82F6;">
            <span style="font-size:0.75rem; color:#64748B; font-weight:700; text-transform:uppercase; letter-spacing:1px;">Active Terminal Target Profile</span>
            <h2 style="margin: 0.1rem 0 0.2rem 0; font-size:1.75rem;">{meta_profile['name']}</h2>
            <span style="font-size:0.9rem; color:#94A3B8; font-family:'JetBrains Mono';">
                Exchange Index Identification: <b style="color:#F8FAFC;">{meta_profile['exchange']}</b> &nbsp;&bull;&nbsp; Global Allocation Sector: <b style="color:#F8FAFC;">{meta_profile['sector']}</b>
            </span>
        </div>
    """, unsafe_allow_html=True)

    # RECLASSIFIED RISK PARAMETERS LOGIC
    vol_val = risk_metrics["annualized_volatility"]
    risk_label = "LOW RISK" if vol_val < 20 else "MODERATE RISK" if vol_val < 35 else "HIGH RISK"
    risk_class = "signal-bullish" if vol_val < 20 else "signal-neutral" if vol_val < 35 else "signal-bearish"
    
    sent_lbl = sentiment_payload["sentiment_label"]
    sent_class = "signal-bullish" if sent_lbl == "Bullish" else "signal-bearish" if sent_lbl == "Bearish" else "signal-neutral"
    
    dir_lbl = prediction_results["direction"]
    dir_class = "signal-bullish" if dir_lbl == "Upward" else "signal-bearish" if dir_lbl == "Downward" else "signal-neutral"

    # -------------------------------------------------------------------------
    # DOMAIN I: EXECUTIVE SUMMARY INTELLIGENCE MATRIX
    # -------------------------------------------------------------------------
    st.subheader("🏁 Executive Intelligence Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
            <div class="terminal-block" style="border-left: 4px solid #3B82F6;">
                <div class="metric-title">Synthesized Portfolio Signal</div>
                <div class="metric-value" style="font-size:1.25rem; color:#3B82F6;">
                    {dir_lbl if dir_lbl != "Undetermined" else "Consoc"} Outlook
                </div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="terminal-block"><div class="metric-title">Risk Threshold Assessment</div><div class="metric-value {risk_class}">{risk_label}</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="terminal-block"><div class="metric-title">Forecast Vector Direction</div><div class="metric-value {dir_class}">{dir_lbl}</div></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="terminal-block"><div class="metric-title">Textual Sentiment Orientation</div><div class="metric-value {sent_class}">{sent_lbl}</div></div>""", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # DOMAIN II: INTEGRATED MARKET TECHNICAL ANALYSIS
    # -------------------------------------------------------------------------
    left_track, right_track = st.columns([2, 1])
    
    with left_track:
        st.subheader("📈 Core Market Visualization & Analytics")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=historical_df['date'], open=historical_df['open'],
            high=historical_df['high'], low=historical_df['low'], close=historical_df['close'],
            name="Candlestick Price"
        ))
        
        historical_df['sma_20'] = historical_df['close'].rolling(window=min(20, len(historical_df))).mean()
        fig.add_trace(go.Scatter(
            x=historical_df['date'], y=historical_df['sma_20'], 
            mode='lines', name='20-Day Standard SMA', line=dict(color='#00C896', width=1.5)
        ))

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            template="plotly_dark", xaxis_rangeslider_visible=False, height=360,
            margin=dict(l=5, r=5, t=5, b=5),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(gridcolor="#1E293B"), yaxis=dict(gridcolor="#1E293B")
        )
        st.plotly_chart(fig, on_select="rerun")

    with right_track:
        st.subheader("🔮 Predictive Range Matrix")
        
        if ml_success:
            st.markdown(f"""
                <div class="terminal-block" style="text-align:center; padding: 1.5rem 1rem;">
                    <span class="metric-title">Point Estimate Model Forecast</span>
                    <div style="font-family:'JetBrains Mono'; font-size:2.5rem; font-weight:700; color:#F8FAFC; margin: 0.25rem 0;">
                        ${prediction_results['predicted_price']:.2f}
                    </div>
                    <span style="font-size:0.85rem; color:#94A3B8;">
                        Probabilistic 95% Expected Bounds:<br>
                        <b style="color:#F8FAFC;">${prediction_results['confidence_low']:.2f}</b> to <b style="color:#F8FAFC;">${prediction_results['confidence_high']:.2f}</b>
                    </span>
                </div>
            """, unsafe_allow_html=True)
            
            col_ind1, col_ind2 = st.columns(2)
            with col_ind1:
                st.markdown(f"""<div class="terminal-block" style="text-align:center;"><span class="metric-title">Relative Strength (RSI)</span><div style="font-family:'JetBrains Mono'; font-size:1.25rem; font-weight:700; margin-top:0.25rem;">{risk_metrics['rsi_14']:.2f}</div></div>""", unsafe_allow_html=True)
            with col_ind2:
                macd_delta = risk_metrics['macd_value'] - risk_metrics['macd_signal']
                macd_color = "signal-bullish" if macd_delta >= 0 else "signal-bearish"
                macd_text = "BULLISH CROSS" if macd_delta >= 0 else "BEARISH CROSS"
                st.markdown(f"""<div class="terminal-block" style="text-align:center;"><span class="metric-title">MACD Momentum</span><div class="{macd_color}" style="font-size:0.85rem; font-weight:700; margin-top:0.6rem;">{macd_text}</div></div>""", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # DOMAIN III: RISK REWARD ANALYTICS LAYER
    # -------------------------------------------------------------------------
    st.subheader("⚖️ Advanced Risk Analytics Diagnostics")
    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
    with col_r1:
        st.markdown(f"""<div class="terminal-block"><span class="metric-title">Latest Close</span><div class="metric-value">${risk_metrics['recent_close']:.2f}</div></div>""", unsafe_allow_html=True)
    with col_r2:
        st.markdown(f"""<div class="terminal-block"><span class="metric-title">Annualized Volatility</span><div class="metric-value">{risk_metrics['annualized_volatility']:.2f}%</div></div>""", unsafe_allow_html=True)
    with col_r3:
        st.markdown(f"""<div class="terminal-block"><span class="metric-title">Max Historical Drawdown</span><div class="metric-value" style="color:#FF6B6B;">{risk_metrics['max_drawdown']:.2f}%</div></div>""", unsafe_allow_html=True)
    with col_r4:
        st.markdown(f"""<div class="terminal-block"><span class="metric-title">Sharpe Performance (R_f=5%)</span><div class="metric-value" style="color:#00C896;">{risk_metrics['sharpe_ratio']:.2f}</div></div>""", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # DOMAIN IV: TEXTUAL INTELLIGENCE & INTERPRETABILITY LEDGERS
    # -------------------------------------------------------------------------
    col_b1, col_b2 = st.columns([1, 1])
    
    with col_b1:
        st.subheader("📰 Textual News Intelligence Feed")
        if not news_feed:
            st.write("No institutional text matrices matching reference parameters were indexed.")
        else:
            for idx, item in enumerate(news_feed[:3]):
                st.markdown(f"""
                <div class="terminal-block" style="background: #091220; border-color: #1E293B;">
                    <span class="info-tag" style="background:#1E293B; color:#94A3B8;">{item['source']}</span>
                    <h5 style="margin-top:0.4rem; margin-bottom:0.25rem;">{idx+1}. {item['title']}</h5>
                    <p style="color:#94A3B8; font-size:0.85rem; line-height:1.4; margin:0;">{item['summary']}</p>
                </div>
                """, unsafe_allow_html=True)

    with col_b2:
        st.subheader("🤖 Model Specification & Feature Explainability")
        
        st.markdown(f"""
            <div class="terminal-block" style="background:#091220; font-size:0.85rem; line-height:1.5;">
                <h5 style="margin-top:0; color:#F8FAFC; border-bottom:1px solid #1E293B; padding-bottom:0.3rem;">ℹ️ Regressor Parameters Specification</h5>
                <ul style="margin:0; padding-left:1.2rem; color:#94A3B8;">
                    <li><b>Core Algorithmic Target Model:</b> Random Forest Regressor Matrix</li>
                    <li><b>Tree Depth Limits:</b> 4 Levels (Regularization Bound Fixed)</li>
                    <li><b>Autoregressive Lookback Shifts:</b> {prediction_results.get('lag_days', 5)} Trading Days</li>
                    <li><b>Local Fit Optimization ($R^2$ Score Metric):</b> {(prediction_results['model_accuracy_score']*100):.2f}%</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        feat_data = prediction_results["feature_importances"]
        fig_feat = go.Figure(go.Bar(
            x=list(feat_data.values()), y=list(feat_data.keys()),
            orientation='h', marker_color='#3B82F6'
        ))
        fig_feat.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            template="plotly_dark", height=150, margin=dict(l=10, r=10, t=5, b=5),
            xaxis=dict(title="Gini Feature Importance Contribution (%)", gridcolor="#1E293B"),
            yaxis=dict(gridcolor="#1E293B")
        )
        st.plotly_chart(fig_feat, on_select="rerun")

    # 🚀 ADDITION 5: LEGAL RISK COMPLIANCE DISCLAIMER FOOTER CARD
    st.markdown("""
        <div style="border-top: 1px solid #1E293B; padding-top: 1.5rem; text-align: center; margin-top: 2rem; padding-bottom: 1rem;">
            <p style="font-size: 0.8rem; color: #64748B; max-width: 900px; margin: 0 auto; line-height: 1.5;">
                <b>System Disclaimer:</b> This terminal platform is designed exclusively for academic evaluation, portfolio visualization, and decision-support analytical workflow simulation. The quantitative forecasts, textual sentiment arrays, and technical indicator metrics rendered herein do not constitute formal investment advice, asset valuation endorsements, or fiduciary trading recommendations.
            </p>
        </div>
    """, unsafe_allow_html=True)