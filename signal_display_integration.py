"""
Signal Display Integration for UI Tabs

Integrates Enhanced Signal Generator into the Streamlit UI across multiple tabs
"""

import streamlit as st
import pandas as pd
from typing import Optional, Dict
from datetime import datetime
import asyncio

from src.xgboost_ml_analyzer import XGBoostMLAnalyzer
from src.enhanced_signal_generator import EnhancedSignalGenerator, TradingSignal
from src.telegram_signal_manager import TelegramSignalManager
from telegram_alerts import TelegramBot


def initialize_signal_system():
    """Initialize signal generation components in session state"""
    if 'xgb_analyzer' not in st.session_state:
        st.session_state.xgb_analyzer = XGBoostMLAnalyzer()

    if 'signal_generator' not in st.session_state:
        st.session_state.signal_generator = EnhancedSignalGenerator(
            min_confidence=65.0,
            min_confluence=6
        )

    if 'telegram_manager' not in st.session_state:
        # Initialize Telegram bot if configured
        try:
            telegram_bot = TelegramBot()
            st.session_state.telegram_manager = TelegramSignalManager(
                telegram_bot=telegram_bot,
                enable_telegram=True
            )
        except Exception as e:
            st.session_state.telegram_manager = TelegramSignalManager(
                telegram_bot=None,
                enable_telegram=False
            )

    if 'current_signal' not in st.session_state:
        st.session_state.current_signal = None

    if 'signal_history' not in st.session_state:
        st.session_state.signal_history = []


def generate_trading_signal(
    df: pd.DataFrame,
    bias_results: Optional[Dict] = None,
    option_chain: Optional[Dict] = None,
    volatility_result: Optional[any] = None,
    oi_trap_result: Optional[any] = None,
    cvd_result: Optional[any] = None,
    participant_result: Optional[any] = None,
    liquidity_result: Optional[any] = None,
    ml_regime_result: Optional[any] = None,
    sentiment_score: float = 0.0,
    option_screener_data: Optional[Dict] = None,
    money_flow_signals: Optional[Dict] = None,
    deltaflow_signals: Optional[Dict] = None,
    overall_sentiment_data: Optional[Dict] = None,
    enhanced_market_data: Optional[Dict] = None,
    nifty_screener_data: Optional[Dict] = None,
    current_price: float = 0.0,
    atm_strike: Optional[float] = None
) -> Optional[TradingSignal]:
    """
    Generate trading signal from all available data

    Args:
        df: Price dataframe
        All other args: Data from various indicators
        current_price: Current spot price
        atm_strike: ATM strike price

    Returns:
        TradingSignal object or None if error
    """
    try:
        # Initialize system if needed
        initialize_signal_system()

        # Extract all 146 features
        features_df = st.session_state.xgb_analyzer.extract_features_from_all_tabs(
            df=df,
            bias_results=bias_results,
            option_chain=option_chain,
            volatility_result=volatility_result,
            oi_trap_result=oi_trap_result,
            cvd_result=cvd_result,
            participant_result=participant_result,
            liquidity_result=liquidity_result,
            ml_regime_result=ml_regime_result,
            sentiment_score=sentiment_score,
            option_screener_data=option_screener_data,
            money_flow_signals=money_flow_signals,
            deltaflow_signals=deltaflow_signals,
            overall_sentiment_data=overall_sentiment_data,
            enhanced_market_data=enhanced_market_data,
            nifty_screener_data=nifty_screener_data
        )

        # Get XGBoost prediction
        xgb_result = st.session_state.xgb_analyzer.predict(features_df)

        # Generate trading signal
        signal = st.session_state.signal_generator.generate_signal(
            xgboost_result=xgb_result,
            features_df=features_df,
            current_price=current_price,
            option_chain=option_chain,
            atm_strike=atm_strike
        )

        # Store in session state
        st.session_state.current_signal = signal

        # Add to history
        st.session_state.signal_history.append(signal)
        if len(st.session_state.signal_history) > 50:  # Keep last 50 signals
            st.session_state.signal_history.pop(0)

        return signal

    except Exception as e:
        st.error(f"Error generating signal: {e}")
        return None


async def send_signal_telegram(signal: TradingSignal, force: bool = False) -> Dict:
    """
    Send signal alert via Telegram

    Args:
        signal: TradingSignal to send
        force: Force send even if cooldown active

    Returns:
        Dict with send result
    """
    try:
        initialize_signal_system()
        result = await st.session_state.telegram_manager.send_signal_alert(signal, force=force)
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


def display_market_regime_assessment():
    """
    Display comprehensive market regime assessment
    Combines: Seller Bias + ATM Bias + Moment + Expiry + OI/PCR
    """
    # Check if we have minimum required data
    if 'nifty_option_screener_data' not in st.session_state:
        st.info("💡 Market regime assessment will appear after running option screener analysis.")
        return

    st.markdown("### 📊 FINAL ASSESSMENT (Seller + ATM Bias + Moment + Expiry + OI/PCR)")

    # Get data from session state with safe access
    nifty_screener_data = st.session_state.get('nifty_option_screener_data', {})
    overall_option_data = st.session_state.get('overall_option_data', {})
    option_chain = overall_option_data.get('NIFTY', {}) if isinstance(overall_option_data, dict) else {}
    volatility_result = st.session_state.get('volatility_regime_result', {})

    # Extract key metrics
    seller_bias = nifty_screener_data.get('seller_bias', {})
    atm_bias = nifty_screener_data.get('atm_bias', {})
    moment_detector = nifty_screener_data.get('moment_detector', {})
    oi_pcr_data = nifty_screener_data.get('oi_pcr_analysis', {})
    expiry_data = nifty_screener_data.get('expiry_spike', {})

    # Get spot price
    spot_price = option_chain.get('spot_price', 0)

    # Seller bias interpretation
    seller_direction = seller_bias.get('direction', 'NEUTRAL')
    seller_score = seller_bias.get('score', 0)
    seller_strength = seller_bias.get('strength', 'MILD')

    if seller_direction == "BEARISH":
        seller_text = f"Sellers aggressively WRITING CALLS (bearish conviction). Expecting price to STAY BELOW strikes."
        game_plan = "Bearish breakdown likely. Sellers confident in downside."
    elif seller_direction == "BULLISH":
        seller_text = f"Sellers aggressively WRITING PUTS (bullish conviction). Expecting price to STAY ABOVE strikes."
        game_plan = "Bullish continuation likely. Sellers confident in upside."
    else:
        seller_text = f"Sellers showing MIXED activity (neutral stance). Waiting for clear direction."
        game_plan = "Range-bound market. Wait for breakout confirmation."

    # ATM Bias
    atm_direction = atm_bias.get('bias', 'NEUTRAL')
    atm_score = atm_bias.get('atm_score', 0)

    if atm_direction == "BULLISH":
        atm_icon = "📈"
    elif atm_direction == "BEARISH":
        atm_icon = "📉"
    else:
        atm_icon = "⚖️"

    # Moment Detector
    moment_signal = moment_detector.get('signal', 'NEUTRAL')
    if moment_signal == "BUY":
        moment_text = "Strong buy pressure in orderbook."
    elif moment_signal == "SELL":
        moment_text = "Strong sell pressure in orderbook."
    else:
        moment_text = "Balanced orderbook pressure."

    # OI/PCR Analysis
    pcr_ratio = oi_pcr_data.get('pcr_ratio', 0)
    call_oi = oi_pcr_data.get('total_call_oi', 0)
    put_oi = oi_pcr_data.get('total_put_oi', 0)
    atm_concentration = oi_pcr_data.get('atm_concentration', 0)

    if pcr_ratio > 1.3:
        pcr_sentiment = "STRONG BULLISH"
    elif pcr_ratio > 1.0:
        pcr_sentiment = "MILD BULLISH"
    elif pcr_ratio > 0.7:
        pcr_sentiment = "MILD BEARISH"
    else:
        pcr_sentiment = "STRONG BEARISH"

    # Expiry context
    days_to_expiry = expiry_data.get('days_to_expiry', 0)

    # Key levels from Volatility Sentiment
    volatility_sentiment = volatility_result.get('volatility_sentiment', {})
    support_level = volatility_sentiment.get('support', spot_price - 50)
    resistance_level = volatility_sentiment.get('resistance', spot_price + 50)

    # Max Pain from OI/PCR
    max_pain = oi_pcr_data.get('max_pain', spot_price)

    # Max OI walls
    max_call_strike = oi_pcr_data.get('max_call_oi_strike', 0)
    max_put_strike = oi_pcr_data.get('max_put_oi_strike', 0)

    # Display the assessment
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                padding: 20px; border-radius: 10px; border: 2px solid #0f3460; margin-bottom: 20px;'>
        <div style='color: white; line-height: 1.8;'>
            <p style='font-size: 16px; margin-bottom: 15px;'>
                <strong style='color: #00d4ff;'>Market Makers are telling us:</strong> {seller_text}
            </p>

            <p style='font-size: 15px; margin-bottom: 15px;'>
                <strong style='color: #00d4ff;'>ATM Zone Analysis:</strong> ATM Bias: {atm_icon} <strong>{atm_direction}</strong> ({atm_score:.2f} score)
            </p>

            <p style='font-size: 15px; margin-bottom: 15px;'>
                <strong style='color: #00d4ff;'>Their game plan:</strong> {game_plan}
            </p>

            <p style='font-size: 15px; margin-bottom: 15px;'>
                <strong style='color: #00d4ff;'>Moment Detector:</strong> {moment_text}
            </p>

            <p style='font-size: 15px; margin-bottom: 15px;'>
                <strong style='color: #00d4ff;'>OI/PCR Analysis:</strong> PCR: {pcr_ratio:.2f} ({pcr_sentiment}) |
                CALL OI: {call_oi:,.0f} | PUT OI: {put_oi:,.0f} | ATM Conc: {atm_concentration:.1f}%
            </p>

            <p style='font-size: 15px; margin-bottom: 15px;'>
                <strong style='color: #00d4ff;'>Expiry Context:</strong> Expiry in {days_to_expiry:.1f} days
            </p>

            <p style='font-size: 15px; margin-bottom: 15px;'>
                <strong style='color: #00d4ff;'>Key defense levels:</strong>
                ₹{support_level:,.0f} (Support) | ₹{resistance_level:,.0f} (Resistance)
            </p>

            <p style='font-size: 15px; margin-bottom: 15px;'>
                <strong style='color: #00d4ff;'>Max OI Walls:</strong>
                CALL: ₹{max_call_strike:,.0f} | PUT: ₹{max_put_strike:,.0f}
            </p>

            <p style='font-size: 15px; margin-bottom: 0;'>
                <strong style='color: #00d4ff;'>Preferred price level:</strong>
                ₹{max_pain:,.0f} (Max Pain)
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)


def display_signal_card(signal: TradingSignal):
    """
    Display trading signal as a prominent card

    Args:
        signal: TradingSignal to display
    """
    if signal is None:
        return

    # Determine colors based on signal type and direction
    if signal.signal_type == "ENTRY":
        if signal.direction == "LONG":
            bg_color = "#1a4d2e"
            border_color = "#00ff88"
            icon = "🚀"
        else:  # SHORT
            bg_color = "#4d1a1a"
            border_color = "#ff4444"
            icon = "🔻"
    elif signal.signal_type == "EXIT":
        bg_color = "#4d3d1a"
        border_color = "#ffa500"
        icon = "🔻"
    elif signal.signal_type == "WAIT":
        bg_color = "#2d2d2d"
        border_color = "#6495ED"
        icon = "⏸️"
    elif signal.signal_type == "DIRECTION_CHANGE":
        bg_color = "#3d2d4d"
        border_color = "#9370DB"
        icon = "🔄"
    else:  # BIAS_CHANGE
        bg_color = "#2d3d4d"
        border_color = "#4682B4"
        icon = "⚠️"

    # Format timestamp
    timestamp_str = signal.timestamp.strftime('%H:%M:%S') if signal.timestamp else ""

    # Build signal card
    st.markdown(f"""
    <div style='background: {bg_color};
                padding: 25px; border-radius: 15px; margin-bottom: 20px;
                border: 2px solid {border_color}; box-shadow: 0 4px 6px rgba(0,0,0,0.3);'>
        <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;'>
            <h2 style='margin: 0; color: white; font-size: 28px;'>{icon} {signal.signal_type} SIGNAL</h2>
            <span style='color: #888; font-size: 14px;'>⏰ {timestamp_str}</span>
        </div>
        <div style='font-size: 24px; color: {border_color}; font-weight: bold; margin-bottom: 15px;'>
            Direction: {signal.direction}
        </div>
    """, unsafe_allow_html=True)

    # Display based on signal type
    if signal.signal_type == "ENTRY":
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 📊 Option Details")
            st.markdown(f"**Type:** {signal.option_type}")
            st.markdown(f"**Strike:** {signal.strike}")
            st.markdown(f"**Entry:** ₹{signal.entry_price:.2f}")
            st.markdown(f"**Range:** ₹{signal.entry_range_low:.2f} - ₹{signal.entry_range_high:.2f}")

        with col2:
            st.markdown("### 🎯 Targets & Risk")
            st.markdown(f"**Stop Loss:** ₹{signal.stop_loss:.2f}")
            if signal.target_1:
                pct1 = ((signal.target_1 / signal.entry_price - 1) * 100)
                st.markdown(f"**T1:** ₹{signal.target_1:.2f} (+{pct1:.1f}%)")
            if signal.target_2:
                pct2 = ((signal.target_2 / signal.entry_price - 1) * 100)
                st.markdown(f"**T2:** ₹{signal.target_2:.2f} (+{pct2:.1f}%)")
            if signal.target_3:
                pct3 = ((signal.target_3 / signal.entry_price - 1) * 100)
                st.markdown(f"**T3:** ₹{signal.target_3:.2f} (+{pct3:.1f}%)")
            st.markdown(f"**R:R Ratio:** {signal.risk_reward_ratio:.2f}")

        with col3:
            st.markdown("### 💪 Strength")
            st.markdown(f"**Confidence:** {signal.confidence:.1f}%")
            st.markdown(f"**Confluence:** {signal.confluence}/{signal.total_indicators}")
            confluence_pct = (signal.confluence / signal.total_indicators * 100) if signal.total_indicators > 0 else 0
            st.progress(confluence_pct / 100)
            st.markdown(f"**Market Regime:** {signal.market_regime}")
            st.markdown(f"**XGBoost:** {signal.xgboost_prediction} ({signal.xgboost_probability*100:.1f}%)")

        st.markdown("---")
        st.markdown(f"**💡 Reason:** {signal.reason}")

    elif signal.signal_type == "EXIT":
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Action:** Close all positions")
            st.markdown(f"**Market Regime:** {signal.market_regime}")
        with col2:
            st.markdown(f"**Confidence:** {signal.confidence:.1f}%")
            st.markdown(f"**Confluence:** {signal.confluence}/{signal.total_indicators}")
        st.markdown("---")
        st.markdown(f"**💡 Reason:** {signal.reason}")

    elif signal.signal_type == "WAIT":
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Status:** No clear edge")
            st.markdown(f"**XGBoost:** {signal.xgboost_prediction}")
        with col2:
            st.markdown(f"**Confidence:** {signal.confidence:.1f}%")
            st.markdown(f"**Confluence:** {signal.confluence}/{signal.total_indicators}")
        st.markdown("---")
        st.markdown(f"**💡 Reason:** {signal.reason}")

    else:  # DIRECTION_CHANGE or BIAS_CHANGE
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**New Direction:** {signal.direction}")
            st.markdown(f"**Market Regime:** {signal.market_regime}")
        with col2:
            st.markdown(f"**Confidence:** {signal.confidence:.1f}%")
            st.markdown(f"**Confluence:** {signal.confluence}/{signal.total_indicators}")
        st.markdown("---")
        st.markdown(f"**💡 Reason:** {signal.reason}")

    st.markdown("</div>", unsafe_allow_html=True)

    # Action buttons
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button("📱 Send to Telegram", key=f"telegram_{signal.timestamp}"):
            with st.spinner("Sending to Telegram..."):
                result = asyncio.run(send_signal_telegram(signal, force=True))
                if result.get('success'):
                    st.success("✅ Sent to Telegram!")
                else:
                    st.error(f"❌ Failed: {result.get('reason', 'Unknown error')}")

    with col2:
        if signal.signal_type == "ENTRY":
            if st.button("📝 Create Trade Setup", key=f"setup_{signal.timestamp}"):
                # Store signal for Trade Setup tab
                st.session_state.signal_for_setup = signal
                st.success("✅ Signal saved! Go to Trade Setup tab")

    with col3:
        st.caption(f"Signal generated at {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S') if signal.timestamp else 'N/A'}")


def display_signal_history():
    """Display recent signal history"""
    if 'signal_history' not in st.session_state or not st.session_state.signal_history:
        st.info("No signal history available yet")
        return

    st.markdown("### 📊 Recent Signals")

    # Create dataframe from history
    history_data = []
    for sig in st.session_state.signal_history[-10:]:  # Last 10 signals
        history_data.append({
            'Time': sig.timestamp.strftime('%H:%M:%S') if sig.timestamp else 'N/A',
            'Type': sig.signal_type,
            'Direction': sig.direction,
            'Confidence': f"{sig.confidence:.1f}%",
            'Confluence': f"{sig.confluence}/{sig.total_indicators}",
            'XGBoost': sig.xgboost_prediction
        })

    if history_data:
        df = pd.DataFrame(history_data)
        st.dataframe(df, use_container_width=True)


def display_telegram_stats():
    """Display Telegram alert statistics"""
    initialize_signal_system()

    if st.session_state.telegram_manager:
        stats = st.session_state.telegram_manager.get_statistics()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Alerts", stats['total_alerts_generated'])

        with col2:
            st.metric("Sent", stats['total_alerts_sent'])

        with col3:
            st.metric("Blocked (Cooldown)", stats['total_alerts_blocked_cooldown'])

        with col4:
            st.metric("Errors", stats['total_telegram_errors'])

        # Show cooldown status
        with st.expander("📊 Cooldown Status"):
            for alert_type, status in stats['cooldown_status'].items():
                if 'error' not in status:
                    active = "🔴 ACTIVE" if status['cooldown_active'] else "🟢 READY"
                    remaining = f" ({status['time_remaining']:.0f}s remaining)" if status['cooldown_active'] else ""
                    st.text(f"{alert_type}: {active}{remaining}")


def get_signal_for_trade_setup() -> Optional[TradingSignal]:
    """
    Get the most recent ENTRY signal for Trade Setup auto-fill

    Returns:
        TradingSignal or None if no ENTRY signal available
    """
    if 'signal_for_setup' in st.session_state:
        return st.session_state.signal_for_setup

    if 'current_signal' in st.session_state:
        signal = st.session_state.current_signal
        if signal and signal.signal_type == "ENTRY":
            return signal

    return None


def apply_signal_to_trade_setup(signal: TradingSignal) -> Dict:
    """
    Convert TradingSignal to Trade Setup parameters

    Args:
        signal: TradingSignal to convert

    Returns:
        Dict with trade setup parameters
    """
    if signal.signal_type != "ENTRY":
        return None

    # Determine direction from option type
    direction = signal.option_type  # "CALL" or "PUT"

    # Calculate VOB levels from signal
    # For CALL: support = strike - buffer, resistance = target level
    # For PUT: support = target level, resistance = strike + buffer

    if direction == "CALL":
        # Call means bullish, so support is below and resistance is target
        vob_support = signal.strike - 100  # Buffer below strike
        vob_resistance = signal.strike + 200  # Target above strike
    else:  # PUT
        # Put means bearish, so resistance is above and support is target
        vob_support = signal.strike - 200  # Target below strike
        vob_resistance = signal.strike + 100  # Buffer above strike

    return {
        'index': 'NIFTY',  # Default, can be overridden
        'direction': direction,
        'vob_support': vob_support,
        'vob_resistance': vob_resistance,
        'entry_price': signal.entry_price,
        'stop_loss': signal.stop_loss,
        'target_1': signal.target_1,
        'target_2': signal.target_2,
        'target_3': signal.target_3,
        'strike': signal.strike,
        'confidence': signal.confidence,
        'signal': signal
    }


def display_signal_autofill_banner():
    """
    Display banner at top of Trade Setup tab to allow signal auto-fill
    """
    signal = get_signal_for_trade_setup()

    if signal:
        st.info(f"💡 **Signal Available:** {signal.direction} {signal.option_type} at strike {signal.strike} "
                f"(Confidence: {signal.confidence:.1f}%)")

        col1, col2 = st.columns([3, 1])

        with col1:
            st.caption(f"Generated at {signal.timestamp.strftime('%H:%M:%S') if signal.timestamp else 'N/A'} | "
                      f"Entry: ₹{signal.entry_price:.2f} | SL: ₹{signal.stop_loss:.2f}")

        with col2:
            if st.button("📥 Auto-Fill from Signal", type="primary", use_container_width=True):
                setup_params = apply_signal_to_trade_setup(signal)
                st.session_state.signal_setup_params = setup_params
                st.success("✅ Trade Setup auto-filled from signal!")
                st.rerun()

        st.markdown("---")


def create_active_signal_from_trading_signal(signal: TradingSignal, signal_manager) -> Optional[str]:
    """
    Auto-create an Active Signal entry when ENTRY signal is generated

    Args:
        signal: TradingSignal with signal_type="ENTRY"
        signal_manager: SignalManager instance from session state

    Returns:
        signal_id if created, None otherwise
    """
    if signal.signal_type != "ENTRY":
        return None

    try:
        # Calculate VOB levels from signal
        if signal.option_type == "CALL":
            vob_support = signal.strike - 100
            vob_resistance = signal.strike + 200
        else:  # PUT
            vob_support = signal.strike - 200
            vob_resistance = signal.strike + 100

        # Create setup using signal_manager
        signal_id = signal_manager.create_setup(
            index='NIFTY',
            direction=signal.option_type,
            vob_support=vob_support,
            vob_resistance=vob_resistance
        )

        # Store signal reference
        setup = signal_manager.get_setup(signal_id)
        if setup:
            setup['ai_signal'] = signal
            setup['auto_created'] = True

        return signal_id

    except Exception as e:
        st.error(f"Failed to auto-create signal: {e}")
        return None


def check_and_display_exit_alerts(active_positions: Dict, current_signal: Optional[TradingSignal]):
    """
    Check if EXIT signal matches any active positions and display alerts

    Args:
        active_positions: Dict of active positions from session state
        current_signal: Current TradingSignal (may be EXIT type)
    """
    if not current_signal or current_signal.signal_type != "EXIT":
        return

    # Display EXIT alert
    st.warning("⚠️ **EXIT SIGNAL DETECTED** - Consider closing positions!")

    st.markdown(f"""
    <div style='background: #4d1a1a; padding: 20px; border-radius: 10px; border: 2px solid #ff4444; margin-bottom: 20px;'>
        <h3 style='color: #ff4444; margin: 0 0 10px 0;'>🚨 EXIT ALERT</h3>
        <p style='color: white; margin: 0;'>
            <strong>Direction:</strong> {current_signal.direction}<br>
            <strong>Confidence:</strong> {current_signal.confidence:.1f}%<br>
            <strong>Reason:</strong> {current_signal.reason}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Check if any positions match the exit direction
    matching_positions = []
    for order_id, pos in active_positions.items():
        if pos['status'] == 'active':
            # If EXIT is for LONG, close CALL positions
            # If EXIT is for SHORT, close PUT positions
            if (current_signal.direction == "LONG" and pos['direction'] == "CALL") or \
               (current_signal.direction == "SHORT" and pos['direction'] == "PUT"):
                matching_positions.append(order_id)

    if matching_positions:
        st.error(f"⚠️ {len(matching_positions)} active position(s) affected by EXIT signal!")

        # Add quick exit buttons
        if st.button("🚨 EXIT ALL MATCHING POSITIONS", type="primary", use_container_width=True):
            for order_id in matching_positions:
                active_positions[order_id]['status'] = 'exited'
                active_positions[order_id]['exit_reason'] = 'AI EXIT SIGNAL'
            st.success(f"✅ Exited {len(matching_positions)} position(s)")
            st.rerun()
