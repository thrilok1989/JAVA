# NIFTY/SENSEX Trader - Streamlit App

A comprehensive trading application for NIFTY and SENSEX indices with advanced technical analysis, signal generation, and automated trading capabilities.

## Features

- Real-time market data tracking for NIFTY and SENSEX
- Advanced technical analysis with multiple indicators
- Signal generation using Volume Order Blocks (VOB) and HTF Support/Resistance
- Option chain analysis
- Bias analysis and market sentiment tracking
- Telegram alerts integration
- Automated trade execution via Dhan API
- Advanced chart analysis with proximity alerts

## Prerequisites

- Python 3.8 or higher
- A Dhan trading account (for API access)
- Telegram bot token (for alerts)

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd JAVA
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure your settings in `config.py`:
   - Add your Dhan API credentials
   - Set up Telegram bot token
   - Configure trading parameters

## Running the App

### Local Development
```bash
streamlit run app.py
```

### Streamlit Cloud
1. Push this repository to GitHub
2. Go to [Streamlit Cloud](https://share.streamlit.io/)
3. Deploy from your GitHub repository
4. Add secrets in Streamlit Cloud settings (if needed)

## Configuration

The app uses the following configuration files:
- `config.py` - Main configuration for API keys and trading parameters
- `.streamlit/config.toml` - Streamlit UI and server settings

## Dependencies

Key dependencies include:
- streamlit - Web framework
- pandas - Data manipulation
- numpy - Numerical computing
- scipy - Scientific computing
- plotly - Interactive charts
- dhanhq - Dhan trading API
- python-telegram-bot - Telegram integration

See `requirements.txt` for complete list.

## Project Structure

```
JAVA/
├── app.py                          # Main Streamlit application
├── config.py                       # Configuration settings
├── requirements.txt                # Python dependencies
├── .streamlit/
│   └── config.toml                # Streamlit configuration
├── indicators/                     # Technical indicators
│   ├── volume_order_blocks.py
│   ├── htf_volume_footprint.py
│   └── ...
├── integrations/                   # External integrations
│   ├── ai_market_engine.py
│   └── news_fetcher.py
└── [various analysis modules]      # Market analysis modules
```

## Features Overview

### Signal Generators
- **VOB Signal Generator**: Volume Order Block based signals
- **HTF SR Signal Generator**: Higher Timeframe Support/Resistance signals

### Analysis Tools
- **Bias Analysis Pro**: Advanced market bias detection
- **Option Chain Analyzer**: Options data analysis
- **Advanced Chart Analysis**: Multi-timeframe chart analysis
- **Overall Market Sentiment**: AI-powered sentiment analysis

### Trading Features
- **Signal Manager**: Manages trading signals
- **Trade Executor**: Executes trades via Dhan API
- **Strike Calculator**: Calculates optimal strike prices

### Alerts & Notifications
- **Telegram Bot**: Real-time alerts via Telegram
- **Proximity Alerts**: Alert when price approaches key levels

## Market Hours

The app includes market hours scheduling to ensure operations only during trading hours (9:15 AM - 3:30 PM IST).

## Important Notes

- This app requires active market data subscriptions
- Ensure proper API credentials are configured
- Test in paper trading mode before live trading
- Monitor the app during market hours

## Troubleshooting

### ModuleNotFoundError
If you encounter module errors:
1. Ensure all dependencies are installed: `pip install -r requirements.txt`
2. Check Python version compatibility (3.8+)
3. Verify virtual environment is activated

### API Connection Issues
- Verify Dhan API credentials in `config.py`
- Check internet connectivity
- Ensure API rate limits are not exceeded

## Disclaimer

This software is for educational purposes only. Trading involves risk of financial loss. Always test thoroughly before using with real money.

## Support

For issues and questions, please check the application logs or contact support.
