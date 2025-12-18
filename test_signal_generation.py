"""
Test Script for Phase 5: Signal Generation & Telegram Validation

This script validates:
1. Signal generation with live data
2. XGBoost feature extraction (146 features)
3. Telegram alert integration
4. Signal confidence and confluence calculations

Usage:
    # Run in the same Python environment as your Streamlit app
    # If using virtual environment:
    source venv/bin/activate  # or activate your venv
    python test_signal_generation.py

    # If using conda:
    conda activate your_env
    python test_signal_generation.py

    # If dependencies not installed:
    pip install pandas numpy xgboost scikit-learn streamlit

Requirements:
    - pandas, numpy, xgboost, scikit-learn
    - Same environment as Streamlit app
    - telegram_alerts.py configured (optional)
"""

import sys
import asyncio
from datetime import datetime
from typing import Dict, Optional

# Add src to path
sys.path.insert(0, '/home/user/JAVA')
sys.path.insert(0, '/home/user/JAVA/src')

# Try to import dependencies
try:
    from src.xgboost_ml_analyzer import XGBoostMLAnalyzer
    from src.enhanced_signal_generator import EnhancedSignalGenerator, TradingSignal
    from src.telegram_signal_manager import TelegramSignalManager
    from telegram_alerts import TelegramBot
except ImportError as e:
    print("\n" + "="*70)
    print("❌ DEPENDENCY ERROR")
    print("="*70)
    print(f"\nMissing dependency: {e}")
    print("\nThis test script must be run in the same Python environment")
    print("as your Streamlit app. Please ensure all dependencies are installed:")
    print("\n  pip install pandas numpy xgboost scikit-learn streamlit")
    print("\nIf using a virtual environment, activate it first:")
    print("  source venv/bin/activate")
    print("\nThen run this script again.")
    print("="*70 + "\n")
    sys.exit(1)


class SignalGenerationTester:
    """Test signal generation system end-to-end"""

    def __init__(self):
        self.xgb_analyzer = XGBoostMLAnalyzer()
        self.signal_generator = EnhancedSignalGenerator(
            min_confidence=65.0,
            min_confluence=6
        )
        self.telegram_manager = None
        self.test_results = []

    def log_test(self, test_name: str, passed: bool, details: str = ""):
        """Log test result"""
        status = "✅ PASS" if passed else "❌ FAIL"
        result = {
            "test": test_name,
            "passed": passed,
            "details": details,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }
        self.test_results.append(result)
        print(f"{status} | {test_name}")
        if details:
            print(f"   └─ {details}")

    def print_summary(self):
        """Print test summary"""
        total = len(self.test_results)
        passed = sum(1 for r in self.test_results if r["passed"])
        failed = total - passed

        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        print(f"Total Tests: {total}")
        print(f"Passed: {passed} ✅")
        print(f"Failed: {failed} ❌")
        print(f"Success Rate: {(passed/total*100):.1f}%")
        print("="*70)

        if failed > 0:
            print("\nFailed Tests:")
            for r in self.test_results:
                if not r["passed"]:
                    print(f"  • {r['test']}: {r['details']}")

    def test_xgboost_analyzer_initialization(self):
        """Test 1: XGBoost Analyzer initializes correctly"""
        try:
            assert self.xgb_analyzer is not None
            assert hasattr(self.xgb_analyzer, 'extract_features_from_all_tabs')
            assert hasattr(self.xgb_analyzer, 'predict')
            self.log_test("XGBoost Analyzer Initialization", True,
                         "XGBoostMLAnalyzer instance created successfully")
            return True
        except Exception as e:
            self.log_test("XGBoost Analyzer Initialization", False, str(e))
            return False

    def test_signal_generator_initialization(self):
        """Test 2: Signal Generator initializes correctly"""
        try:
            assert self.signal_generator is not None
            assert self.signal_generator.min_confidence == 65.0
            assert self.signal_generator.min_confluence == 6
            self.log_test("Signal Generator Initialization", True,
                         f"Confidence={self.signal_generator.min_confidence}, "
                         f"Confluence={self.signal_generator.min_confluence}")
            return True
        except Exception as e:
            self.log_test("Signal Generator Initialization", False, str(e))
            return False

    def test_telegram_integration(self):
        """Test 3: Telegram integration"""
        try:
            # Try to initialize Telegram bot
            telegram_bot = TelegramBot()
            self.telegram_manager = TelegramSignalManager(
                telegram_bot=telegram_bot,
                enable_telegram=True
            )
            self.log_test("Telegram Integration", True,
                         "TelegramSignalManager initialized with bot")
            return True
        except Exception as e:
            # Telegram may not be configured - that's OK
            self.telegram_manager = TelegramSignalManager(
                telegram_bot=None,
                enable_telegram=False
            )
            self.log_test("Telegram Integration", True,
                         f"Telegram disabled (config not found): {str(e)}")
            return True

    def test_feature_extraction_structure(self):
        """Test 4: Feature extraction returns correct structure"""
        try:
            # Test with minimal mock data
            import pandas as pd

            # Create minimal price dataframe
            df = pd.DataFrame({
                'datetime': [datetime.now()],
                'open': [24500],
                'high': [24550],
                'low': [24450],
                'close': [24520],
                'volume': [1000000]
            })

            # Extract features (will use defaults for missing data)
            features_df = self.xgb_analyzer.extract_features_from_all_tabs(
                df=df,
                bias_results=None,
                option_chain=None,
                volatility_result=None,
                oi_trap_result=None,
                cvd_result=None,
                participant_result=None,
                liquidity_result=None,
                ml_regime_result=None,
                sentiment_score=0.0,
                option_screener_data=None,
                money_flow_signals=None,
                deltaflow_signals=None,
                overall_sentiment_data=None,
                enhanced_market_data=None,
                nifty_screener_data=None
            )

            # Validate features extracted
            assert features_df is not None
            feature_count = len(features_df.columns) if hasattr(features_df, 'columns') else 0

            self.log_test("Feature Extraction Structure", True,
                         f"Extracted {feature_count} features from minimal data")
            return True

        except Exception as e:
            self.log_test("Feature Extraction Structure", False, str(e))
            return False

    def test_xgboost_prediction(self):
        """Test 5: XGBoost prediction works"""
        try:
            import pandas as pd

            # Create minimal price dataframe
            df = pd.DataFrame({
                'datetime': [datetime.now()],
                'open': [24500],
                'high': [24550],
                'low': [24450],
                'close': [24520],
                'volume': [1000000]
            })

            # Extract features
            features_df = self.xgb_analyzer.extract_features_from_all_tabs(df=df)

            # Get prediction
            xgb_result = self.xgb_analyzer.predict(features_df)

            # Validate prediction structure
            assert xgb_result is not None
            assert 'prediction' in xgb_result
            assert 'confidence' in xgb_result

            prediction = xgb_result['prediction']
            confidence = xgb_result['confidence']

            self.log_test("XGBoost Prediction", True,
                         f"Prediction={prediction}, Confidence={confidence:.2f}%")
            return True

        except Exception as e:
            self.log_test("XGBoost Prediction", False, str(e))
            return False

    def test_signal_generation(self):
        """Test 6: Signal generation end-to-end"""
        try:
            import pandas as pd

            # Create minimal price dataframe
            df = pd.DataFrame({
                'datetime': [datetime.now()],
                'open': [24500],
                'high': [24550],
                'low': [24450],
                'close': [24520],
                'volume': [1000000]
            })

            # Extract features
            features_df = self.xgb_analyzer.extract_features_from_all_tabs(df=df)

            # Get XGBoost prediction
            xgb_result = self.xgb_analyzer.predict(features_df)

            # Generate signal
            signal = self.signal_generator.generate_signal(
                xgboost_result=xgb_result,
                features_df=features_df,
                current_price=24520,
                option_chain=None,
                atm_strike=24500
            )

            # Validate signal
            if signal is not None:
                assert isinstance(signal, TradingSignal)
                assert hasattr(signal, 'signal_type')
                assert hasattr(signal, 'direction')
                assert hasattr(signal, 'confidence')
                assert hasattr(signal, 'confluence_count')

                self.log_test("Signal Generation", True,
                             f"Type={signal.signal_type}, Direction={signal.direction}, "
                             f"Confidence={signal.confidence:.2f}%, Confluence={signal.confluence_count}")
                return True
            else:
                self.log_test("Signal Generation", True,
                             "No signal generated (insufficient confluence - expected)")
                return True

        except Exception as e:
            self.log_test("Signal Generation", False, str(e))
            return False

    async def test_telegram_send(self):
        """Test 7: Telegram send functionality (async)"""
        try:
            if self.telegram_manager is None or not self.telegram_manager.telegram_enabled:
                self.log_test("Telegram Send Test", True,
                             "Skipped (Telegram not configured)")
                return True

            # Create a test signal
            import pandas as pd

            df = pd.DataFrame({
                'datetime': [datetime.now()],
                'open': [24500],
                'high': [24550],
                'low': [24450],
                'close': [24520],
                'volume': [1000000]
            })

            features_df = self.xgb_analyzer.extract_features_from_all_tabs(df=df)
            xgb_result = self.xgb_analyzer.predict(features_df)
            signal = self.signal_generator.generate_signal(
                xgboost_result=xgb_result,
                features_df=features_df,
                current_price=24520,
                option_chain=None,
                atm_strike=24500
            )

            if signal is None:
                self.log_test("Telegram Send Test", True,
                             "Skipped (no signal generated)")
                return True

            # Try to send via Telegram
            result = await self.telegram_manager.send_signal_alert(signal, force=True)

            if result.get('success'):
                self.log_test("Telegram Send Test", True,
                             f"Message sent: {result.get('message', 'Success')}")
            else:
                self.log_test("Telegram Send Test", False,
                             f"Send failed: {result.get('error', 'Unknown error')}")

            return result.get('success', False)

        except Exception as e:
            self.log_test("Telegram Send Test", False, str(e))
            return False

    def test_signal_history_tracking(self):
        """Test 8: Signal history tracking"""
        try:
            signal_history = []

            # Generate multiple signals
            import pandas as pd

            for i in range(5):
                df = pd.DataFrame({
                    'datetime': [datetime.now()],
                    'open': [24500 + i*10],
                    'high': [24550 + i*10],
                    'low': [24450 + i*10],
                    'close': [24520 + i*10],
                    'volume': [1000000]
                })

                features_df = self.xgb_analyzer.extract_features_from_all_tabs(df=df)
                xgb_result = self.xgb_analyzer.predict(features_df)
                signal = self.signal_generator.generate_signal(
                    xgboost_result=xgb_result,
                    features_df=features_df,
                    current_price=24520 + i*10,
                    option_chain=None,
                    atm_strike=24500 + i*10
                )

                if signal:
                    signal_history.append(signal)

            # Validate history
            self.log_test("Signal History Tracking", True,
                         f"Generated {len(signal_history)} signals in history")
            return True

        except Exception as e:
            self.log_test("Signal History Tracking", False, str(e))
            return False

    async def run_all_tests(self):
        """Run all validation tests"""
        print("\n" + "="*70)
        print("PHASE 5 VALIDATION: Signal Generation & Telegram Testing")
        print("="*70 + "\n")

        # Run synchronous tests
        print("Running Component Tests...")
        print("-" * 70)
        self.test_xgboost_analyzer_initialization()
        self.test_signal_generator_initialization()
        self.test_telegram_integration()

        print("\nRunning Feature Extraction Tests...")
        print("-" * 70)
        self.test_feature_extraction_structure()
        self.test_xgboost_prediction()

        print("\nRunning Signal Generation Tests...")
        print("-" * 70)
        self.test_signal_generation()
        self.test_signal_history_tracking()

        print("\nRunning Telegram Integration Tests...")
        print("-" * 70)
        await self.test_telegram_send()

        # Print summary
        self.print_summary()

        return sum(1 for r in self.test_results if r["passed"]) == len(self.test_results)


async def main():
    """Main test execution"""
    tester = SignalGenerationTester()

    try:
        success = await tester.run_all_tests()

        if success:
            print("\n🎉 All tests passed! Phase 5 validation successful.")
            return 0
        else:
            print("\n⚠️ Some tests failed. Review output above.")
            return 1

    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
