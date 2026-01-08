import logging
import calendar
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import date
import warnings
from typing import Union, List

warnings.filterwarnings('ignore')

# Local imports (centralized via helpers facade)
from helpers.app_helpers import (
    apply_style,
    download_data,
    download_multiple_tickers,
    analyze_month_patterns,
    get_month_summary,
    compute_return_distribution,
    fit_distribution_to_returns,
    create_monthly_chart,
    create_correlation_heatmap,
    create_relative_performance_chart,
    create_return_distribution_chart,
    compute_correlation_matrix,
    calculate_relative_performance,
    calculate_statistics,
    render_sidebar,
    TRADING_MONTHS,
    DEFAULT_TICKER,
    POPULAR_ASSETS,
    WIN_RATE_THRESHOLDS,
    PERCENTAGE_DECIMALS,
    WIN_RATE_DECIMALS,
    MONTH_SPLIT_RATIO,
    COLOR_POSITIVE,
    COLOR_NEGATIVE,
    COLOR_TEXT_SECONDARY,
    FONT_SIZE_TITLE,
    CHART_HEIGHT,
    create_metric_card,
    get_win_rate_color,
    get_win_rate_status,
    get_return_color,
    create_insight_box,
    create_metric_cards_row,
    get_ticker_display_name,
    get_distribution_stats,
    get_goodness_of_fit,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

apply_style()

# ============================================================================
# Page Header
# ============================================================================
st.markdown('<h1 class="main-header">📈 Financial Analysis Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Analyze monthly patterns, momentum trends, correlations, and performance metrics</p>', unsafe_allow_html=True)

# ============================================================================
# Sidebar Configuration
# ============================================================================
tickers, start_date_str, end_date_str, show_trend_lines, show_summary_stats, analysis_mode = render_sidebar()

# Convert single ticker to list for uniform handling
if isinstance(tickers, str):
    tickers_list = [tickers]
else:
    tickers_list = tickers

# ============================================================================
# Cached Analysis Functions
# ============================================================================
@st.cache_data(ttl=3600)
def fetch_and_validate_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Cached data fetch and validation."""
    data, error = download_data(ticker, start_date, end_date)
    if error:
        logger.error(f"Data fetch error: {error}")
        st.error(error)
        return None
    return data


@st.cache_data(ttl=3600)
def fetch_multiple_and_validate(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Cached multi-ticker data fetch."""
    data, error = download_multiple_tickers(tickers, start_date, end_date)
    if error:
        logger.error(f"Multi-ticker fetch error: {error}")
        st.error(error)
        return None
    return data


@st.cache_data(ttl=3600)
def analyze_all_months(data: pd.DataFrame) -> dict:
    """Cache analysis results for all months.
    
    Returns dict mapping month_num -> (weekly_returns, momentum, win_rates, years)
    """
    if data is None or data.empty:
        return {}
    
    results = {}
    for month_num in TRADING_MONTHS:
        results[month_num] = analyze_month_patterns(month_num, data)
    
    logger.info(f"Cached analysis for {len(results)} months")
    return results


def build_overview_summary(all_analysis: dict, data: pd.DataFrame) -> pd.DataFrame:
    """Build summary dataframe for overview tab."""
    summary_data = []
    
    for month_num, (weekly_returns, momentum, win_rates, years) in all_analysis.items():
        if not years:
            continue
        # Compute basic aggregates
        avg_momentum = np.mean(list(momentum.values())) if momentum else 0
        avg_win_rate = np.mean(list(win_rates.values())) if win_rates else 0

        # Compute per-month summary (avg, median, volatility, positive years)
        try:
            ms = get_month_summary(month_num, data)
            monthly_avg = ms['avg_monthly_return']
            median_ret = ms['median_monthly_return']
            vol = ms['volatility']
            positive_years = ms['positive_years']
            years_count = len(ms['years_processed'])
        except Exception:
            # Fallback values when summary computation fails
            monthly_avg = (max(weekly_returns.values()) * 100) if weekly_returns else 0
            median_ret = monthly_avg
            vol = 0.0
            positive_years = 0
            years_count = len(years)

        summary_data.append({
            'Month': calendar.month_name[month_num],
            'Monthly Return': monthly_avg,
            'Median_Return': median_ret,
            'Volatility': vol,
            'Positive_Years': positive_years,
            'Avg Momentum': avg_momentum * 100,
            'Win Rate': avg_win_rate,
            'Years': years_count
        })
    
    return pd.DataFrame(summary_data) if summary_data else None


# ============================================================================
# Main Application
# ============================================================================
def main():
    try:
        if analysis_mode == "Single Ticker":
            render_single_ticker_analysis()
        else:
            render_multi_ticker_analysis()
    
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        st.error(f"An unexpected error occurred: {str(e)}")


def render_single_ticker_analysis():
    """Render single-ticker monthly patterns analysis."""
    ticker = tickers_list[0]
    
    try:
        # Fetch data
        with st.spinner(f"Downloading data for {ticker}..."):
            data = fetch_and_validate_data(ticker, start_date_str, end_date_str)
        
        if data is None or data.empty:
            st.error("No data available for the selected parameters.")
            st.stop()
        
        # Show success message
        st.markdown(
            f'<div class="success-message">✅ Data loaded successfully for {ticker}</div>',
            unsafe_allow_html=True
        )
        
        # Data info metrics
        create_metric_cards_row([
            {
                'value': f"{start_date_str} to {end_date_str}",
                'label': 'Date Range'
            },
            {
                'value': f"{len(data):,}",
                'label': 'Total Days'
            },
            {
                'value': str(data['Year'].nunique()),
                'label': 'Years Covered'
            },
        ], num_columns=3)

        # Icon-only action panel (download, refresh, view raw)
        # Prepare CSV for download
        try:
            csv_str = data.to_csv(index=True)
        except Exception:
            csv_str = None

        # Initialize session state for raw data toggle
        if 'show_raw' not in st.session_state:
            st.session_state.show_raw = False

        cols_icons = st.columns([1,1,1,6])

        # Download button (icon-only)
        with cols_icons[0]:
            if csv_str is not None:
                st.download_button(label='⬇️', data=csv_str, file_name=f"{ticker}_{start_date_str}_{end_date_str}.csv", mime='text/csv', help='Download dataset as CSV')
            else:
                st.button('⬇️', help='Download not available', disabled=True)

        # Refresh button
        with cols_icons[1]:
            if st.button('🔄', help='Refresh data'):
                st.experimental_rerun()

        # Toggle raw data view
        with cols_icons[2]:
            if st.button('📄', help='Toggle raw data view'):
                st.session_state.show_raw = not st.session_state.show_raw

        # Small spacer column
        with cols_icons[3]:
            # If raw view toggled, show data in an expander
            if st.session_state.get('show_raw'):
                with st.expander('Raw data (first 500 rows)'):
                    st.dataframe(data.head(500))
        
        # Analysis explanation
        with st.expander("📚 What does this analysis show?", expanded=False):
            good_threshold = WIN_RATE_THRESHOLDS['good']
            excellent_threshold = WIN_RATE_THRESHOLDS['excellent']
            
            st.markdown("""
            <div class="explanation-box">
            <h3>📊 Understanding Monthly Seasonality & Return Patterns</h3>
            
            <p>This dashboard reveals statistical patterns in how assets behave across different months and time periods. 
            It answers questions like: <em>"Does this asset tend to perform better in certain months? Are there reliable trading windows?"</em></p>
            
            <h4>🔍 Three Core Metrics Analyzed:</h4>
            
            <div style="background: rgba(200,150,100,0.1); border-left: 4px solid #c89664; padding: 12px; margin: 10px 0; border-radius: 4px;">
            <h4 style="margin-top:0;">1. 📈 Weekly Return Progression</h4>
            <table style="width:100%; border-collapse:collapse;">
                <tr style="border-bottom: 1px solid rgba(200,150,100,0.2);">
                    <td style="padding:6px; width:20%; font-weight:bold;">Definition:</td>
                    <td>Cumulative weekly returns throughout each month</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(200,150,100,0.2);">
                    <td style="padding:6px; font-weight:bold;">Interpretation:</td>
                    <td>Shows whether gains accumulate evenly or concentrate in specific weeks</td>
                </tr>
                <tr>
                    <td style="padding:6px; font-weight:bold;">Application:</td>
                    <td>Identify optimal entry/exit windows within the month</td>
                </tr>
            </table>
            </div>
            
            <div style="background: rgba(100,150,255,0.1); border-left: 4px solid #6496ff; padding: 12px; margin: 10px 0; border-radius: 4px;">
            <h4 style="margin-top:0;">2. 🎯 Momentum (3-Day Rolling Average)</h4>
            <table style="width:100%; border-collapse:collapse;">
                <tr style="border-bottom: 1px solid rgba(100,150,255,0.2);">
                    <td style="padding:6px; width:20%; font-weight:bold;">Definition:</td>
                    <td>Smoothed daily return trends, reducing daily noise</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(100,150,255,0.2);">
                    <td style="padding:6px; font-weight:bold; vertical-align:top;">Interpretation:</td>
                    <td style="padding:6px;">
                        <div style="margin:2px 0;">📈 <strong>Uptrend</strong> = accelerating gains</div>
                        <div style="margin:2px 0;">📉 <strong>Downtrend</strong> = weakening performance</div>
                    </td>
                </tr>
                <tr>
                    <td style="padding:6px; font-weight:bold;">Application:</td>
                    <td>Spot inflection points and momentum shifts during the month</td>
                </tr>
            </table>
            </div>
            
            <div style="background: rgba(100,200,100,0.1); border-left: 4px solid #64c864; padding: 12px; margin: 10px 0; border-radius: 4px;">
            <h4 style="margin-top:0;">3. ✅ Win Rate (Positive Days %)</h4>
            <table style="width:100%; border-collapse:collapse;">
                <tr style="border-bottom: 1px solid rgba(100,200,100,0.2);">
                    <td style="padding:6px; width:20%; font-weight:bold;">Definition:</td>
                    <td>Percentage of historical years with positive returns on each calendar day</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(100,200,100,0.2);">
                    <td style="padding:6px; font-weight:bold;">Why it matters:</td>
                    <td>Shows <em>consistency</em> of positive performance, not just average magnitude</td>
                </tr>
                <tr>
                    <td style="padding:6px; font-weight:bold; vertical-align:top;">Benchmarks:</td>
                    <td style="padding:6px;">
                        <div style="margin:2px 0;">50% = <em>Random (no edge)</em></div>
                        <div style="margin:2px 0;">70%+ = <strong style="color:#64c864;">Reliable (strong signal)</strong></div>
                        <div style="margin:2px 0;">≤30% = <strong style="color:#ff6b6b;">Likely negative (avoid)</strong></div>
                    </td>
                </tr>
            </table>
            </div>
            
            <div style="background: rgba(150,150,200,0.1); border-left: 4px solid #9696c8; padding: 12px; margin: 10px 0; border-radius: 4px;">
            <h4 style="margin-top:0;">📊 The Overview Tab Contains:</h4>
            <table style="width:100%; border-collapse:collapse;">
                <tr style="border-bottom: 1px solid rgba(150,150,200,0.2);">
                    <td style="padding:6px; width:35%; font-weight:bold;">📈 Annual Overview:</td>
                    <td>Best/worst performing months and most reliable months across your date range</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(150,150,200,0.2);">
                    <td style="padding:6px; font-weight:bold;">🔬 Return Distribution:</td>
                    <td>Statistical distribution of daily returns with fitted probability models (normal, t, Laplace, logistic)</td>
                </tr>
                <tr>
                    <td style="padding:6px; font-weight:bold;">📋 Monthly Summary:</td>
                    <td>Detailed statistics table with returns, momentum, and win rates for all months</td>
                </tr>
            </table>
            </div>
            
            <div style="background: rgba(200,150,200,0.1); border-left: 4px solid #c896c8; padding: 12px; margin: 10px 0; border-radius: 4px;">
            <h4 style="margin-top:0;">🎨 Visual Color Guide:</h4>
            <table style="width:100%; border-collapse:collapse;">
                <tr style="border-bottom: 1px solid rgba(200,150,200,0.2);">
                    <td style="padding:6px; width:20%; font-weight:bold;">🟢 Green</td>
                    <td>Excellent performance (win rate > {excellent}%)</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(200,150,200,0.2);">
                    <td style="padding:6px; font-weight:bold;">🔵 Blue</td>
                    <td>Good performance ({good}–{excellent}% win rate)</td>
                </tr>
                <tr>
                    <td style="padding:6px; font-weight:bold;">🟠 Orange/Red</td>
                    <td>Weak or negative performance (< {good}% win rate)</td>
                </tr>
            </table>
            </div>
            
            <div style="background: rgba(255,200,100,0.1); border-left: 4px solid #ffc864; padding: 12px; margin: 10px 0; border-radius: 4px;">
            <h4 style="margin-top:0;">💡 How to Use These Insights:</h4>
            <table style="width:100%; border-collapse:collapse;">
                <tr style="border-bottom: 1px solid rgba(255,200,100,0.2);">
                    <td style="padding:6px; font-weight:bold; vertical-align:top;">📌 Identify seasonal opportunities:</td>
                    <td>Find months with consistently high win rates</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,200,100,0.2);">
                    <td style="padding:6px; font-weight:bold; vertical-align:top;">⏱️ Time your trades:</td>
                    <td>Use weekly progression charts to enter/exit within optimal windows</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,200,100,0.2);">
                    <td style="padding:6px; font-weight:bold; vertical-align:top;">⚠️ Assess risk:</td>
                    <td>Check distribution shape for tail risks and volatility patterns</td>
                </tr>
                <tr>
                    <td style="padding:6px; font-weight:bold; vertical-align:top;">✅ Validate strategy:</td>
                    <td>Backtesting historical patterns increases confidence in tactical decisions</td>
                </tr>
            </table>
            </div>
            </div>
            """.format(good=good_threshold, excellent=excellent_threshold), unsafe_allow_html=True)
        
        # Get cached analysis for all months
        all_analysis = analyze_all_months(data)
        
        # Create tabs
        tab_names = ["Overview"] + [calendar.month_name[i] for i in TRADING_MONTHS]
        tabs = st.tabs(tab_names)
        
        # ====================================================================
        # Overview Tab
        # ====================================================================
        with tabs[0]:
            render_overview_tab(all_analysis, show_summary_stats, ticker, data)
        
        # ====================================================================
        # Individual Month Tabs
        # ====================================================================
        for month_idx, month_num in enumerate(TRADING_MONTHS, start=1):
            with tabs[month_idx]:
                render_month_tab(month_num, all_analysis[month_num], show_trend_lines, show_summary_stats)
    
    except Exception as e:
        logger.error(f"Single ticker analysis error: {e}", exc_info=True)
        st.error(f"Error in single ticker analysis: {str(e)}")


def render_multi_ticker_analysis():
    """Render multi-ticker correlation and performance analysis."""
    try:
        # Fetch data for all tickers
        with st.spinner(f"Downloading data for {len(tickers_list)} tickers..."):
            multi_data = fetch_multiple_and_validate(tickers_list, start_date_str, end_date_str)
        
        if multi_data is None or multi_data.empty:
            st.error("No data available for the selected tickers.")
            st.stop()
        
        # Show success message
        st.markdown(
            f'<div class="success-message">✅ Data loaded for {len(tickers_list)} tickers: {", ".join(tickers_list)}</div>',
            unsafe_allow_html=True
        )
        
        # Data info metrics
        create_metric_cards_row([
            {
                'value': f"{start_date_str} to {end_date_str}",
                'label': 'Date Range'
            },
            {
                'value': f"{len(multi_data):,}",
                'label': 'Common Trading Days'
            },
            {
                'value': str(len(tickers_list)),
                'label': 'Tickers'
            },
        ], num_columns=3)
        
        # Information Expander
        with st.expander("📚 What does Multi-Ticker Comparison show?", expanded=False):
            st.markdown("""
            <div style="background: rgba(100,200,150,0.1); border-left: 4px solid #64c896; padding: 12px; margin: 10px 0; border-radius: 4px;">
            <h4 style="margin-top:0;">🔗 Multi-Ticker Analysis Overview</h4>
            
            <p>Compare performance, volatility, and relationships across multiple assets simultaneously. 
            This mode reveals how assets move together and which are most resilient during market shifts.</p>
            
            <table style="width:100%; border-collapse:collapse;">
                <tr style="border-bottom: 1px solid rgba(100,200,150,0.2);">
                    <td style="padding:6px; width:20%; font-weight:bold;">📊 Correlation Matrix:</td>
                    <td>Shows how tightly different assets move together (-1 to +1, where 1 = perfectly correlated)</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(100,200,150,0.2);">
                    <td style="padding:6px; font-weight:bold;">📈 Relative Performance:</td>
                    <td>Compares cumulative returns across all selected tickers with synchronized baselines</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(100,200,150,0.2);">
                    <td style="padding:6px; font-weight:bold;">📉 Volatility Comparison:</td>
                    <td>Standard deviation of returns for each ticker (lower = more stable, higher = more risky)</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(100,200,150,0.2);">
                    <td style="padding:6px; font-weight:bold; vertical-align:top;">🎯 Performance Statistics:</td>
                    <td style="padding:6px;">
                        <div style="margin:2px 0;"><strong>Mean Return:</strong> Average daily/periodic return over the period (higher is better, compare with risk)</div>
                        <div style="margin:2px 0;"><strong>Max Drawdown:</strong> Largest peak-to-trough decline; shows worst-case loss (e.g., -25% = fell 25% from peak)</div>
                        <div style="margin:2px 0;"><strong>Sharpe Ratio:</strong> Return per unit of risk (>1.0 = excellent, 0.5–1.0 = good, <0.5 = weak, <0 = worse than risk-free)</div>
                    </td>
                </tr>
            </table>
            </div>
            
            <div style="background: rgba(200,150,255,0.1); border-left: 4px solid #c896ff; padding: 12px; margin: 10px 0; border-radius: 4px;">
            <h4 style="margin-top:0;">💡 How to Interpret Results:</h4>
            
            <table style="width:100%; border-collapse:collapse;">
                <tr style="border-bottom: 1px solid rgba(200,150,255,0.2);">
                    <td style="padding:6px; font-weight:bold; vertical-align:top;">🔴 High Correlation (>0.7):</td>
                    <td style="padding:6px;">
                        <div style="margin:2px 0;">Assets move in sync (good for diversification score)</div>
                        <div style="margin:2px 0; font-size:12px; color:#999;">Means less portfolio diversification benefit</div>
                    </td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(200,150,255,0.2);">
                    <td style="padding:6px; font-weight:bold; vertical-align:top;">🟡 Low Correlation (<0.3):</td>
                    <td style="padding:6px;">
                        <div style="margin:2px 0;">Assets move independently (excellent for diversification)</div>
                        <div style="margin:2px 0; font-size:12px; color:#999;">Combined portfolio is more stable</div>
                    </td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(200,150,255,0.2);">
                    <td style="padding:6px; font-weight:bold; vertical-align:top;">🔵 Negative Correlation (<-0.3):</td>
                    <td style="padding:6px;">
                        <div style="margin:2px 0;">Assets move opposite directions (perfect hedges)</div>
                        <div style="margin:2px 0; font-size:12px; color:#999;">When one falls, the other typically rises</div>
                    </td>
                </tr>
            </table>
            </div>
            
            <div style="background: rgba(200,150,255,0.1); border-left: 4px solid #c896ff; padding: 12px; margin: 10px 0; border-radius: 4px;">
            <h4 style="margin-top:0;">✅ Best Practices:</h4>
            
            <table style="width:100%; border-collapse:collapse;">
                <tr style="border-bottom: 1px solid rgba(255,200,100,0.2);">
                    <td style="padding:6px; font-weight:bold; vertical-align:top;">🎯 Select Diverse Assets:</td>
                    <td>Mix stocks, bonds, commodities, or crypto for meaningful insights</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,200,100,0.2);">
                    <td style="padding:6px; font-weight:bold; vertical-align:top;">📅 Use Consistent Timeframes:</td>
                    <td>Longer periods (2+ years) reveal more reliable correlation patterns</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,200,100,0.2);">
                    <td style="padding:6px; font-weight:bold; vertical-align:top;">⚠️ Monitor Market Changes:</td>
                    <td>Correlations shift during crises; re-check regularly</td>
                </tr>
                <tr>
                    <td style="padding:6px; font-weight:bold; vertical-align:top;">🔄 Rebalance Periodically:</td>
                    <td>Use relative performance charts to identify rebalancing opportunities</td>
                </tr>
            </table>
            </div>
            """, unsafe_allow_html=True)
        
        tab_names = ["Correlation", "Performance", "Metrics"]
        tabs = st.tabs(tab_names)
        
        # ====================================================================
        # Correlation Tab
        # ====================================================================
        with tabs[0]:
            st.markdown("## 📊 Correlation Analysis")
            st.markdown("Daily return correlations across selected tickers")
            
            corr_matrix = compute_correlation_matrix(multi_data)
            if corr_matrix is not None and not corr_matrix.empty:
                fig_corr = create_correlation_heatmap(corr_matrix)
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # Correlation insights
                st.markdown("### 🔍 Key Insights")
                max_corr_ticker = []
                min_corr_ticker = []
                for i, ticker1 in enumerate(corr_matrix.columns):
                    for j, ticker2 in enumerate(corr_matrix.columns):
                        if i < j:
                            max_corr_ticker.append((ticker1, ticker2, corr_matrix.iloc[i, j]))
                            min_corr_ticker.append((ticker1, ticker2, corr_matrix.iloc[i, j]))
                
                if max_corr_ticker:
                    max_corr_ticker.sort(key=lambda x: x[2], reverse=True)
                    min_corr_ticker.sort(key=lambda x: x[2])
                    
                    create_metric_cards_row([
                        {
                            'value': f"{max_corr_ticker[0][0]} ↔ {max_corr_ticker[0][1]}",
                            'label': 'Highest Correlation',
                            'subtext': f"{max_corr_ticker[0][2]:.3f}",
                            'color': COLOR_POSITIVE
                        },
                        {
                            'value': f"{min_corr_ticker[0][0]} ↔ {min_corr_ticker[0][1]}",
                            'label': 'Lowest Correlation',
                            'subtext': f"{min_corr_ticker[0][2]:.3f}",
                            'color': COLOR_NEGATIVE
                        }
                    ], num_columns=2)
            else:
                st.warning("Unable to compute correlation matrix.")
        
        # ====================================================================
        # Performance Tab
        # ====================================================================
        with tabs[1]:
            st.markdown("## 📈 Relative Performance")
            st.markdown("Cumulative returns comparison over time")
            
            cumulative_returns = calculate_relative_performance(multi_data)
            if cumulative_returns is not None and not cumulative_returns.empty:
                fig_perf = create_relative_performance_chart(cumulative_returns)
                st.plotly_chart(fig_perf, use_container_width=True)
            else:
                st.warning("Unable to compute relative performance.")
        
        # ====================================================================
        # Metrics Tab
        # ====================================================================
        with tabs[2]:
            st.markdown("## 📊 Performance Metrics")
            st.markdown("Annualized return, volatility, Sharpe ratio, and max drawdown")
            
            stats = calculate_statistics(multi_data)
            if stats:
                # Display as cards
                cards = []
                for ticker, metric_dict in sorted(stats.items()):
                    cards.append({
                        'value': f"{metric_dict['annualized_return']:.2f}%",
                        'label': ticker,
                        'subtext': f"Vol: {metric_dict['volatility']:.1f}% | Sharpe: {metric_dict['sharpe']:.2f}",
                        'color': COLOR_POSITIVE if metric_dict['annualized_return'] > 0 else COLOR_NEGATIVE
                    })
                
                # Display in rows of 3
                for i in range(0, len(cards), 3):
                    create_metric_cards_row(cards[i:i+3], num_columns=min(3, len(cards[i:i+3])))
                
                # Detailed table
                st.markdown("### 📋 Detailed Statistics")
                stats_df = pd.DataFrame(stats).T
                stats_df = stats_df.round(2)
                st.dataframe(stats_df, use_container_width=True)
            else:
                st.warning("Unable to compute performance metrics.")
    
    except Exception as e:
        logger.error(f"Multi-ticker analysis error: {e}", exc_info=True)
        st.error(f"Error in multi-ticker analysis: {str(e)}")


def render_overview_tab(all_analysis: dict, show_stats: bool, ticker: str, data: pd.DataFrame) -> None:
    """Render the overview tab with sub-tabs."""
    # Create sub-tabs within the Overview
    sub_tab1, sub_tab2, sub_tab3 = st.tabs(["📊 Annual Overview", "🔬 Return Distribution", "📋 Monthly Summary"])
    
    # Build summary once for reuse
    summary_df = build_overview_summary(all_analysis, data)
    
    if summary_df is None or summary_df.empty:
        st.warning("No data available for overview.")
        return
    
    # ====================================================================
    # Sub-tab 1: Annual Overview
    # ====================================================================
    with sub_tab1:
        st.markdown("Get a comprehensive view of all months' performance at a glance")
        
        # Overview charts
        fig_returns = px.bar(
            summary_df, x='Month', y='Monthly Return',
            title='💹 Average Monthly Returns (%)',
            color='Monthly Return',
            color_continuous_scale='RdYlGn',
            text='Monthly Return'
        )
        fig_returns.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        _apply_chart_styling(fig_returns)
        st.plotly_chart(fig_returns, use_container_width=True)
        
        fig_win_rates = px.bar(
            summary_df, x='Month', y='Win Rate',
            title='🎯 Average Win Rates (%)',
            color='Win Rate',
            color_continuous_scale='RdYlGn',
            text='Win Rate'
        )
        fig_win_rates.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_win_rates.add_hline(y=50, line_dash="dash", line_color=COLOR_TEXT_SECONDARY, annotation_text="Break-even")
        _apply_chart_styling(fig_win_rates)
        st.plotly_chart(fig_win_rates, use_container_width=True)
        
        # Key insights
        st.markdown("### 🔍 Key Insights")
        best_month = summary_df.loc[summary_df['Monthly Return'].idxmax()]
        worst_month = summary_df.loc[summary_df['Monthly Return'].idxmin()]
        best_win_rate = summary_df.loc[summary_df['Win Rate'].idxmax()]
        
        create_metric_cards_row([
            {
                'value': f"🚀 {best_month['Month']}",
                'label': 'Best Performing Month',
                'subtext': f"{best_month['Monthly Return']:.2f}% avg return",
                'color': COLOR_POSITIVE
            },
            {
                'value': f"⚠️ {worst_month['Month']}",
                'label': 'Weakest Month',
                'subtext': f"{worst_month['Monthly Return']:.2f}% avg return",
                'color': COLOR_NEGATIVE
            },
            {
                'value': f"🎯 {best_win_rate['Month']}",
                'label': 'Most Reliable Month',
                'subtext': f"{best_win_rate['Win Rate']:.1f}% win rate",
                'color': '#4ECDC4'
            },
        ], num_columns=3)
    
    # ====================================================================
    # Sub-tab 2: Return Distribution
    # ====================================================================
    with sub_tab2:
        try:
            returns, dist_stats = compute_return_distribution(data, period='daily')
            dist_info = fit_distribution_to_returns(returns)
            ticker_name = get_ticker_display_name(ticker)
            fig_dist = create_return_distribution_chart(returns, title=f"{ticker_name} - Daily Return Distribution", dist_info=dist_info)
            st.plotly_chart(fig_dist, use_container_width=True)

            # Show fitted distribution name and AIC/BIC values
            fitted_name = dist_info.get('best_dist', 'normal').capitalize()
            aic = dist_info.get('aic')
            bic = dist_info.get('bic')
            
            # Format AIC/BIC stats, handling None values
            aic_str = f"{aic:.2f}" if aic is not None else "N/A"
            bic_str = f"{bic:.2f}" if bic is not None else "N/A"
            st.caption(f"📊 Best-fit distribution: **{fitted_name}** (AIC: {aic_str}, BIC: {bic_str})")

            # Calculate statistics from the fitted distribution
            fitted_stats = get_distribution_stats(dist_info)
            goodness_fit = get_goodness_of_fit(returns, dist_info)
            
            # Show raw data statistics
            st.markdown("**Raw Data Statistics:**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mean", f"{dist_stats['mean']*100:.3f}%")
            c2.metric("Std (σ)", f"{dist_stats['std']*100:.3f}%")
            c3.metric("Skew", f"{dist_stats['skew']:.3f}")
            c4.metric("Kurtosis", f"{dist_stats['kurtosis']:.3f}")
            
            # Show normality test (Jarque-Bera)
            if dist_stats.get('jb') is not None:
                st.caption(f"Jarque-Bera Test (normality): {dist_stats['jb']:.2f}, p-value: {dist_stats.get('jb_pvalue', 'N/A')}")
            
            # Show fitted distribution statistics
            st.markdown("**Fitted Distribution Statistics:**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mean (fitted)", f"{fitted_stats['mean']*100:.3f}%")
            c2.metric("Std (fitted)", f"{fitted_stats['std']*100:.3f}%")
            c3.metric("Skew (fitted)", f"{fitted_stats['skew']:.3f}")
            c4.metric("Kurtosis (fitted)", f"{fitted_stats['kurtosis']:.3f}")
            
            # Show goodness-of-fit test (Kolmogorov–Smirnov for fitted distribution)
            ks_stat = goodness_fit.get('ks_stat')
            ks_pvalue = goodness_fit.get('ks_pvalue')
            if ks_stat is not None and ks_pvalue is not None:
                st.caption(f"Kolmogorov–Smirnov Goodness-of-Fit: stat={ks_stat:.3f}, p-value={ks_pvalue:.3f}")
            else:
                st.caption("Kolmogorov–Smirnov Goodness-of-Fit: N/A")
        except Exception as e:
            st.warning(f"Unable to compute distribution: {e}")
    
    # ====================================================================
    # Sub-tab 3: Monthly Summary
    # ====================================================================
    with sub_tab3:
        if show_stats:
            st.markdown("Monthly performance statistics across all analyzed years")
            display_df = summary_df.copy()
            display_df['Monthly Return'] = display_df['Monthly Return'].apply(lambda x: f"{x:.2f}%")
            display_df['Avg Momentum'] = display_df['Avg Momentum'].apply(lambda x: f"{x:.3f}%")
            display_df['Win Rate'] = display_df['Win Rate'].apply(lambda x: f"{x:.1f}%")
            
            st.dataframe(display_df, hide_index=True, use_container_width=True)
        else:
            st.info("Enable 'Show summary statistics' in the sidebar to view this data.")


def render_month_tab(month_num: int, analysis_data: tuple, show_trends: bool, show_stats: bool) -> None:
    """Render individual month tab."""
    weekly_returns, momentum, win_rates, years = analysis_data
    
    if not years:
        create_insight_box(
            f"⚠️ Insufficient data for {calendar.month_name[month_num]}",
            "Try selecting a longer date range or different asset",
            color=COLOR_NEGATIVE
        )
        return
    
    # Month header
    st.markdown(f"## 📅 {calendar.month_name[month_num]} Analysis")
    st.markdown(f"*Based on {len(years)} years of data ({min(years)}-{max(years)})*")
    
    # Create chart
    fig = create_monthly_chart(month_num, weekly_returns, momentum, win_rates, show_trends)
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics
    if show_stats:
        st.markdown("### 📈 Monthly Performance Summary")
        _render_month_summary(month_num, weekly_returns, momentum, win_rates)
        
        st.markdown("### 💡 Key Insights")
        _render_month_insights(win_rates)


def _render_month_summary(month_num: int, weekly_returns: dict, momentum: dict, win_rates: dict) -> None:
    """Render monthly performance summary cards."""
    cards = []
    
    # Weekly returns card
    if weekly_returns:
        total_return = max(weekly_returns.values())
        best_week = max(weekly_returns, key=weekly_returns.get)
        cards.append({
            'value': f"{total_return:.2f}%",
            'label': 'Average Monthly Return',
            'color': get_return_color(total_return),
            'subtext': f"🏆 Best Week: Week {best_week}"
        })
    
    # Momentum card
    if momentum:
        avg_momentum = np.mean(list(momentum.values()))
        days = sorted(momentum.keys())
        mom_values = [momentum[day] for day in days]
        
        if len(days) > 5:
            trend_corr = np.corrcoef(days, mom_values)[0, 1]
            if trend_corr > 0.1:
                trend, trend_color = "📈 Accelerating", COLOR_POSITIVE
            elif trend_corr < -0.1:
                trend, trend_color = "📉 Decelerating", COLOR_NEGATIVE
            else:
                trend, trend_color = "📊 Stable", '#4ECDC4'
        else:
            trend, trend_color = "📊 Stable", '#4ECDC4'
        
        cards.append({
            'value': f"{avg_momentum:.3f}%",
            'label': 'Average Daily Momentum',
            'color': get_return_color(avg_momentum),
            'subtext': trend
        })
    
    # Win rate card
    if win_rates:
        avg_win_rate = np.mean(list(win_rates.values()))
        status, status_color = get_win_rate_status(avg_win_rate)
        
        cards.append({
            'value': f"{avg_win_rate:.1f}%",
            'label': 'Average Win Rate',
            'color': get_win_rate_color(avg_win_rate),
            'subtext': status
        })
    
    create_metric_cards_row(cards, num_columns=min(3, len(cards)))


def _render_month_insights(win_rates: dict) -> None:
    """Render first half vs second half analysis."""
    if not win_rates:
        return
    
    days = sorted(win_rates.keys())
    mid_day = int(max(days) * MONTH_SPLIT_RATIO)
    
    first_half_days = [d for d in days if d <= mid_day]
    second_half_days = [d for d in days if d > mid_day]
    
    first_half_win = np.mean([win_rates[d] for d in first_half_days]) if first_half_days else 0
    second_half_win = np.mean([win_rates[d] for d in second_half_days]) if second_half_days else 0
    
    col1, col2 = st.columns(2)
    
    with col1:
        first_color = COLOR_POSITIVE if first_half_win > 50 else COLOR_NEGATIVE
        create_insight_box(
            "First Half Performance",
            f"Days 1-{mid_day}: {first_half_win:.1f}% win rate",
            color=first_color
        )
    
    with col2:
        second_color = COLOR_POSITIVE if second_half_win > 50 else COLOR_NEGATIVE
        create_insight_box(
            "Second Half Performance",
            f"Days {mid_day+1}-{max(days)}: {second_half_win:.1f}% win rate",
            color=second_color
        )


def _apply_chart_styling(fig) -> None:
    """Apply consistent chart styling."""
    fig.update_layout(
        height=CHART_HEIGHT,
        showlegend=False,
        title_font_size=FONT_SIZE_TITLE,
        title_x=0.5,
        title_font_color='#FFFFFF',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#FFFFFF')
    )
    fig.update_xaxes(
        tickangle=45,
        gridcolor='rgba(255,255,255,0.1)',
        tickcolor=COLOR_TEXT_SECONDARY,
        linecolor='rgba(255,255,255,0.2)',
        tickfont=dict(color=COLOR_TEXT_SECONDARY)
    )
    fig.update_yaxes(
        gridcolor='rgba(255,255,255,0.1)',
        tickcolor=COLOR_TEXT_SECONDARY,
        linecolor='rgba(255,255,255,0.2)',
        tickfont=dict(color=COLOR_TEXT_SECONDARY)
    )
    fig.update_traces(textfont=dict(color='#FFFFFF'))


if __name__ == "__main__":
    main()
