"""Sidebar controls and configuration rendering."""

import logging
from datetime import date
from typing import Tuple, List, Union
import streamlit as st
import yfinance as yf
from helpers.config import (
    POPULAR_ASSETS, PRESET_DATE_RANGES, DEFAULT_DATE_RANGE,
    DEFAULT_TICKER, DEFAULT_SHOW_TRENDS, DEFAULT_SHOW_STATS,
    DEFAULT_ASSET_METHOD, TICKER_DATABASE, DEFAULT_MULTI_TICKERS
)

logger = logging.getLogger(__name__)


@st.cache_data(ttl=600)
def validate_ticker_with_yfinance(ticker: str) -> bool:
    """Validate if ticker exists in yfinance using cached result.
    
    Args:
        ticker: Ticker symbol to validate
        
    Returns:
        True if ticker is valid, False otherwise
    """
    try:
        ticker_obj = yf.Ticker(ticker)
        # Try to fetch info - if successful, ticker is valid
        info = ticker_obj.info
        return bool(info and 'regularMarketPrice' in info or 'bid' in info)
    except Exception as e:
        logger.debug(f"Validation failed for {ticker}: {e}")
        return False


def render_sidebar() -> Tuple[Union[str, List[str]], str, str, bool, bool, str]:
    """Render sidebar controls and return user selections.
    
    Returns:
        Tuple of (ticker(s), start_date_str, end_date_str, show_trends, show_stats, analysis_mode)
        - ticker(s): str for single ticker, List[str] for multiple
        - analysis_mode: "single" or "multi"
    """
    with st.sidebar:
        st.header("🎛️ Analysis Settings")
        
        # ====================================================================
        # Analysis Mode (Single vs Multi-Ticker)
        # ====================================================================
        st.subheader("🔧 Analysis Mode")
        analysis_mode = st.radio(
            "Select analysis type:",
            ["Single Ticker", "Multi-Ticker Comparison"],
            index=0
        )
        
        # ====================================================================
        # Asset Selection
        # ====================================================================
        st.subheader("📊 Asset Selection")
        
        if analysis_mode == "Single Ticker":
            asset_method = st.radio(
                "Choose selection method:",
                ["Popular Assets", "Custom Ticker"],
                index=0 if DEFAULT_ASSET_METHOD == "Popular Assets" else 1
            )
            
            try:
                if asset_method == "Popular Assets":
                    selected_asset_name = st.selectbox(
                        "Select an asset:",
                        list(POPULAR_ASSETS.keys()),
                        index=0
                    )
                    tickers = POPULAR_ASSETS[selected_asset_name]
                else:
                    ticker_input = st.text_input(
                        "Enter ticker symbol:",
                        value=DEFAULT_TICKER,
                        help="Examples: AAPL, MSFT, ^GSPC, BTC-USD"
                    ).upper().strip()
                    
                    if not ticker_input:
                        ticker_input = DEFAULT_TICKER
                    
                    tickers = ticker_input
            except Exception as e:
                logger.error(f"Error in asset selection: {e}")
                st.error("Error in asset selection. Using default.")
                tickers = DEFAULT_TICKER
        
        else:  # Multi-Ticker
            st.markdown("**🔍 Search & Select Tickers**")
            
            # Tabs: Database search vs Custom input
            search_tab, custom_tab = st.tabs(["📚 Database", "🆓 Custom"])
            
            selected_tickers = []
            
            # ================================================================
            # Database Search Tab
            # ================================================================
            with search_tab:
                # Search box at top
                search_query = st.text_input(
                    "🔎 Search tickers:",
                    placeholder="Type to filter (e.g., 'Apple', 'S&P', 'BTC')",
                    help="Filter all available tickers by name or symbol",
                    key="db_search"
                ).lower()
                
                # Filter tickers based on search
                filtered_db = {
                    name: ticker for name, ticker in TICKER_DATABASE.items()
                    if search_query in name.lower() or search_query in ticker.lower()
                }
                
                # Show count
                st.caption(f"📊 Showing {len(filtered_db)} of {len(TICKER_DATABASE)} tickers")
                
                # Initialize session state for tracking selections
                if 'db_selected' not in st.session_state:
                    st.session_state.db_selected = {}
                
                # Create expander to hold paginated checkboxes (robust across Streamlit versions)
                with st.expander("📋 Select Tickers", expanded=True):
                    # Small styling for nav/buttons
                    st.markdown(
                        """
                    <style>
                    /* Only style small nav metadata; avoid global button rules to prevent layout issues */
                    .db-nav { text-align:center; font-size:12px; color:#444; }
                    .db-meta { font-size:12px; color:#666; }
                    </style>
                    """,
                        unsafe_allow_html=True,
                    )

                    # Pagination settings (toggle to show all)
                    items_per_page = 5

                    entries = sorted(filtered_db.items())
                    total = len(entries)

                    show_all = st.checkbox("Show all tickers", value=False, key="db_show_all")

                    if 'db_page' not in st.session_state:
                        st.session_state.db_page = 0

                    # If showing all, override paging
                    if show_all:
                        start_idx = 0
                        end_idx = total
                        total_pages = 1
                        page = 0
                    else:
                        import math
                        total_pages = max(1, math.ceil(total / items_per_page))
                        page = max(0, min(st.session_state.db_page, total_pages - 1))
                        start_idx = page * items_per_page
                        end_idx = min(start_idx + items_per_page, end_idx if 'end_idx' in locals() else start_idx + items_per_page)

                    # Navigation controls
                    if total == 0:
                        st.info("No tickers match your search.")
                    else:
                        nav_col1, nav_col2, nav_col3 = st.columns([1, 6, 1])
                        with nav_col1:
                            if not show_all and st.button("◀", key="db_prev", help="Previous page"):
                                if page > 0:
                                    st.session_state.db_page = page - 1
                                    st.rerun()
                        with nav_col2:
                            st.markdown(f"<div class='db-meta'>Showing {start_idx+1}–{end_idx} of {total} — Page {page+1}/{total_pages}</div>", unsafe_allow_html=True)
                        with nav_col3:
                            if not show_all and st.button("▶", key="db_next", help="Next page"):
                                if page < total_pages - 1:
                                    st.session_state.db_page = page + 1
                                    st.rerun()

                        # Display the slice of items for the current page
                        selected_tickers = []
                        for name, ticker in entries[start_idx:end_idx]:
                            key = f"db_checkbox_{ticker}"

                            is_checked = st.checkbox(
                                f"{name} ({ticker})",
                                value=st.session_state.db_selected.get(key, False),
                                key=key,
                            )

                            st.session_state.db_selected[key] = is_checked

                            if is_checked:
                                selected_tickers.append(ticker)
                
                if selected_tickers:
                    st.success(f"✅ Selected {len(selected_tickers)} ticker(s): {', '.join(selected_tickers)}")
                else:
                    st.info("👈 Check boxes to select tickers")
            
            with custom_tab:
                st.markdown("**Enter custom tickers** (validated with yfinance)")
                custom_input = st.text_area(
                    "Paste ticker symbols:",
                    placeholder="AAPL\nMSFT\nGOOGL\nTSLA",
                    help="One ticker per line or comma-separated",
                    height=100
                )
                
                if custom_input:
                    raw_tickers = []
                    for line in custom_input.split('\n'):
                        raw_tickers.extend([t.strip().upper() for t in line.split(',') if t.strip()])
                    
                    st.markdown("**Validation Status:**")
                    valid_custom = []
                    for ticker in raw_tickers:
                        if ticker:
                            with st.spinner(f"Checking {ticker}..."):
                                is_valid = validate_ticker_with_yfinance(ticker)
                            
                            if is_valid:
                                st.success(f"✅ {ticker} - Valid")
                                valid_custom.append(ticker)
                            else:
                                st.warning(f"⚠️ {ticker} - Not found or invalid")
                    
                    selected_tickers.extend(valid_custom)
            
            if selected_tickers:
                tickers = list(dict.fromkeys(selected_tickers))
            else:
                st.info("No tickers selected. Using defaults.")
                tickers = DEFAULT_MULTI_TICKERS
        
        # ====================================================================
        # Date Range Selection
        # ====================================================================
        st.subheader("📅 Date Range")
        
        date_range_option = st.selectbox(
            "Select date range:",
            list(PRESET_DATE_RANGES.keys()),
            index=list(PRESET_DATE_RANGES.keys()).index(DEFAULT_DATE_RANGE)
        )
        
        try:
            if date_range_option == "Custom Range":
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input(
                        "Start Date:",
                        value=date(2020, 1, 1),
                        min_value=date(1950, 1, 1),
                        max_value=date.today()
                    )
                with col2:
                    end_date = st.date_input(
                        "End Date:",
                        value=date.today(),
                        min_value=date(1950, 1, 1),
                        max_value=date.today()
                    )
                
                if start_date >= end_date:
                    st.error("Start date must be before end date!")
                    start_date_str = "2020-01-01"
                    end_date_str = "2024-12-31"
                else:
                    start_date_str = start_date.strftime("%Y-%m-%d")
                    end_date_str = end_date.strftime("%Y-%m-%d")
            else:
                start_date_str, end_date_str = PRESET_DATE_RANGES[date_range_option]
        except Exception as e:
            logger.error(f"Error in date range selection: {e}")
            st.error("Error in date selection. Using default range.")
            start_date_str, end_date_str = PRESET_DATE_RANGES[DEFAULT_DATE_RANGE]
        
        # ====================================================================
        # Analysis Options
        # ====================================================================
        st.subheader("⚙️ Analysis Options")
        show_trend_lines = st.checkbox("Show trend lines", value=DEFAULT_SHOW_TRENDS)
        show_summary_stats = st.checkbox("Show summary statistics", value=DEFAULT_SHOW_STATS)
    
    return tickers, start_date_str, end_date_str, show_trend_lines, show_summary_stats, analysis_mode
