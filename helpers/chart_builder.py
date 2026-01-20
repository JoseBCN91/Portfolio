import calendar
import logging
from typing import Dict, Optional
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
logger = logging.getLogger(__name__)
from helpers.chart_style import hex_to_rgba, update_common_axes_styling


def get_ticker_display_name(ticker: str) -> str:
    """Convert ticker symbol to human-readable name.
    
    Args:
        ticker: Ticker symbol (e.g., '^GSPC', 'AAPL')
        
    Returns:
        Display name (e.g., 'S&P 500', 'Apple')
    """
    # Reverse mapping of common tickers to names
    ticker_names = {
        '^GSPC': 'S&P 500',
        '^IXIC': 'NASDAQ Composite',
        '^DJI': 'Dow Jones Industrial',
        '^RUT': 'Russell 2000',
        '^FTSE': 'FTSE 100',
        '^GDAXI': 'DAX',
        '^FCHI': 'CAC 40',
        '^N225': 'Nikkei 225',
        '^HSI': 'Hang Seng',
        '000001.SS': 'Shanghai Composite',
        'AAPL': 'Apple',
        'MSFT': 'Microsoft',
        'AMZN': 'Amazon',
        'GOOGL': 'Google/Alphabet',
        'META': 'Meta',
        'TSLA': 'Tesla',
        'NVDA': 'NVIDIA',
        'INTC': 'Intel',
        'BRK.B': 'Berkshire Hathaway B',
        'JNJ': 'Johnson & Johnson',
        'V': 'Visa',
        'WMT': 'Walmart',
        'JPM': 'JPMorgan Chase',
        'VZ': 'Verizon',
        'KO': 'Coca-Cola',
        'BTC-USD': 'Bitcoin',
        'ETH-USD': 'Ethereum',
        'GC=F': 'Gold Futures',
        'CL=F': 'Crude Oil Futures',
        'ES=F': 'S&P 500 E-mini Futures',
    }
    return ticker_names.get(ticker, ticker)

# Color palette - centralized and easy to maintain
COLOR_PALETTE = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'positive': '#06D6A0',
    'negative': '#F18F01',
    'neutral': '#B8BCC8',
    'grid': 'rgba(255,255,255,0.1)',
    'border': 'rgba(255,255,255,0.2)',
    'white': '#FFFFFF',
    'text_secondary': '#B8BCC8',
}

# Chart styling constants
SUBPLOT_TITLES = ('Weekly Progression', 'Momentum Patterns', 'Success Rates')
SUBPLOT_COLS = 3
CHART_HEIGHT = 450
FONT_SIZE_TITLE = 18
FONT_SIZE_LABELS = 12
FONT_SIZE_ANNOTATION = 14
WIN_RATE_THRESHOLD_HIGH = 55
WIN_RATE_THRESHOLD_MID = 50
MARKER_SIZE = 8
LINE_WIDTH = 3
MIN_DATA_POINTS_FOR_TREND = 5


# moved to helpers.chart_style: hex_to_rgba


def _validate_input_data(weekly_returns: Dict, momentum: Dict, win_rates: Dict) -> bool:
    """Validate that at least some data is provided.
    
    Args:
        weekly_returns: Weekly returns data
        momentum: Momentum data
        win_rates: Win rate data
        
    Returns:
        True if valid data exists, False otherwise
    """
    has_data = bool(weekly_returns or momentum or win_rates)
    if not has_data:
        logger.warning("No data provided to chart builder")
    return has_data



def create_monthly_chart(
    month_num: int,
    weekly_returns: Optional[Dict] = None,
    momentum: Optional[Dict] = None,
    win_rates: Optional[Dict] = None,
    show_trends: bool = True
) -> go.Figure:
    """Create interactive chart for a specific month.
    
    Args:
        month_num: Month number (1-12)
        weekly_returns: Dict of trading week -> return percentage
        momentum: Dict of day -> momentum percentage
        win_rates: Dict of day -> win rate percentage
        show_trends: Whether to show trend lines on momentum
        
    Returns:
        Plotly figure object
        
    Raises:
        ValueError: If invalid month_num
    """
    if not isinstance(month_num, int) or month_num < 1 or month_num > 12:
        raise ValueError(f"Invalid month_num: {month_num}")
    
    month_name = calendar.month_name[month_num]
    
    # Provide defaults for None inputs
    weekly_returns = weekly_returns or {}
    momentum = momentum or {}
    win_rates = win_rates or {}
    
    if not _validate_input_data(weekly_returns, momentum, win_rates):
        logger.warning(f"Creating empty chart for {month_name}")

    # Create subplots
    fig = make_subplots(
        rows=1, cols=SUBPLOT_COLS,
        subplot_titles=SUBPLOT_TITLES,
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
    )

    # 1. Weekly Progression
    if weekly_returns:
        weeks = sorted(weekly_returns.keys())
        week_values = [weekly_returns[w] for w in weeks]

        plot_weeks = [0] + weeks
        plot_values = [0] + week_values

        fill_color = hex_to_rgba(COLOR_PALETTE['primary'], 0.2)
        
        fig.add_trace(
            go.Scatter(
                x=plot_weeks,
                y=plot_values,
                mode='lines+markers',
                name='Weekly Returns',
                line=dict(color=COLOR_PALETTE['primary'], width=LINE_WIDTH),
                marker=dict(
                    size=MARKER_SIZE,
                    color='white',
                    line=dict(color=COLOR_PALETTE['primary'], width=2)
                ),
                fill='tonexty',
                fillcolor=fill_color
            ),
            row=1, col=1
        )

        fig.add_hline(
            y=0, line_dash="dash", line_color=COLOR_PALETTE['neutral'],
            opacity=0.7, row=1, col=1
        )

    # 2. Momentum Patterns
    if momentum:
        days = sorted(momentum.keys())
        mom_values = [momentum[day] for day in days]

        fig.add_trace(
            go.Scatter(
                x=days,
                y=mom_values,
                mode='lines',
                name='Momentum',
                line=dict(color=COLOR_PALETTE['secondary'], width=LINE_WIDTH),
            ),
            row=1, col=2
        )

        # Add trend line if requested
        if show_trends and len(days) > MIN_DATA_POINTS_FOR_TREND:
            z = np.polyfit(days, mom_values, 1)
            p = np.poly1d(z)
            trend_color = COLOR_PALETTE['positive'] if z[0] > 0 else COLOR_PALETTE['secondary']

            fig.add_trace(
                go.Scatter(
                    x=days,
                    y=p(days),
                    mode='lines',
                    name='Trend',
                    line=dict(color=trend_color, width=2, dash='dash'),
                ),
                row=1, col=2
            )

        fig.add_hline(
            y=0, line_dash="dash", line_color=COLOR_PALETTE['neutral'],
            opacity=0.7, row=1, col=2
        )

    # 3. Success Rates
    if win_rates:
        days = sorted(win_rates.keys())
        win_values = [win_rates[day] for day in days]

        # Color bars based on win rate thresholds
        bar_colors = []
        for w in win_values:
            if w >= WIN_RATE_THRESHOLD_HIGH:
                bar_colors.append(COLOR_PALETTE['positive'])
            elif w >= WIN_RATE_THRESHOLD_MID:
                bar_colors.append(COLOR_PALETTE['primary'])
            else:
                bar_colors.append(COLOR_PALETTE['secondary'])

        fig.add_trace(
            go.Bar(
                x=days,
                y=win_values,
                name='Win Rate',
                marker_color=bar_colors,
                opacity=0.8
            ),
            row=1, col=3
        )

        fig.add_hline(
            y=WIN_RATE_THRESHOLD_MID, line_dash="dash", line_color=COLOR_PALETTE['neutral'],
            opacity=0.7, row=1, col=3
        )

    # Update layout
    fig.update_layout(
        title={
            'text': f"{month_name} Analysis",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': FONT_SIZE_TITLE, 'color': COLOR_PALETTE['white']}
        },
        height=CHART_HEIGHT,
        showlegend=False,
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=COLOR_PALETTE['white'], size=FONT_SIZE_LABELS),
        margin=dict(l=60, r=60, t=60, b=60)
    )

    fig.update_annotations(font=dict(size=FONT_SIZE_ANNOTATION, color=COLOR_PALETTE['text_secondary']))

    # Update axis labels
    fig.update_xaxes(title_text="Trading Week", row=1, col=1, title_font=dict(size=FONT_SIZE_LABELS, color=COLOR_PALETTE['text_secondary']))
    fig.update_xaxes(title_text="Calendar Day", row=1, col=2, title_font=dict(size=FONT_SIZE_LABELS, color=COLOR_PALETTE['text_secondary']))
    fig.update_xaxes(title_text="Calendar Day", row=1, col=3, title_font=dict(size=FONT_SIZE_LABELS, color=COLOR_PALETTE['text_secondary']))

    fig.update_yaxes(title_text="Cumulative Return (%)", row=1, col=1, title_font=dict(size=FONT_SIZE_LABELS, color=COLOR_PALETTE['text_secondary']))
    fig.update_yaxes(title_text="Momentum (%)", row=1, col=2, title_font=dict(size=FONT_SIZE_LABELS, color=COLOR_PALETTE['text_secondary']))
    fig.update_yaxes(title_text="Win Rate (%)", row=1, col=3, title_font=dict(size=FONT_SIZE_LABELS, color=COLOR_PALETTE['text_secondary']))

    # Apply consistent styling
    update_common_axes_styling(fig, rows_cols=(1, SUBPLOT_COLS), palette=COLOR_PALETTE)

    return fig

def create_correlation_heatmap(corr_matrix: 'pd.DataFrame') -> go.Figure:
    """Create correlation heatmap for multiple tickers.
    
    Args:
        corr_matrix: Correlation matrix DataFrame (tickers x tickers)
        
    Returns:
        Plotly Figure (heatmap)
    """
    if corr_matrix is None or corr_matrix.empty:
        logger.warning("Empty correlation matrix provided")
        return _create_empty_figure("No correlation data available")
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu',
        zmid=0,
        zmin=-1,
        zmax=1,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text:.2f}',
        textfont={"size": 12},
        colorbar=dict(title="Correlation", thickness=15)
    ))
    
    fig.update_layout(
        title="Correlation Matrix (Daily Returns)",
        height=400 + len(corr_matrix) * 30,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=COLOR_PALETTE['white']),
        title_font_color=COLOR_PALETTE['white'],
        xaxis_title="",
        yaxis_title="",
    )
    
    return fig


def create_relative_performance_chart(cumulative_returns: 'pd.DataFrame') -> go.Figure:
    """Create line chart of cumulative returns for multiple tickers.
    
    Args:
        cumulative_returns: DataFrame with cumulative returns (%) per ticker
        
    Returns:
        Plotly Figure (line chart)
    """
    if cumulative_returns is None or cumulative_returns.empty:
        logger.warning("Empty cumulative returns provided")
        return _create_empty_figure("No performance data available")
    
    colors = [
        COLOR_PALETTE['primary'],
        COLOR_PALETTE['secondary'],
        COLOR_PALETTE['positive'],
        COLOR_PALETTE['negative'],
        COLOR_PALETTE['neutral'],
    ]
    
    fig = go.Figure()
    for i, ticker in enumerate(cumulative_returns.columns):
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=cumulative_returns.index,
            y=cumulative_returns[ticker],
            name=ticker,
            line=dict(color=color, width=LINE_WIDTH),
            hovertemplate=f"{ticker}: %{{y:.2f}}%<extra></extra>"
        ))
    
    fig.update_layout(
        title="Cumulative Returns Comparison",
        height=CHART_HEIGHT,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=COLOR_PALETTE['white']),
        title_font_color=COLOR_PALETTE['white'],
        xaxis_title="Date",
        yaxis_title="Cumulative Return (%)",
        xaxis_title_font=dict(size=FONT_SIZE_LABELS, color=COLOR_PALETTE['text_secondary']),
        yaxis_title_font=dict(size=FONT_SIZE_LABELS, color=COLOR_PALETTE['text_secondary']),
        hovermode='x unified',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(0,0,0,0.3)'
        )
    )
    
    fig.update_xaxes(gridcolor=COLOR_PALETTE['grid'], tickcolor=COLOR_PALETTE['text_secondary'])
    fig.update_yaxes(gridcolor=COLOR_PALETTE['grid'], tickcolor=COLOR_PALETTE['text_secondary'])
    
    return fig


def _create_empty_figure(message: str) -> go.Figure:
    """Create placeholder figure for empty/invalid data."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=16, color=COLOR_PALETTE['text_secondary']),
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis_visible=False,
        yaxis_visible=False,
        height=CHART_HEIGHT
    )
    return fig


def create_return_distribution_chart(
    returns: 'pd.Series',
    title: str = "Return Distribution",
    dist_info: dict = None
) -> go.Figure:
    """Create a histogram of returns with fitted distribution overlay and summary lines.
    Includes an inset plot focusing on the center of the distribution.

    Args:
        returns: pandas Series of returns (decimal, e.g., 0.01 for 1%)
        title: Chart title
        dist_info: dict from fit_distribution_to_returns() with best_dist, fitted_loc, fitted_scale, fitted_shape

    Returns:
        Plotly Figure with main chart and inset
    """
    import pandas as pd
    if returns is None or (isinstance(returns, pd.Series) and returns.dropna().empty):
        return _create_empty_figure("No return data available")

    r = returns.dropna().astype(float)
    mean = r.mean()
    median = r.median()
    std = r.std(ddof=1) if r.shape[0] > 1 else 0.0

    # Create figure
    fig = go.Figure()
    
    # ========== MAIN CHART ==========
    # Histogram with increased bins (70 bins for finer granularity)
    fig.add_trace(go.Histogram(
        x=r,
        nbinsx=70,
        histnorm='probability density',
        marker_color=COLOR_PALETTE['primary'],
        opacity=0.8,
        name='Histogram',
        xaxis='x1',
        yaxis='y1'
    ))

    # Fitted distribution curve for main chart
    if dist_info and std > 0:
        try:
            from scipy import stats
            # Make x-axis symmetric for better visualization
            x_range = max(abs(r.min()), abs(r.max()))
            x = np.linspace(-x_range, x_range, 300)
            dist_name = dist_info.get('best_dist', 'normal')
            loc = dist_info.get('fitted_loc', mean)
            scale = dist_info.get('fitted_scale', std)
            shape = dist_info.get('fitted_shape', {})

            # Try to use scipy's distribution dynamically
            pdf = None
            try:
                dist_obj = getattr(stats, dist_name)
                if shape:
                    # Pass shape parameters first, then loc and scale
                    pdf = dist_obj.pdf(x, **shape, loc=loc, scale=scale)
                else:
                    pdf = dist_obj.pdf(x, loc=loc, scale=scale)
            except:
                # Fallback to hardcoded known distributions
                if dist_name == 'normal':
                    pdf = stats.norm.pdf(x, loc=loc, scale=scale)
                elif dist_name == 't':
                    df = shape.get('df', 10) if shape else 10
                    pdf = stats.t.pdf(x, df=df, loc=loc, scale=scale)
                elif dist_name == 'laplace':
                    pdf = stats.laplace.pdf(x, loc=loc, scale=scale)
                elif dist_name == 'logistic':
                    pdf = stats.logistic.pdf(x, loc=loc, scale=scale)
                else:
                    pdf = stats.norm.pdf(x, loc=loc, scale=scale)

            fig.add_trace(go.Scatter(
                x=x, y=pdf, mode='lines',
                name=f'Fitted {dist_name.capitalize()}',
                line=dict(color=COLOR_PALETTE['secondary'], width=2),
                xaxis='x1',
                yaxis='y1'
            ))
        except Exception as e:
            logger.debug(f"Failed to plot fitted distribution: {e}")
    elif std > 0:
        # Fallback to normal if no dist_info with symmetric x-axis
        x_range = max(abs(r.min()), abs(r.max()))
        x = np.linspace(-x_range, x_range, 300)
        pdf = (1.0 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
        fig.add_trace(go.Scatter(
            x=x, y=pdf, mode='lines',
            name='Normal (default)',
            line=dict(color=COLOR_PALETTE['secondary'], width=2),
            xaxis='x1',
            yaxis='y1'
        ))

    # ========== INSET CHART ==========
    # Calculate range for inset - VERY TIGHT zoom on the center
    # Focus on a very narrow band to clearly distinguish 0 from mean
    inset_range = min(std * 0.75, abs(mean) * 3) if abs(mean) > 0.001 else std * 0.75
    inset_min = min(-inset_range, mean - inset_range/2)
    inset_max = max(inset_range, mean + inset_range/2)
    
    # Filter data for inset
    inset_data = r[(r >= inset_min) & (r <= inset_max)]
    
    if len(inset_data) > 0:
        # Inset histogram
        fig.add_trace(go.Histogram(
            x=inset_data,
            nbinsx=20,
            histnorm='probability density',
            marker_color=COLOR_PALETTE['primary'],
            opacity=0.8,
            name='Inset Histogram',
            xaxis='x2',
            yaxis='y2',
            showlegend=False
        ))
        
        # Add fitted distribution to inset
        if dist_info and std > 0:
            try:
                from scipy import stats
                x_inset = np.linspace(inset_min, inset_max, 150)
                dist_name = dist_info.get('best_dist', 'normal')
                loc = dist_info.get('fitted_loc', mean)
                scale = dist_info.get('fitted_scale', std)
                shape = dist_info.get('fitted_shape', {})

                # Try to use scipy's distribution dynamically
                pdf_inset = None
                try:
                    dist_obj = getattr(stats, dist_name)
                    if shape:
                        pdf_inset = dist_obj.pdf(x_inset, **shape, loc=loc, scale=scale)
                    else:
                        pdf_inset = dist_obj.pdf(x_inset, loc=loc, scale=scale)
                except:
                    # Fallback to hardcoded known distributions
                    if dist_name == 'normal':
                        pdf_inset = stats.norm.pdf(x_inset, loc=loc, scale=scale)
                    elif dist_name == 't':
                        df = shape.get('df', 10) if shape else 10
                        pdf_inset = stats.t.pdf(x_inset, df=df, loc=loc, scale=scale)
                    elif dist_name == 'laplace':
                        pdf_inset = stats.laplace.pdf(x_inset, loc=loc, scale=scale)
                    elif dist_name == 'logistic':
                        pdf_inset = stats.logistic.pdf(x_inset, loc=loc, scale=scale)
                    else:
                        pdf_inset = stats.norm.pdf(x_inset, loc=loc, scale=scale)

                fig.add_trace(go.Scatter(
                    x=x_inset, y=pdf_inset, mode='lines',
                    showlegend=False,
                    line=dict(color=COLOR_PALETTE['secondary'], width=2),
                    xaxis='x2',
                    yaxis='y2'
                ))
                
                # Get max y value for vertical lines
                max_y_inset = max(pdf_inset) * 1.1
            except Exception as e:
                logger.debug(f"Failed to plot inset fitted distribution: {e}")
                max_y_inset = 50  # fallback
        elif std > 0:
            # Fallback to normal
            x_inset = np.linspace(inset_min, inset_max, 150)
            pdf_inset = (1.0 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_inset - mean) / std) ** 2)
            fig.add_trace(go.Scatter(
                x=x_inset, y=pdf_inset, mode='lines',
                showlegend=False,
                line=dict(color=COLOR_PALETTE['secondary'], width=2),
                xaxis='x2',
                yaxis='y2'
            ))
            max_y_inset = max(pdf_inset) * 1.1
        else:
            max_y_inset = 50  # fallback
        
        # Add reference lines to inset using shapes (better approach)
        fig.add_shape(
            type="line",
            x0=0, x1=0,
            y0=0, y1=max_y_inset,
            line=dict(color='rgba(255,255,255,0.8)', width=2),
            xref='x2', yref='y2'
        )
        
        fig.add_shape(
            type="line",
            x0=mean, x1=mean,
            y0=0, y1=max_y_inset,
            line=dict(color=COLOR_PALETTE['positive'], width=2, dash='dash'),
            xref='x2', yref='y2'
        )

    # Update layout with main and secondary axes
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            font=dict(size=FONT_SIZE_TITLE, color=COLOR_PALETTE['white'])
        ),
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=COLOR_PALETTE['white']),
        height=CHART_HEIGHT,
        margin=dict(l=60, r=60, t=70, b=60),
        hovermode='x unified',
        # Main axes
        xaxis=dict(
            title='Return',
            tickformat='.2%',
            anchor='y1'
        ),
        yaxis=dict(
            title='Density',
            anchor='x1'
        ),
        # Secondary axes for inset (positioned top-right)
        xaxis2=dict(
            domain=[0.65, 0.95],
            anchor='y2',
            tickformat='.2%',
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(255,255,255,0.1)'
        ),
        yaxis2=dict(
            domain=[0.55, 0.95],
            anchor='x2',
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(255,255,255,0.1)'
        )
    )

    # Add border/background to inset area for visual distinction
    fig.add_shape(
        type='rect',
        xref='paper', yref='paper',
        x0=0.64, y0=0.54,
        x1=0.96, y1=0.96,
        line=dict(color=COLOR_PALETTE['secondary'], width=2),
        fillcolor='rgba(30,30,50,0.3)',
        layer='below'
    )

    return fig
