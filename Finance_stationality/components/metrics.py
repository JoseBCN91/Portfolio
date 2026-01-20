"""Reusable UI components for metrics and styling."""

from typing import Optional
import streamlit as st
from helpers.config import (
    WIN_RATE_THRESHOLDS, WIN_RATE_COLORS, COLOR_TEXT, COLOR_TEXT_SECONDARY
)


def create_metric_card(
    value: str,
    label: str,
    color: Optional[str] = None,
    subtext: Optional[str] = None
) -> None:
    """Create and display a metric card in Streamlit.
    
    Args:
        value: The main metric value to display (e.g., "45.2%")
        label: The label describing the metric (e.g., "Average Return")
        color: Hex color for the value text (default: white)
        subtext: Optional secondary text below the value
    """
    # Only add inline color if explicitly provided. Allow CSS to control default colors
    style_value = f' style="color: {color};"' if color else ''
    subtext_html = ""

    if subtext:
        subtext_html = f'<div style="color: #4ECDC4; font-weight: 500; margin-top: 0.5rem;">{subtext}</div>'

    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-value"{style_value}>{value}</div>
        <div class="metric-label">{label}</div>
        {subtext_html}
    </div>
    """, unsafe_allow_html=True)


def create_metric_cards_row(cards: list[dict], num_columns: int = 3) -> None:
    """Create a row of metric cards.
    
    Args:
        cards: List of dicts with keys: value, label, color (optional), subtext (optional)
        num_columns: Number of columns in the row
    """
    cols = st.columns(num_columns)
    
    for idx, card_data in enumerate(cards):
        if idx >= len(cols):
            break
        
        with cols[idx]:
            create_metric_card(
                value=card_data['value'],
                label=card_data['label'],
                color=card_data.get('color'),
                subtext=card_data.get('subtext')
            )


def get_win_rate_color(win_rate: float) -> str:
    """Map win rate percentage to appropriate color.
    
    Args:
        win_rate: Win rate as a percentage (0-100)
        
    Returns:
        Hex color string
    """
    if win_rate >= WIN_RATE_THRESHOLDS['excellent']:
        return WIN_RATE_COLORS['excellent']
    elif win_rate >= WIN_RATE_THRESHOLDS['good']:
        return WIN_RATE_COLORS['good']
    else:
        return WIN_RATE_COLORS['poor']


def get_win_rate_status(win_rate: float) -> tuple[str, str]:
    """Get win rate status emoji and text.
    
    Args:
        win_rate: Win rate as a percentage (0-100)
        
    Returns:
        Tuple of (emoji_and_text, color)
    """
    color = get_win_rate_color(win_rate)
    
    if win_rate >= WIN_RATE_THRESHOLDS['excellent']:
        status = "🎯 Excellent"
    elif win_rate >= WIN_RATE_THRESHOLDS['good']:
        status = "👍 Good"
    else:
        status = "⚠️ Poor"
    
    return status, color


def get_return_color(value: float) -> str:
    """Map return value to color (positive = green, negative = red).
    
    Args:
        value: Return value (positive or negative)
        
    Returns:
        Hex color string
    """
    if value > 0:
        return "#00D4AA"  # Green
    elif value < 0:
        return "#FF6B6B"  # Red
    else:
        return COLOR_TEXT_SECONDARY  # Neutral


def create_insight_box(title: str, description: str, color: Optional[str] = None) -> None:
    """Create an insight box with title and description.
    
    Args:
        title: Bold title text
        description: Description or metric value
        color: Hex color for the title (optional)
    """
    color = color or COLOR_TEXT_SECONDARY

    # Use the shared metric-container class so hover effects and styling are consistent
    st.markdown(f"""
    <div class="metric-container">
        <strong style="color: {color};">{title}</strong>
        <div style="margin-top:0.5rem; color: #B8BCC8;">{description}</div>
    </div>
    """, unsafe_allow_html=True)
