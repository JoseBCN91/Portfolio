import logging
from typing import Tuple
import plotly.express as px

logger = logging.getLogger(__name__)


def hex_to_rgba(hex_color: str, alpha: float = 0.2) -> str:
    """Convert hex color to rgba string."""
    try:
        rgb = px.colors.hex_to_rgb(hex_color)
        return f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{alpha})"
    except Exception as e:
        logger.warning(f"Failed to convert color {hex_color}: {e}, using default")
        return f"rgba(46,134,171,{alpha})"


def update_common_axes_styling(fig, rows_cols: Tuple[int, int] = (1, 3), palette: dict | None = None):
    """Apply consistent axis styling to all subplots."""
    if palette is None:
        palette = {
            'grid': 'rgba(255,255,255,0.1)',
            'text_secondary': '#B8BCC8',
            'border': 'rgba(255,255,255,0.2)'
        }
    rows, cols = rows_cols
    for i in range(1, cols + 1):
        fig.update_xaxes(
            gridcolor=palette['grid'],
            tickcolor=palette['text_secondary'],
            linecolor=palette['border'],
            tickfont=dict(color=palette['text_secondary']),
            row=1, col=i
        )
        fig.update_yaxes(
            gridcolor=palette['grid'],
            tickcolor=palette['text_secondary'],
            linecolor=palette['border'],
            tickfont=dict(color=palette['text_secondary']),
            row=1, col=i
        )
