import streamlit as st

def apply_style():
    """Apply page configuration and CSS styling."""
    st.set_page_config(
        page_title="Monthly Patterns Analysis | BQuant",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Minimal wrapper to keep the heavy CSS centralized
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    :root{
        --bg: rgba(0,0,0,0);
        --surface: rgba(255,255,255,0.04);
        --border: rgba(255,255,255,0.08);
        --muted: #B8BCC8;
        --text: #FFFFFF;
        --primary: #2E86AB;
        --accent: #4ECDC4;
        --positive: #06D6A0;
        --negative: #FF6B6B;
        --glass-blur: 8px;
    }
    .stApp { font-family: 'Inter', sans-serif; color: var(--text); }

    .main-header{
        font-size: clamp(2.4rem, 4.2vw, 3.4rem);
        line-height: 1.1;
        font-weight:800;
        letter-spacing: 0.02em;
        background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 60%, #A6FFEA 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align:center;
        margin: 0.25rem auto 0.5rem auto;
        padding: 0.25rem 0;
        text-shadow: 0 2px 18px rgba(78, 205, 196, 0.18);
        display: inline-block;
        width: 100%;
    }

    .main-header:after{
        content: "";
        display: block;
        width: 96px;
        height: 3px;
        margin: 10px auto 0 auto;
        background: linear-gradient(90deg, rgba(46,134,171,0.0) 0%, var(--accent) 50%, rgba(46,134,171,0.0) 100%);
        border-radius: 999px;
        filter: drop-shadow(0 2px 8px rgba(78,205,196,0.35));
    }

    .sub-header{
        font-size: 1.15rem;
        line-height: 1.45;
        color: rgba(184,188,200,0.95);
        text-align:center;
        margin: 0.35rem auto 1.35rem auto;
        font-weight: 400;
        max-width: 980px;
        letter-spacing: 0.01em;
    }

    .bquant-brand{
        position: fixed; bottom: 18px; right: 18px;
        background: linear-gradient(135deg, #FF6B6B 0%, var(--accent) 100%);
        color: var(--text); padding: 8px 14px; border-radius: 999px; font-size: 0.9rem; font-weight: 600; z-index:9999;
        box-shadow: 0 6px 18px rgba(0,0,0,0.45);
    }

    .metric-container{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.25rem;
        margin: 0.5rem 0;
        backdrop-filter: blur(var(--glass-blur));
        transition: transform 120ms ease, box-shadow 120ms ease;
    }

    .metric-container:hover{ transform: translateY(-4px); box-shadow: 0 10px 30px rgba(0,0,0,0.45); }

    .metric-value{ font-size:1.5rem; font-weight:700; color:var(--text); margin-bottom:0.25rem; }
    .metric-label{ font-size:0.85rem; color:var(--muted); font-weight:600; text-transform:uppercase; letter-spacing:0.02em; }

    .success-message{ background: linear-gradient(135deg, var(--positive) 0%, #00B894 100%); color: var(--text); padding: 0.9rem 1.25rem; border-radius: 8px; margin: 0.9rem 0; font-weight:600; }

    .explanation-box{ background: var(--surface); border: 1px solid var(--border); padding: 1.25rem; border-radius: 12px; margin:1rem 0; color: var(--text); }

    /* Plotly charts visual separation and card styling */
    .stPlotlyChart{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 10px 12px;
        margin: 10px 0 20px 0;
        box-shadow: 0 8px 24px rgba(0,0,0,0.18);
    }

    /* Add subtle inner spacing to charts on small screens */
    @media (max-width: 640px){
        .stPlotlyChart{ padding: 8px 10px; margin: 8px 0 16px 0; }
    }

    /* Responsive metric grid for small widths */
    @media (max-width: 640px){
        .metric-value{ font-size:1.25rem; }
        .main-header{ font-size:2rem; }
    }

    /* Streamlit specific tweaks to reduce padding on wide layout */
    .css-1y4p8pa { padding-top: 0 !important; }

    /* Ensure markdown text uses white by default */
    .stMarkdown, .stText, .st-ae { color: var(--text) !important; }
    
    /* Icon action panel */
    .icon-panel { display:flex; gap: 8px; align-items:center; margin-top: 8px; }
    .icon-button { background: rgba(255,255,255,0.03); border: 1px solid var(--border); padding: 10px 12px; border-radius: 10px; cursor: pointer; display:inline-flex; align-items:center; justify-content:center; font-size:1.05rem; transition: transform 120ms ease, box-shadow 120ms ease; }
    .icon-button:hover { transform: translateY(-3px); box-shadow: 0 8px 20px rgba(0,0,0,0.45); }
    </style>
    """, unsafe_allow_html=True)

    # Light theme overrides to ensure contrast when Streamlit uses a light background
    st.markdown("""
    <style>
    @media (prefers-color-scheme: light) {
        :root{
            --surface: rgba(255,255,255,0.96);
            --border: rgba(2,6,23,0.06);
            --muted: #475569;
            --text: #0f172a;
            --primary: #2E86AB;
            --accent: #4ECDC4;
            --positive: #06D6A0;
            --negative: #FF6B6B;
        }

        .metric-container{
            background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(250,250,250,0.95));
            border: 1px solid var(--border);
            box-shadow: 0 6px 18px rgba(2,6,23,0.04);
            color: var(--text) !important;
        }

        .explanation-box{
            background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(250,250,250,0.95));
            border: 1px solid var(--border);
            color: var(--text) !important;
        }

        /* Light theme chart card adjustments */
        .stPlotlyChart{
            background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(250,250,250,0.95));
            border: 1px solid var(--border);
            box-shadow: 0 8px 22px rgba(2,6,23,0.06);
        }

        .bquant-brand{ box-shadow: 0 6px 18px rgba(2,6,23,0.06); }
    }
    </style>
    """, unsafe_allow_html=True)
